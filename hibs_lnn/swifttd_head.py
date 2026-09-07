"""
V31.1: SwiftTD 式 per-feature step-size 优化 (Javed et al., RLC 2024)
=====================================================================
解决 V31.0 发现的问题: IDBD 融合的 β 通过 BPTT 外循环学 (Meta-SGD 风格),
meta 梯度 ~1e-6 学不动 (arXiv 2401.17401 Figure 4)。

SwiftTD 三件套 (全部在线, 无 BPTT) — 精确对应论文 Algorithm 1:
  1. per-feature step-size 优化: β[i] = e^{θ[i]}, IDBD 在线更新:
         θ[i] += κ·δ[i]·h[i]·φ[i]                    (meta 更新, 在 decay/重置之前)
         h[i] ← (1-β[i]·φ[i]²)⁺·h[i] + β[i]·δ[i]·φ[i]  (step-size 迹)
  2. overshoot bound: correction ratio τ = Σᵢ β[i]·φ[i]²,
         τ > η 时更新缩放 s = η/τ (η = 0.1, max correction ratio)
  3. step-size decay: τ > η 时 θ[i] += φ[i]²·ln(ϵ) (ϵ = 0.99 decay rate,
       论文 Algorithm 1: 对 θ 的绝对衰减, 无 β 因子; β ← β·ϵ^{φ²})
       且重置该组的迹 (论文 L524-526)

论文关键超参: η = 0.1, ϵ = 0.99, η_min = e^-15, β 裁剪到 [ln(η_min), ln(η)] → β_max = η。

本实现的关键适配 (记录在 wiki):
  τ 按输出行 (per-row) 计算。论文的 correction ratio 定义在单个标量预测
  上 (v = w·φ, 更新后 v' = v + s·β·δ·Σφ²); 我们的 head 有 n2 = 1616 个
  标量输出 (8 响应 × 202 vocab), 每个输出是独立的预测、独立的误差。
  w2 的行 u 有它自己的 τ_u = Σₖ β[u,k]·xₖ² 和缩放 s_u — 这是论文
  "单预测 bound" 对多输出层的直接推广。若用全局 τ (早期版本), 1616 行
  复制结构使 τ 被维度主导 → bound 永远触发 → 更新塌缩到 ~0 (实测失败)。
  同样地, w1 按 d 行, b1/b2 是向量 (单一 τ)。

  没有特征归一化: raw 特征保持 IDBD 元梯度 O(1) 尺度 (归一化会把
  per-feature 元更新稀释到 1/N, 40 步内 β 纹丝不动, 实测失败)。

行为: β ≤ η 且 τ ≈ β̄·Σφ² ≫ η (raw 特征) → bound 常态触发 → 每行每步
修正恰好 η·errs (10%), decay 把 β 压到 τ≈η 的预算, IDBD 在预算内把
步长分配给有用特征 (const 特征 → β_min, 相关特征 → 维持)。这正是论文
Section 6 描述的收敛行为。

应用对象: PLN (2 层 MLP head)。每组用"入边激活"作为 IDBD 特征 φ:
  w1 (d,d)  : φ = hidden[j]      (SSM 输出), δ = 回传误差 g[i]
  b1 (d,)   : φ = 1, δ = g[i]
  w2 (n2,d) : φ = x[i]           (tanh 激活), δ = errs[u]
  b2 (n2,)  : φ = 1, δ = errs[u]
其中 errs = p - onehot 是 CE 对 logits 的残差 (梯度)。
"""

import math
import torch
import torch.nn.functional as F


class SwiftTDHead:
    """
    2 层 MLP head 的 SwiftTD 步长状态机 (V31.1)。

    参数布局 (与 V31 head_forward 一致):
      W = [w1 (d,d), b1 (d,), w2 (n2,d), b2 (n2,)],  n2 = vocab*n_resp
      x = tanh(hidden @ w1^T + b1);  logits = x @ w2^T + b2 → (n_resp, vocab)

    状态:
      theta: 每参数 log 步长 θ (对应 β = e^θ, 裁剪到 [β_min, β_max=η])
      h    : 每参数 step-size 迹 (IDBD 的 h 向量)

    超参 (论文默认):
      beta_init : 初始 β (论文 1e-7 长流; 短视野用 0.01)
      kappa     : IDBD meta step-size θ
      eta       : max correction ratio η = 0.1 (bound 阈值 = β 上限)
      eps       : decay rate ϵ = 0.99 (bound 触发时的衰减速率, 独立超参)
    """

    def __init__(self, d_model, world_vocab, n_resp,
                 beta_init=0.01, kappa=0.05, eta=0.1, eps=0.99,
                 beta_min=1e-7, input_dim=None, tau_norm=None):
        """
        input_dim (V32 意图注入用): w1 的输入维度。
        默认 input_dim=d_model → 与 V31 布局完全一致 (向后兼容)。
        d_model 始终是隐藏层宽度 (x 的维度, w2 的输入)。
        tau_norm (V35.2 诊断 A): 按有效维度归一 τ。
          None/False: 原版 τ = Σ β·φ² (绝对和)
          'mean'    : τ = Σ β·φ² / dim (per-dim 平均 — 修复 V32+ 3d 意图
                      注入后 384 维求和恒超 η → bound 常态触发 → h 迹归零
                      → IDBD 分化死亡, β_std 1.6e-05 的诊断根因)
        """
        self.d_model = d_model
        self.input_dim = input_dim if input_dim is not None else d_model
        self.world_vocab = world_vocab
        self.n_resp = n_resp
        self.n2 = world_vocab * n_resp
        self.tau_norm = tau_norm
        # 布局: [w1 (d, input_dim), b1 (d,), w2 (n2,d), b2 (n2,)]
        self.shapes = [(d_model, self.input_dim), (d_model,),
                       (self.n2, d_model), (self.n2,)]
        self.beta_init = beta_init
        self.kappa = kappa
        self.eta = eta              # max correction ratio (bound 阈值)
        self.eps = eps              # decay rate (独立超参!)
        self.beta_max = eta         # 论文: β 裁剪上限 = ln(η)
        self.beta_min = beta_min
        self.reset_state()

    # ── 状态管理 ─────────────────────────────────────────────
    def reset_state(self):
        self.theta = [torch.full(s, math.log(self.beta_init)) for s in self.shapes]
        self.h = [torch.zeros(s) for s in self.shapes]
        self.n_bound_triggers = 0
        self.n_decays = 0
        self.n_steps = 0

    def to(self, device):
        """搬移 SwiftTD 状态 (θ/h 迹) 到 device (head 是普通类, 非 nn.Module)."""
        self.theta = [t.to(device) for t in self.theta]
        self.h = [t.to(device) for t in self.h]
        return self

    def cpu(self):
        return self.to('cpu')

    def beta(self):
        return [t.exp().clamp(self.beta_min, self.beta_max) for t in self.theta]

    def step_size_stats(self):
        b = torch.cat([x.reshape(-1) for x in self.beta()])
        return {
            'n_params': int(b.numel()),
            'beta_mean': float(b.mean()),
            'beta_std': float(b.std()),
            'beta_min': float(b.min()),
            'beta_max': float(b.max()),
            'frac_clipped_min': float(
                (b <= self.beta_min * (1 + 1e-3)).float().mean()),
            'frac_clipped_max': float(
                (b >= self.beta_max * (1 - 1e-3)).float().mean()),
            'n_bound_triggers': self.n_bound_triggers,
            'n_decays': self.n_decays,
            'n_steps': self.n_steps,
        }

    def state(self):
        """保存当前状态 (θ/h 迹), 用于 sleep 前后快照。"""
        return ([t.clone() for t in self.theta], [t.clone() for t in self.h])

    def load_state(self, st):
        theta, h = st
        for t, v in zip(self.theta, theta):
            t.copy_(v)
        for t, v in zip(self.h, h):
            t.copy_(v)

    # ── 核心: 单样本在线一步 ─────────────────────────────────
    def step(self, W, hidden, x, errs):
        """
        对一个样本做一步 SwiftTD 更新 (原地修改 W, 无计算图)。
        W     : [w1,b1,w2,b2] 当前参数
        hidden: SSM 输出 (d,)
        x     : tanh 激活 (d,)
        errs  : (n2,) = p - onehot (CE 对 logits 的残差)
        """
        newW = self._rates(W, hidden, x, errs, graph=False)
        for old, new in zip(W, newW):
            old.copy_(new)
        return W

    def step_graph(self, W, hidden, x, errs):
        """
        可微内循环一步 (V31.2 BPTT 外循环用):
        与 step() 相同的 SwiftTD 速率 (s, β 全 detached),
        但权重更新保持计算图 → 外循环梯度穿过内循环链
        (RLN 特征 φ → 误差 δ → w 更新 → query loss)。
        """
        return self._rates(W, hidden, x, errs, graph=True)

    def _rates(self, W, hidden, x, errs, graph):
        """
        SwiftTD 速率计算 + 权重更新 (graph=False 原地, graph=True 新张量)。
        s 与 β 为 detached 速率 (bound/decay/clip 不可微, 论文 BPTT 不可用
        → 直通: 速率当作给定常数, 只对 w 链与特征链反传)。
        """
        w1, b1, w2, b2 = W
        d, din, n2 = self.d_model, self.input_dim, self.n2

        # 各组特征 φ 与误差 δ (IDBD 逐元素形式)
        f_w1 = hidden.unsqueeze(0).expand(d, self.input_dim).contiguous()
        f_b1 = torch.ones(d)
        f_w2 = x.unsqueeze(0).expand(n2, d).contiguous()        # (n2,d)
        f_b2 = torch.ones(n2)

        d_w1 = (w2.t() @ errs) * (1.0 - x * x)                  # (d,) 回传误差
        d_w1 = d_w1.unsqueeze(1).expand(d, self.input_dim).contiguous()
        d_b1 = d_w1[:, 0]
        d_w2 = errs.unsqueeze(1).expand(n2, d).contiguous()     # (n2,d)
        d_b2 = errs

        beta = self.beta()
        ln_eps = math.log(self.eps)  # ϵ<1 → 负; 论文 Algorithm 1: β += φ²·ln(ϵ)

        def _group(w, f, dg, gi, row_tau):
            """返回 (更新后的 w, 速率 s·β 用于图形更新)。"""
            b = beta[gi]
            # 1) meta-θ 更新 (先于 decay/reset — 论文顺序)
            self.theta[gi].add_(self.kappa * dg.detach() * self.h[gi] * f.detach())
            # 2) 速率: τ 与缩放 s (detached)
            if row_tau:
                tau = (b * f.detach() * f.detach()).sum(dim=1)
                if self.tau_norm == 'mean':
                    tau = tau / f.shape[1]     # per-dim 平均 (诊断 A)
            else:
                tau = (b * f.detach() * f.detach()).sum()
                if self.tau_norm == 'mean' and f.dim() == 1:
                    tau = tau / f.shape[0]     # b1/b2 向量: 按元素数归一
            s = torch.ones_like(tau)
            trig = tau > self.eta
            if bool(trig.any()):
                s = torch.where(trig, self.eta / tau, s)
                # 3) decay + 迹重置 (仅触发行)
                self.theta[gi][trig].add_(
                    ln_eps * (f.detach() * f.detach())[trig])
                self.h[gi][trig].zero_()
                self.n_bound_triggers += int(trig.sum())
                self.n_decays += int(trig.sum())
            # 4) h 更新 + clip θ → β ∈ [β_min, η] (论文 Section 6)
            self.h[gi].mul_(torch.clamp(
                1.0 - b * f.detach() * f.detach(), min=0.0))
            self.h[gi].add_(b * dg.detach() * f.detach())
            self.theta[gi].clamp_(math.log(self.beta_min),
                                  math.log(self.beta_max))
            # 速率 (detached): s·β, 广播到 f 形状
            if f.dim() == 2:
                s = s.unsqueeze(1)
            rate = (s * b).detach()
            if graph:
                return w - rate * dg * f, rate
            return w - rate * dg * f, rate

        w1n, _ = _group(w1, f_w1, d_w1, 0, row_tau=True)
        b1n, _ = _group(b1, f_b1, d_b1, 1, row_tau=False)
        w2n, _ = _group(w2, f_w2, d_w2, 2, row_tau=True)
        b2n, _ = _group(b2, f_b2, d_b2, 3, row_tau=False)

        self.n_steps += 1
        return [w1n, b1n, w2n, b2n]

    # ── 便捷封装: 单样本适应 ─────────────────────────────────
    def adapt_sample(self, W, hidden, tgt, V):
        """hidden: (1,d) SSM 输出; tgt: (n_resp,) 目标 token。"""
        d = self.d_model
        x = torch.tanh(hidden @ W[0].t() + W[1])
        logits = x @ W[2].t() + W[3]                            # (1, n2)
        logits = logits.view(1, self.n_resp, self.world_vocab)
        p = F.softmax(logits, dim=-1)
        onehot = F.one_hot(tgt, self.world_vocab).float()
        errs = (p - onehot).reshape(-1)                         # (n2,)
        self.step(W, hidden[0], x[0], errs.detach())
        return logits


# ============================================================
# 自测 1: 论文 weight-flipping 问题 (arXiv 2401.17401 §3)
# 20 维特征, 前 15 维权重恒为 0 (该学低步长), 后 5 维每 20 步翻转 ±1
# (该学高步长)。IDBD 应学到 per-feature 步长分化, 且性能好于标量 SGD。
# 单行 = 单预测, 与模块 row_tau=True 逻辑一致。
# ============================================================
def _idbd_linear_step(w, theta, h, phi, delta, kappa, eta, eps,
                      beta_min, beta_max):
    """单行线性预测的 SwiftTD 一步 (与模块逻辑一致, 供自测复用)。"""
    b = theta.exp().clamp(beta_min, beta_max)
    # 1) meta-θ 更新 (先于 decay/reset — 论文顺序)
    theta.add_(kappa * delta * h * phi)
    # 2) 权重更新 (bound 缩放)
    tau = (b * phi * phi).sum()
    s = 1.0
    decayed = False
    if tau > eta:
        s = eta / tau
        # 3) decay (绝对衰减, 无 β 因子) + 迹重置
        theta.add_(math.log(eps) * phi * phi)
        h.zero_()
        decayed = True
    w.add_(s * b * delta * phi)
    # 4) h 更新
    h.mul_(torch.clamp(1.0 - b * phi * phi, min=0.0))
    h.add_(b * delta * phi)
    theta.clamp_(math.log(beta_min), math.log(beta_max))
    return w, theta, h, decayed


def _self_test_weight_flipping(seed=42, n_steps=50000, kappa=0.05,
                               beta_init=0.1, verbose=True):
    torch.manual_seed(seed)
    d = 20
    w_true = torch.zeros(d)
    w_true[15:] = 1.0

    def run_idbd():
        # 快速追踪域 (论文 Figure 5 探索范围): η=0.5, β_max=1.0
        # 翻转每 20 步一次 → 每步需 ~100% 修正 → β 上限需 > η。
        # MLP head 的收敛域 (β_max=η=0.1) 由自测 3 覆盖。
        beta_min, beta_max, eta, eps = 1e-5, 1.0, 0.5, 0.99
        theta = torch.full((d,), math.log(beta_init))
        h = torch.zeros(d)
        w = torch.zeros(d)
        errs = []
        for t in range(n_steps):
            if t % 20 == 0:
                idx = 15 + (t // 20) % 5
                w_true[idx] *= -1
            phi = torch.randn(d)
            y = (w_true * phi).sum()
            delta = y - (w * phi).sum()
            w, theta, h, _ = _idbd_linear_step(
                w, theta, h, phi, delta, kappa, eta, eps, beta_min, beta_max)
            errs.append(delta.item() ** 2)
        return sum(errs) / len(errs), theta.exp()

    def run_scalar(alpha):
        w = torch.zeros(d)
        errs = []
        for t in range(n_steps):
            if t % 20 == 0:
                idx = 15 + (t // 20) % 5
                w_true[idx] *= -1
            phi = torch.randn(d)
            y = (w_true * phi).sum()
            delta = y - (w * phi).sum()
            w.add_(alpha * delta * phi)
            errs.append(delta.item() ** 2)
        return sum(errs) / len(errs)

    mse_idbd, beta_end = run_idbd()
    b_hi = beta_end[15:].mean().item()
    b_lo = beta_end[:15].mean().item()
    # 标量基线: 在非发散区间内扫 α, 取最优
    mse_scalar = min(run_scalar(a) for a in (0.001, 0.005, 0.01, 0.02, 0.05))

    if verbose:
        print(f"  [weight-flipping] MSE: idbd={mse_idbd:.5f} "
              f"scalar_best={mse_scalar:.5f}")
        print(f"    β 均值: const dims={b_lo:.5f} (应低), "
              f"flip dims={b_hi:.5f} (应高), 比值={b_hi / max(b_lo, 1e-9):.1f}x")

    ok = (mse_idbd <= mse_scalar * 1.05 and b_hi > b_lo * 3)
    print(f"  {'OK' if ok else 'FAIL'} IDBD 学到 per-feature 步长 "
          f"(误差更低 + 高/低步长分化)")
    return ok


# ============================================================
# 自测 2: 三件套在 β_init 过大时不发散 (bound + decay + 迹重置)
# ============================================================
def _self_test_bound_stability(seed=42, n_steps=20000, beta_init=0.5,
                               verbose=True):
    torch.manual_seed(seed)
    d = 20
    w_true = torch.zeros(d)
    w_true[15:] = 1.0
    kappa, eta, eps, beta_min, beta_max = 0.05, 0.5, 0.99, 1e-5, 1.0

    def run(alpha, use_swift=False):
        theta = torch.full((d,), math.log(beta_init))
        h = torch.zeros(d)
        w = torch.zeros(d)
        errs = []
        for t in range(n_steps):
            if t % 20 == 0:
                idx = 15 + (t // 20) % 5
                w_true[idx] *= -1
            phi = torch.randn(d)
            y = (w_true * phi).sum()
            delta = y - (w * phi).sum()
            if use_swift:
                w, theta, h, _ = _idbd_linear_step(
                    w, theta, h, phi, delta, kappa, eta, eps,
                    beta_min, beta_max)
            else:
                w.add_(alpha * delta * phi)
            errs.append(delta.item() ** 2)
            if math.isnan(delta.item()):
                return float('inf')
        return sum(errs) / len(errs)

    mse_swift = run(0.0, use_swift=True)
    mse_scalar = run(beta_init)
    ok = math.isfinite(mse_swift) and mse_swift < mse_scalar
    if verbose:
        print(f"  [bound 稳定性] β_init={beta_init}: "
              f"scalar(α={beta_init}) MSE={mse_scalar:.5f} (应发散/爆炸), "
              f"SwiftTD MSE={mse_swift:.5f}")
    print(f"  {'OK' if ok else 'FAIL'} bound+decay 防止发散")
    return ok


# ============================================================
# 自测 3: MLP head 单步形状 + loss 下降 + β 分化
# ============================================================
def _self_test_mlp_head():
    torch.manual_seed(0)
    d, V, n_resp = 8, 20, 2
    head = SwiftTDHead(d, V, n_resp, beta_init=0.01)
    w1 = torch.randn(d, d) * 0.5
    b1 = torch.zeros(d)
    w2 = torch.randn(V * n_resp, d) * 0.5
    b2 = torch.zeros(V * n_resp)
    W = [w1, b1, w2, b2]
    hidden = torch.randn(1, d)
    tgt = torch.tensor([3, 17])

    def loss():
        x = torch.tanh(hidden @ w1.t() + b1)
        logits = (x @ w2.t() + b2).view(1, n_resp, V)
        return F.cross_entropy(logits.view(-1, V), tgt.view(-1))

    l0 = loss().item()
    for _ in range(30):
        x = torch.tanh(hidden @ w1.t() + b1)
        logits = (x @ w2.t() + b2).view(1, n_resp, V)
        p = F.softmax(logits, dim=-1)
        onehot = F.one_hot(tgt, V).float()
        errs = (p - onehot).reshape(-1)
        head.step(W, hidden[0], x[0], errs.detach())
    l1 = loss().item()
    stats = head.step_size_stats()
    ok = l1 < l0 and stats['beta_mean'] > 0
    print(f"  {'OK' if ok else 'FAIL'} MLP head 30 步: loss "
          f"{l0:.4f} → {l1:.4f}, β_mean={stats['beta_mean']:.4f}, "
          f"bound 触发={stats['n_bound_triggers']}, decay={stats['n_decays']}")
    return ok


def _self_test():
    ok1 = _self_test_weight_flipping()
    ok2 = _self_test_bound_stability()
    ok3 = _self_test_mlp_head()
    print(f"\n  {int(ok1) + int(ok2) + int(ok3)}/3 passed")
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    _self_test()
