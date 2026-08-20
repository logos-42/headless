"""V35 分形持续学习引擎 (Fractal Continual Learning Engine)

按 lm-principle 83 条机器验证定理重设计 (docs/wiki/v35-plan.md v2)。

科研结论 → 架构决策:
  T1/T2 (Fractal.lean prefix_allocation_optimal / connection_density_strict_anti)
      分形预算: 预算集中前缀, 浅层密集深层稀疏 (1-d)^(D-1)
  T3/T5 (ArchCompare.lean residual_no_collapse_n / lstm_memory_retention)
      残差门控: h' = h + gate(h) — 防坍缩 (距离 ≥ (1-c)^n) + 门控记忆 (α→1)
  T6-T9 (Hopfield.lean) 原子检索 = 能量下降: x_new = X·Softmax(β·Xᵀx)
      β 分岔相变 = 探索/利用旋钮 (训练 β≈3 混合 / 部署 β≥10 锁定)
  T10/T11 (CriticalPoint.lean) 自适应 K: 边际收益 Δ_k < λ ⟹ 提前终止
  T9 + v1 主动查询: 原子池采样组合 → 反事实输入 → 世界执行器真值 → 喂回

Cell (预注册, 同进程, 共享 Stage1):
  B   : V34.7 base (对照)
  F   : + FractalRLN (分形预算 + 残差门控)
  FH  : + HopfieldBank (能量检索替代滚动平均)
  FHA : + 自适应 K (临界点终止)
  FHAQ: + ActiveQueryGenerator (组合需求注入)
"""
import os, sys, time, json, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_v31_meta_learning import batch_pairs, head_forward, V31_RLN, V31_PLN
from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    V32_1_Learner, build_model_resp, make_regs_tasks, build_u_struct,
    build_t_struct, tome_fraction)
from run_v33_drl import V33_Learner, V33_3_Learner
from run_v34_1_gen import generator_signature, cycle_type, TRANS, IDENT, _apply
from run_v34_3_cpg import AtomRouteBank, N_GEN
from run_v34_7_compose import ComposeModule, V34_7_Learner
from hibs_lnn.swifttd_head import SwiftTDHead
from hibs_lnn.meta_rule_world import gen_task_data_split

_gen_cache = {}


# ============================================================
# V35.4: FractalRLN4 构建 (4 层分形 RLN + 可选分形 PLN)
# ============================================================
def build_fractal4_resp(V, n_response, s1_state=None, fractal_pln=False):
    """FractalRLN4 (等比宽度 128/102/81/64) + PLN.

    fractal_pln=False: PLN 隐藏宽 = RLN 顶层宽 (64), 输入 3×64 (意图注入).
    fractal_pln=True:  PLN 隐藏宽独立 = 96 (等比中间值), 3×64=192 → 96 → 808
      — PLN 自己的分形瓶颈 (输入宽, 隐藏窄), 由 V35_Learner 重建。
    s1_state 来自 run_v35_4_fractal4.py 的 FractalRLN4_LM (键带 rln. 前缀).
    """
    from run_v35_4_fractal4 import FractalRLN4
    rln = FractalRLN4(V + 2, d0=128, n_layers=4, D=1.5, use_gate=True)
    if s1_state is not None:
        own = rln.state_dict()
        stripped = {}
        for k, v in s1_state.items():
            kk = k
            if kk.startswith('rln.'):
                kk = kk[4:]          # FractalRLN4_LM 的 rln.rln.layers → rln.layers
            if kk.startswith('rln.'):
                kk = kk[4:]
            if kk in own and own[kk].shape == v.shape:
                stripped[kk] = v
        own.update(stripped)
        try:
            rln.load_state_dict(own)
        except Exception as e:
            print("F4 Stage1 加载部分失败(忽略):", e)
    top = rln.widths[-1]
    # 注意: V32_Learner 意图注入假设 pln.d_model == rln.d_model (V32 契约),
    # 故 build 时总是用 top; 分形 PLN 的独立隐藏宽 (96) 由 V35_Learner
    # 在 super().__init__ 之后重建 (predictor + step_beta + head).
    pln = V31_PLN(top, V + 2, n_response=n_response)
    pln.fractal_pln = fractal_pln
    pln.fractal_pln_h = 96 if fractal_pln else None
    return rln, pln


# ============================================================
# T1/T2/T3/T5: FractalRLN — 分形预算 + 残差门控
# ============================================================
class FractalRLN(nn.Module):
    """包装 V31_RLN (Stage1 可加载), 每层后加门控残差.

    分形预算: 第 i 层门控投影宽度 = max(8, d·(1-depth)^(D-1))
      浅层满宽 (预算集中前缀), 深层收窄 (连接密度幂律 T2).
    残差门控: y' = y + gate_i(y), gate 零初始化 → 加载 Stage1 后初始 ≡ 原网络
      (防坍缩 T3 + 门控记忆 T5: 门控可学, 保留/丢弃由梯度决定).
    """

    def __init__(self, base_rln, D=1.5):
        super().__init__()
        self.rln = base_rln
        self.embed = base_rln.embed
        self.d_model = base_rln.d_model
        self.vocab_size = base_rln.vocab_size
        n = len(base_rln.layers)
        d = self.d_model
        self.gates = nn.ModuleList()
        for i in range(n):
            depth = i / max(n - 1, 1)
            width = max(8, int(d * (1 - depth) ** (D - 1)))
            g = nn.Sequential(
                nn.Linear(d, width), nn.Tanh(), nn.Linear(width, d))
            with torch.no_grad():
                g[2].weight.zero_(); g[2].bias.zero_()
            self.gates.append(g)
        self.n_layers = n

    def forward(self, ids, cond=None):
        x = self.embed(ids)
        for i, layer in enumerate(self.rln.layers):
            y, _ = layer(x, cond=cond)
            x = y + self.gates[i](y)   # 门控残差 (零初始化 → Stage1 等价)
        return x  # (B, L, d)


# ============================================================
# T6-T9: HopfieldBank — 能量下降检索 (替代滚动平均)
# ============================================================
class HopfieldBank:
    """现代 Hopfield 检索: x_new = X·Softmax(β·Xᵀx).

    存储: per-generator 原型 (与 AtomRouteBank 同, 训练滚动平均).
    检索: 能量下降 (hopfield_update_eq_attn T6), β 控制锁定度
      (softmax_weight_concentration T9): 训练 β_train≈3 混合态探索,
      部署 β_test≥10 模式锁定 (hopfield_retrieval_error_bound T7).
    零可学习参数 (X 是存储矩阵).
    """

    def __init__(self, d, beta_train=3.0, beta_test=10.0, alpha=1.0):
        self.d = d
        self.beta_train = beta_train
        self.beta_test = beta_test
        self.alpha = alpha
        self.atoms = [None] * N_GEN
        self._count = [0] * N_GEN
        self.n_add = 0
        self.n_route = 0
        self.n_hit = 0
        self.beta_log = []

    def _norm(self, m):
        return m / (m.norm() + 1e-8)

    def add(self, m, sig):
        v = m.detach().float()
        for i in range(N_GEN):
            if sig[i]:
                c = self._count[i]
                if self.atoms[i] is None:
                    self.atoms[i] = self._norm(v)
                else:
                    self.atoms[i] = self._norm(
                        (self.atoms[i] * c + v) / (c + 1))
                self._count[i] = c + 1
        self.n_add += 1

    def route(self, m, sig, beta=None, detach_m=True):
        """能量下降检索: x_new = X·Softmax(β·Xᵀx) (X = 命中原子矩阵).

        detach_m=True (部署): query 停止梯度; False (训练): 保留图 —
        梯度经 softmax 检索权重回流到 m (检索进训练, V35.1 ②).
        """
        hit = [i for i in range(N_GEN) if sig[i] and self.atoms[i] is not None]
        if not hit:
            return m, False
        beta = beta or self.beta_test
        mq = m.detach().float() if detach_m else m.float()
        X = torch.stack([self.atoms[i] for i in hit])          # (k, d)
        w = F.softmax(beta * (X @ mq), dim=0)                  # (k,)
        comp = (w.unsqueeze(-1) * X).sum(0)                    # (d,)
        out = self.alpha * comp + (1 - self.alpha) * mq
        out = self._norm(out).to(m.dtype)
        self.n_route += 1
        self.n_hit += len(hit)
        self.beta_log.append((beta, w.max().item()))
        return out, True

    def stats(self):
        return {'n_gen': N_GEN, 'n_add': self.n_add, 'n_route': self.n_route,
                'n_hit': self.n_hit,
                'filled': sum(1 for a in self.atoms if a is not None),
                'beta_last': (self.beta_log[-1] if self.beta_log else None)}


# ============================================================
# 主动查询生成器 (v1 保留 + β 调制)
# ============================================================
def compose_swaps(gen_indices):
    """6 个生成元 (对换) 索引序列 → 复合排列. 2-3 个对换复合直指 c4."""
    p = IDENT
    for gi in gen_indices:
        p = _apply(p, TRANS[gi])
    return p


class ActiveQueryGenerator:
    """从已学原子采样组合 → 反事实输入 → 世界执行器真值 → 查询对.

    探索信号: ① 组合不确定性 (熵优先) ② 签名覆盖 (未填格位优先)
    ③ β 调制: 训练期低 β (混合探索), 组合广度由采样分布控制.
    真值源 = MetaRuleWorld 执行器 (gen_task_data_split, remap=组合 perm).
    V35.1 (③): 3 原子复合占比提高 (p_combo3 控制), 组合签名去重
    (避免反复生成同一 perm 的训练对).
    """

    def __init__(self, V, seed=42, n_support=8, n_query=8, max_combo=3,
                 p_combo3=0.6):
        self.V = V
        self.rng = random.Random(seed)
        self.n_support = n_support
        self.n_query = n_query
        self.max_combo = max_combo
        self.p_combo3 = p_combo3
        self.gen_history = []   # [(perm, sig, len)]

    def _pick_gens(self, bank):
        """采样 2..max_combo 个已学原子; p_combo3 概率取 3 原子 (直指 c4)."""
        filled = [i for i in range(N_GEN) if bank.atoms[i] is not None]
        if len(filled) < 2:
            return None
        if self.p_combo3 > 0 and len(filled) >= 3 and \
                self.rng.random() < self.p_combo3:
            k = min(3, self.max_combo)
        else:
            k = self.rng.randint(2, min(self.max_combo, len(filled)))
        gens = self.rng.sample(filled, k)
        return gens

    def generate(self, bank, n=4):
        """生成 n 个组合任务: {'perm', 'gen_sig', 'support', 'query'}."""
        out = []
        seen_sigs = set(h[1] for h in self.gen_history)
        for _ in range(n * 3):          # 尝试预算: 去重后可能不足 n
            gens = self._pick_gens(bank)
            if gens is None:
                break
            perm = compose_swaps(gens)
            sig, ln = generator_signature(perm)
            sig_t = tuple(sig) if sig else None
            if sig is None or sig_t in seen_sigs:
                continue
            seen_sigs.add(sig_t)
            seed = self.rng.randint(0, 10 ** 6)
            sup, qry = gen_task_data_split(
                None, self.V, n_support=self.n_support, n_query=self.n_query,
                seed=seed, remap=perm, stop_at_halt=True)
            out.append({'perm': perm, 'gen_sig': sig, 'gen_len': ln,
                        'support': sup, 'query': qry})
            self.gen_history.append((perm, sig_t, ln))
            if len(out) >= n:
                break
        return out


# ============================================================
# V35_Learner
# ============================================================
class V35_Learner(V34_7_Learner):
    """V34.7 + 分形 (fractal) / Hopfield 检索 (hopfield) /
    自适应 K (adaptive_k) / 主动查询 (active_query)."""

    def __init__(self, rln, pln, V, use_compose=False, alpha=1.0,
                 use_atoms=True, use_fractal=False, use_hopfield=False,
                 adaptive_k=False, active_query=False,
                 train_retrieval=False, aq_p3=0.6, aq_eval=False,
                 tau_norm=None, beta_init=None,
                 use_think=False, n_think=6, think_tau=0.01,
                 lambda_think=0.0, lambda_combo=0.0,
                 use_perm=False, lambda_perm=0.5, lambda_table=0.5,
                 beta_train=3.0, beta_test=10.0, D=1.5,
                 aq_n=4, aq_max_combo=3, k_lam=0.005, k_patience=3,
                 **kw):
        self._use_fractal = use_fractal
        self._use_hopfield = use_hopfield
        self._train_retrieval = train_retrieval
        self._aq_eval = aq_eval
        self._tau_norm = tau_norm
        self._beta_init = beta_init
        self._use_think = use_think
        self._n_think = n_think
        self._think_tau = think_tau
        self._lambda_think = lambda_think
        self._lambda_combo = lambda_combo
        self._use_perm = use_perm
        self._lambda_perm = lambda_perm
        self._lambda_table = lambda_table
        self._use_remap = kw.pop('use_remap', False)
        self._aq_curriculum = kw.pop('aq_curriculum', False)
        self._adaptive_k = adaptive_k
        self._active_query = active_query
        self._D = D
        self._k_lam = k_lam
        self._k_patience = k_patience
        self._aq_n = aq_n
        self._aq_max_combo = aq_max_combo
        self._p_combo3 = kw.pop('p_combo3', 0.6)   # 先弹出, 不传给上层 **kw
        self._aq_rng = random.Random(seed if 'seed' not in kw else kw['seed'])

        # 分形包装 (Stage1 后加门控, 零初始化 → 等价)
        if use_fractal:
            rln = FractalRLN(rln, D=D)
        self._fractal_gate_params = (list(rln.gates.parameters())
                                     if use_fractal else [])

        # 原子银行: Hopfield (能量检索) 或 AtomRouteBank (滚动平均)
        bank = None
        if use_hopfield:
            bank = HopfieldBank(rln.d_model, beta_train=beta_train,
                                beta_test=beta_test, alpha=alpha)

        super().__init__(rln, pln, V, use_compose=use_compose, alpha=alpha,
                         use_atoms=use_atoms, **kw)
        # 学习效率修复 (V35.2 诊断 A): τ 按有效维度归一 + β_init 提高
        # (IDBD 元梯度 ∝ β_init: 0.001 → 0.05 恢复 V31.1 分化配置)
        if tau_norm or beta_init:
            hd = self.head
            bi = beta_init if beta_init is not None else hd.beta_init
            self.head = SwiftTDHead(
                self.d_model, self.world_vocab, self.n_resp,
                beta_init=bi, kappa=hd.kappa, eta=hd.eta,
                eps=hd.eps, beta_min=hd.beta_min,
                input_dim=self.intent_dim,
                tau_norm=tau_norm or hd.tau_norm)
            self.head.reset_state()
        # V35.4 分形 PLN: 重建 predictor[0] (3d→96) + step_beta + head
        fpln = getattr(pln, 'fractal_pln', False)
        fpln_h = getattr(pln, 'fractal_pln_h', None)
        if fpln and fpln_h:
            d = rln.d_model                      # 64 (顶层宽)
            ph = fpln_h                          # 96 (分形 PLN 隐藏宽)
            # 1) predictor[0]: V32 已注入为 Linear(3d, d) → 扩到 (3d, ph)
            old0 = pln.predictor[0]
            new0 = nn.Linear(d * 3, ph)
            with torch.no_grad():
                k = min(old0.weight.shape[0], ph)
                new0.weight[:k].copy_(old0.weight[:k])
                new0.bias[:k].copy_(old0.bias[:k])
            pln.predictor[0] = new0
            # 2) predictor[2]: (vocab*n_resp, d) → (vocab*n_resp, ph)
            old2 = pln.predictor[2]
            new2 = nn.Linear(ph, old2.out_features)
            with torch.no_grad():
                k = min(old2.weight.shape[1], ph)
                new2.weight[:, :k].copy_(old2.weight[:, :k])
                new2.bias.copy_(old2.bias)
            pln.predictor[2] = new2
            # 3) step_beta 参数数随 predictor 变化 → 重建
            pln.step_beta = nn.Parameter(
                torch.full((pln._n_params(),),
                           math.log(pln.step_beta.exp().mean().item())))
            # 4) head: SwiftTDHead(d_model=ph, input_dim=3d)
            hd = self.head
            self.head = SwiftTDHead(
                ph, self.world_vocab, self.n_resp,
                beta_init=hd.beta_init, kappa=hd.kappa, eta=hd.eta,
                eps=hd.eps, beta_min=hd.beta_min,
                input_dim=self.intent_dim,
                tau_norm=hd.tau_norm)
            self.head.reset_state()
        # super().__init__ 会建 self.bank (AtomRouteBank); 覆盖为 Hopfield
        if bank is not None:
            self.bank = bank
        # V35.5 潜空间思考: 多轮思考网络 (输入 [m; g; h_agg], 比 V33 的
        # [m; g] 多上下文 h; 输出 m 修正量). 零初始化输出 → Stage1 等价.
        if use_think:
            d = rln.d_model
            self.think_net = nn.Sequential(
                nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, d))
            with torch.no_grad():
                self.think_net[2].weight.zero_()
                self.think_net[2].bias.zero_()
            # 纳入 outer_opt (V33 outer_opt 只含 refine/probe/intent)
            base = list(self.outer_opt.param_groups[0]['params'])
            self.outer_opt = torch.optim.Adam(
                base + list(self.think_net.parameters()), lr=self.outer_lr)
            self.think_trace = []    # 每轮检验误差轨迹
            self.think_rounds = []   # 实际思考轮数
        else:
            self.think_net = None
            self.think_trace = []
            self.think_rounds = []
        # V35.9: 置换矩阵复合算子 (S4 忠实表示, 群同态)
        # 24 个 perm 各学一个 4×4 矩阵, 复合 = 矩阵乘法 (精确代数,
        # 未见组合自动正确) — 替代 Embedding 查表 (无归纳偏置).
        if use_perm:
            d = rln.d_model
            self.perm_head = nn.Sequential(
                nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 24))
            # 矩阵复合: P(a)@P(b) → 目标 perm logits
            # mat: (24, 4, 4) — 每个 perm 一个 4×4 矩阵嵌入
            self.perm_mats = nn.Parameter(torch.randn(24, 4, 4) * 0.1)
            base = list(self.outer_opt.param_groups[0]['params'])
            self.outer_opt = torch.optim.Adam(
                base + list(self.perm_head.parameters())
                + [self.perm_mats],
                lr=self.outer_lr)
            self.perm_acc = {'seen': [], 'unseen': []}
            self.compose_acc = []
        else:
            self.perm_head = None
            self.compose_table = None
            self.perm_acc = {'seen': [], 'unseen': []}
            self.compose_acc = []
        self._aq = (ActiveQueryGenerator(V, seed=42,
                                         n_support=aq_n // 2 + 2,
                                         n_query=aq_n // 2 + 2,
                                         max_combo=aq_max_combo,
                                         p_combo3=aq_p3 if active_query
                                         else self._p_combo3)
                    if active_query else None)
        # 分形门控参数已含在 super 的 rln.parameters() (FractalRLN 包装了 base)
        self._k_used = []

    # ── V35.5/35.6 潜空间思考循环 (E-V 闭环内部化) ──────────
    def _think(self, m, task):
        """多轮思考: 每轮用当前假设 m 重读 support → 检验误差 g →
        思考网络修正 m. 完整 BPTT (梯度穿多轮循环链到 RLN/think_net).
        收敛: 检验误差不再降 (Δ < τ, T10 临界点) 提前停.
        V35.6 (目标错位修复): think_net 的监督 = 思考后检验误差
        (共享头 W0 重读 support 的 CE, 存 self._think_loss 由
        _post_define_aux_loss 接入 loss) — 学"修正假设"而非"讨好训练".
        返回 m (思考后假设, 带图)."""
        W = [w.detach() for w in self.pln.clone_params()]
        sup = task['support'][:self.refine_sup]
        if not m.requires_grad:
            # 部署/no_grad 路径: enable_grad 建局部图 (思考仍修正假设),
            # 返回 detach 结果
            with torch.enable_grad():
                m2 = m.detach().clone().requires_grad_(True)
                m2 = self._think_loop(m2, W, sup, build_graph=False)
                return m2.detach()
        return self._think_loop(m.clone(), W, sup, build_graph=True)

    def _think_loop(self, m, W, sup, build_graph):
        rounds = 0
        trace = []
        think_loss = torch.tensor(0.0, device=m.device)
        # V35.7: 组合检验 — 用 AQ 生成的 unseen 组合任务 (仅训练期)
        combo_sup = None
        if build_graph and self._aq is not None and self._lambda_combo > 0:
            combos = self._aq.generate(self.bank, n=1)
            if combos and combos[0]['support']:
                # AQ 任务 tgt 是 8 tokens, n_resp=4 → 切片
                combo_sup = [(c, tg[:4]) for (c, tg)
                             in combos[0]['support'][:2]]
                if not combo_sup:
                    combo_sup = None
        for t in range(self._n_think):
            g = torch.zeros_like(m)
            h_agg = torch.zeros_like(m)
            e = 0.0
            for (c, tg) in sup:
                h = self._rln_fwd(c.unsqueeze(0), m)[:, -1, :]
                h_agg = h_agg + h.detach()[0]
                hin = self._h_in_batch(h, m)
                pred = head_forward(W, hin, self.intent_dim,
                                    self.world_vocab, self.n_resp)
                loss = F.cross_entropy(
                    pred.view(-1, self.world_vocab),
                    tg.view(-1).clamp(0, self.V + 1))
                e += loss.item()
                gi = torch.autograd.grad(loss, m, create_graph=build_graph)[0]
                g = g + gi
            h_agg = h_agg / len(sup)
            corr = self.think_net(torch.cat([m, g, h_agg], dim=-1))
            m = self.m_norm(m + corr)
            trace.append(e)
            rounds += 1
            # V35.6: 思考后检验误差 (修正后的 m 在 support 上的 CE, 带图)
            # 由 _post_define_aux_loss 返回, V33.meta_step 用 lambda_icl 加权
            if build_graph and self.lambda_icl > 0:
                e_after = 0.0
                for (c, tg) in sup:
                    h2 = self._rln_fwd(c.unsqueeze(0), m)[:, -1, :]
                    hin2 = self._h_in_batch(h2, m)
                    p2 = head_forward(W, hin2, self.intent_dim,
                                      self.world_vocab, self.n_resp)
                    e_after = e_after + F.cross_entropy(
                        p2.view(-1, self.world_vocab),
                        tg.view(-1).clamp(0, self.V + 1))
                think_loss = think_loss + e_after / len(sup)
            # V35.7: 组合检验误差 (思考后 m 对 unseen 组合任务的重读误差)
            if build_graph and combo_sup is not None:
                e_combo = 0.0
                for (c, tg) in combo_sup:
                    h3 = self._rln_fwd(c.unsqueeze(0), m)[:, -1, :]
                    hin3 = self._h_in_batch(h3, m)
                    p3 = head_forward(W, hin3, self.intent_dim,
                                      self.world_vocab, self.n_resp)
                    e_combo = e_combo + F.cross_entropy(
                        p3.view(-1, self.world_vocab),
                        tg.view(-1).clamp(0, self.V + 1))
                think_loss = think_loss + (self._lambda_combo
                                           * e_combo / len(combo_sup))
            # 收敛: 误差不再降 (T10)
            if t >= 2 and trace[-2] - e < self._think_tau:
                break
        self.think_trace.append(trace)
        self.think_rounds.append(rounds)
        self._think_loss = think_loss   # 直接监督 (V35.6) + 组合 (V35.7)
        return m

    def _post_define_aux_loss(self, m, t):
        """V35.6: 思考监督接入 — 返回思考后检验误差 (共享头 W0 重读
        support), 由 V33.meta_step 用 lambda_icl 加权 (TH2 复用
        lambda_icl 作为思考监督权重, 不再乘 lambda_think).
        思考分支跳过 super() (避免 ICL 偏置混入).
        V35.8: 叠加 perm 回读损失 (PermHead) + 复合表三元组损失."""
        extra = None
        if self._use_think and self.think_net is not None:
            tl = getattr(self, '_think_loss', None)
            if tl is not None:
                self._think_loss = None   # 单次消费
                extra = tl
        if self._use_perm and self.perm_head is not None and m is not None:
            perm_loss = self._perm_loss(m, t)
            if perm_loss is not None:
                pl = perm_loss * self._lambda_perm
                extra = pl if extra is None else extra + pl
            if self._lambda_table > 0:
                tl2 = self._table_loss()
                if tl2 is not None:
                    extra = (tl2 * self._lambda_table
                             if extra is None else extra + tl2 * self._lambda_table)
        if extra is not None:
            return extra
        return super()._post_define_aux_loss(m, t)

    def _perm_loss(self, m, t):
        """V35.8: latent 回读 — m → 24 类 S4 分类, 监督 = 任务真实 perm.
        用带图 m (不 detach): 回读误差反传 RLN, 推动 m 编码置换结构."""
        import itertools
        perm = t.get('perm')
        if perm is None:
            return None
        idx = list(itertools.permutations(range(4))).index(tuple(perm))
        logits = self.perm_head(m)
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([idx], device=m.device))
        # 记录分类准确率 (seen/unseen 分桶)
        with torch.no_grad():
            pred = logits.argmax().item()
            key = 'seen' if t.get('seen') else 'unseen'
            self.perm_acc[key].append(1 if pred == idx else 0)
        return loss

    def _table_loss(self):
        """V35.9: S4 群乘法 — 置换矩阵复合.

        忠实表示: 每个 perm 一个 4×4 矩阵 P(i). 复合 P(a)@P(b) 应对应
        P(a∘b). 监督 = 三重损失: ① 复合后矩阵与真实置换矩阵的 MSE
        (代数对齐) ② 复合矩阵对 24 类的 logits (分类, 辅助) ③ 置换性
        正则 (每行/列近 one-hot). 矩阵乘法是精确代数 → 未见组合正确.
        """
        import itertools
        import random
        perms = list(itertools.permutations(range(4)))
        self._perm_list = perms
        # 真实置换矩阵 (24,4,4): P[i][r][c] = 1 if perms[i](r)=c
        P_true = torch.zeros(24, 4, 4)
        for i, p in enumerate(perms):
            for r in range(4):
                P_true[i, r, p[r]] = 1.0
        dev = self.perm_mats.device
        P_true = P_true.to(dev)
        idxs = list(range(24))
        step = getattr(self, '_table_step', 0)
        self._table_step = step + 1
        rng = random.Random(42 + step)
        pairs = [(rng.choice(idxs), rng.choice(idxs)) for _ in range(64)]
        def compose(p, q):  # p∘q = p(q(x))
            return tuple(p[q[x]] for x in range(4))
        a_idx = torch.tensor([a for a, _ in pairs], device=dev)
        b_idx = torch.tensor([b for _, b in pairs], device=dev)
        ab_idx = torch.tensor([perms.index(compose(perms[a], perms[b]))
                               for a, b in pairs], device=dev)
        Pa = self.perm_mats[a_idx]            # (B,4,4)
        Pb = self.perm_mats[b_idx]
        Pab = Pa @ Pb                          # (B,4,4) 复合
        # ① 代数对齐: 复合矩阵 → 真实置换矩阵
        loss = F.mse_loss(Pab, P_true[ab_idx])
        # ② 分类辅助: 复合矩阵对 24 类的 logits (用真实矩阵做分类器)
        with torch.no_grad():
            logits_gt = (Pab.detach()[:, None, :, :]
                         * P_true[None, :, :, :]).sum(dim=(-1, -2))
            # (B,24) — 复合矩阵与每个候选矩阵的内积
            pred = logits_gt.argmax(dim=-1)
            self.compose_acc.append((pred == ab_idx).float().mean().item())
        # ③ 置换性正则: 每个矩阵行和/列和接近 1 (soft 置换)
        row_sum = self.perm_mats.sum(dim=-1)     # (24,4)
        col_sum = self.perm_mats.sum(dim=-2)     # (24,4)
        reg = (F.mse_loss(row_sum, torch.ones_like(row_sum))
               + F.mse_loss(col_sum, torch.ones_like(col_sum)))
        return loss + 0.1 * reg

    # ── ② 检索进训练梯度 (V35.1) ─────────────────────────────
    def _define(self, task, detach=True, use_bank=True):
        """define 阶段: train_retrieval=True 时训练期 bank.add 后立即 route
        (带梯度, detach_m=False) → 检索结果参与 compose/head, softmax 检索
        权重梯度回流到 m; 部署期 route 全 detach (β_test 高锁定).
        train_retrieval=False (V35 首轮语义): 训练只 add, 部署只 route."""
        m = super()._define(task, detach=False, use_bank=False)
        sig = task.get('gen_sig') or [0] * N_GEN
        use_bank = self.use_atoms
        hop = isinstance(self.bank, HopfieldBank)  # 检索进训练仅 HopfieldBank
        # V35.5: 潜空间思考 (训练带图, 部署 detach 后纯前向)
        if self._use_think and self.think_net is not None:
            m = self._think(m, task)
            if detach:
                m = m.detach()
        if self._use_compose:
            if use_bank:
                if detach:
                    if hop:
                        m, _ = self.bank.route(m, sig, detach_m=True)
                    else:
                        self.bank.route(m, sig)
                elif self._train_retrieval and hop:
                    self.bank.add(m, sig)
                    m, _ = self.bank.route(m, sig,
                                           beta=self.bank.beta_train,
                                           detach_m=False)
                else:
                    self.bank.add(m, sig)
            comp, sigv = self._compose_inputs(task, m, sig)
            corr = self.compose(m, comp, sigv.to(m.device))
            m = self.m_norm(m + corr)
            if detach:
                m = m.detach()
            return m
        if use_bank:
            if detach:
                if hop:
                    m, _ = self.bank.route(m, sig, detach_m=True)
                else:
                    m, _ = self.bank.route(m, sig)
                m = m.detach()
            elif self._train_retrieval and hop:
                self.bank.add(m, sig)
                m, _ = self.bank.route(m, sig,
                                       beta=self.bank.beta_train,
                                       detach_m=False)
            else:
                self.bank.add(m, sig)
        elif detach:
            m = m.detach()
        return m

    # ── 自适应 K (T10): 边际收益 Δ_k < λ ⟹ 提前终止 ──────────
    def adapt_graph(self, support, K=None, m=None):
        K = K or self.K
        W = self.pln.clone_params()
        prev_loss = None
        stall = 0
        k_used = K
        lim = min(K, len(support))
        for i in range(lim):
            ctx, tgt = support[i]
            hidden = self._rln_fwd(ctx.unsqueeze(0), m)
            h = hidden[:, -1, :]
            hin = self._h_in_batch(h, m)
            x = torch.tanh(hin @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(
                1, self.n_resp, self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            W = self.head.step_graph(W, hin[0], x[0], errs)
            if self._adaptive_k:
                # 边际收益: 当前 W 在下一样本上的 loss 下降量
                if i + 1 < lim:
                    nctx, ntgt = support[i + 1]
                    nh = self._rln_fwd(nctx.unsqueeze(0), m)[:, -1, :]
                    nhin = self._h_in_batch(nh, m)
                    nx = torch.tanh(nhin @ W[0].t() + W[1])
                    nlogits = (nx @ W[2].t() + W[3]).view(
                        1, self.n_resp, self.world_vocab)
                    nloss = F.cross_entropy(
                        nlogits.view(-1, self.world_vocab),
                        ntgt.view(-1).clamp(0, self.V + 1)).item()
                    if prev_loss is not None and (prev_loss - nloss) < self._k_lam:
                        stall += 1
                        if stall >= self._k_patience:
                            k_used = i + 1
                            break
                    else:
                        stall = 0
                    prev_loss = nloss
        self._k_used.append(k_used)
        return W

    # ── meta_step: 主动查询注入 (训练时组合任务加入) ──────────
    def meta_step(self, tasks, train_rules, K=None):
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        # 主动查询: 从 bank 已学原子生成组合任务, 混入训练
        # V35.10: _inject_combo=True 时组合任务带 perm 标签 → probe 回读
        # 监督学会 unseen 组合置换 (probe_loss 在 super.meta_step 内触发,
        # 已与 n_define_refine 解耦)
        if self._active_query and self._aq is not None:
            # 组合课程化 (V35.15): 前期低密度打底 (aq_n=2), 后期
            # 高密度冲刺 (线性涨到配置值) — 原子表示先打稳, 组合再加速
            n_aq = self._aq_n
            if getattr(self, '_aq_curriculum', False):
                p = min(1.0, self._meta_iters / 30.0)
                n_aq = max(2, int(round(2 + (self._aq_n - 2) * p)))
            qs = self._aq.generate(self.bank, n=n_aq)
            for i, q in enumerate(qs):
                t = {'support': [(c, tg[:4]) for (c, tg) in q['support']],
                     'query': [(c, tg[:4]) for (c, tg) in q['query']],
                     'perm': q['perm'], 'gen_sig': q['gen_sig'],
                     'seen': False, 'name': 'AQ%d' % len(self._aq.gen_history),
                     'struct': 'aq'}
                tasks['__aq__%d' % i] = t
                chosen.append('__aq__%d' % i)
        loss = super().meta_step(tasks, chosen, K=K)
        for k in list(tasks):
            if k.startswith('__aq__'):
                del tasks[k]
        return loss

    def long_stream(self, tasks, stream, steps_per_task=60,
                    sleep_every=1, sleep_iters=2, head_state=None,
                    eval_every=20):
        # 部署期 β 高锁定 (HopfieldBank.route 默认 beta_test)
        return super().long_stream(tasks, stream,
                                   steps_per_task=steps_per_task,
                                   sleep_every=sleep_every,
                                   sleep_iters=sleep_iters,
                                   head_state=head_state,
                                   eval_every=eval_every)

    # ── V35.11: 回读驱动预测 ────────────────────────────────
    def _metric(self, W, task, m=None, max_batch=16):
        """super._metric + 回读重映射: probe_head 从 m 回读 perm (4,4),
        部署时用回读 perm 重映射 logits 的寄存器位置 (静态代数:
        组合 perm 已知时, 预测 = 重映射后的常规预测). 只作用于
        部署 (no_grad), 训练仍走连续路径."""
        full, regs = super()._metric(W, task, m=m, max_batch=max_batch)
        if (self._use_remap and m is not None
                and getattr(self, 'probe_head', None) is not None):
            ctx, tgt = batch_pairs(task['query'], max_batch=max_batch)
            with torch.no_grad():
                hidden = self._rln_fwd(ctx, m)
                h = hidden[:, -1, :]
                h_in = self._h_in_batch(h, m)
                pred = head_forward([w.detach() for w in W], h_in,
                                    self.intent_dim, self.world_vocab,
                                    self.n_resp)
                pred = self._route_logits(pred, m)
                pm = self.probe_head(m).view(4, 4).argmax(dim=-1)  # (4,)
                # 重映射: 预测位置 i ← 回读 perm 位置 pm[i]
                # (模型按 identity 学, 组合任务需要按 perm 换位)
                pred_rm = pred.clone()
                pred_rm[:, torch.arange(4)] = pred[:, pm]
                pt = pred_rm.argmax(dim=-1)
                full = (pt == tgt[:, :self.n_resp]).sum().item() / \
                    max(tgt[:, :self.n_resp].numel(), 1)
                regs = (pt[:, :4] == tgt[:, :4]).sum().item() / \
                    max(tgt[:, :4].numel(), 1)
        return full, regs


# ============================================================
# 驱动
# ============================================================
def run_cell(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
             seed=42, steps_per_task=60, verbose=True):
    lm = cfg['lm']
    tasks = make_regs_tasks(tasks)
    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']]
    stream = train_names + unseen_names
    T_STRUCT = build_t_struct(42)
    for n in tasks:
        tasks[n]['name'] = n
        tasks[n]['struct'] = (U_STRUCT.get(n) if not tasks[n]['seen']
                              else T_STRUCT.get(n, '?'))
        sig = generator_signature(tasks[n]['perm'])
        tasks[n]['gen_sig'] = sig[0] if sig else None
        tasks[n]['gen_len'] = sig[1] if sig else 0
    random.seed(seed); torch.manual_seed(seed)
    if cfg.get('fractal4'):
        rln, pln = build_fractal4_resp(200, 4, s1_state,
                                       fractal_pln=cfg.get('fractal_pln',
                                                           False))
    else:
        rln, pln = build_model_resp(200, 4, s1_state)
    use_compose = (lm == 'compose')
    learner = V35_Learner(
        rln, pln, 200, K=K, seed=seed, sleep_iters=2, n_resp=4,
        use_intent=True, use_compose=use_compose,
        use_atoms=(lm != 'base'),
        use_fractal=cfg.get('fractal', False),
        use_hopfield=cfg.get('hopfield', False),
        adaptive_k=cfg.get('adaptive_k', False),
        active_query=cfg.get('active_query', False),
        train_retrieval=cfg.get('train_retrieval', False),
        aq_p3=cfg.get('aq_p3', 0.6),
        aq_eval=cfg.get('aq_eval', False),
        tau_norm=cfg.get('tau_norm', None),
        beta_init=cfg.get('beta_init', None),
        use_think=cfg.get('use_think', False),
        n_think=cfg.get('n_think', 6),
        think_tau=cfg.get('think_tau', 0.01),
        lambda_think=cfg.get('lambda_think', 0.0),
        lambda_combo=cfg.get('lambda_combo', 0.0),
        use_perm=cfg.get('use_perm', False),
        lambda_perm=cfg.get('lambda_perm', 0.5),
        lambda_table=cfg.get('lambda_table', 0.0),
        use_remap=cfg.get('use_remap', False),
        aq_curriculum=cfg.get('aq_curriculum', False),
        lambda_icl=cfg.get('lambda_icl', 0.0),
        alpha=cfg.get('alpha', 1.0),
        beta_train=cfg.get('beta_train', 3.0),
        beta_test=cfg.get('beta_test', 10.0),
        D=cfg.get('D', 1.5),
        aq_n=cfg.get('aq_n', 4),
        aq_max_combo=cfg.get('aq_max_combo', 3),
        k_lam=cfg.get('k_lam', 0.005),
        k_patience=cfg.get('k_patience', 3))
    t0 = time.time()
    n_iters = cfg.get('meta_iters', meta_iters)
    for it in range(n_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_names)
        if verbose and (it + 1) % 20 == 0:
            print("    [%s] it %d/%d loss=%.4f" % (key, it + 1, n_iters, loss), flush=True)
    print("    [%s] 训练 %.0fs" % (key, time.time() - t0), flush=True)
    trained_head = learner.head.state()
    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    print("    [%s] 长流 %.0fs" % (key, time.time() - t0), flush=True)
    by = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen_names:
        s = U_STRUCT[n]
        if s in by:
            by[s].append(st['acc_after'][n])
    struct_mean = {k: (sum(v) / len(v) if v else None) for k, v in by.items()}
    aux = {}
    # ③ AQ 进长流评估: 生成的组合 perm 用训练后 head 独立评估 (acc_after)
    aq_eval_acc = {}
    if cfg.get('aq_eval', False) and learner._aq is not None:
        aq_perms = [h[0] for h in learner._aq.gen_history]
        for i, perm in enumerate(aq_perms):
            sig, _ = generator_signature(perm)
            sup, qry = gen_task_data_split(
                None, 200, n_support=4, n_query=8, seed=1000 + i,
                remap=perm, stop_at_halt=True)
            aqt = {'support': [(c, tg[:4]) for (c, tg) in sup],
                   'query': [(c, tg[:4]) for (c, tg) in qry],
                   'perm': perm, 'gen_sig': sig, 'seen': False,
                   'name': 'AQE%d' % i, 'struct': 'aq'}
            with torch.no_grad():
                m = learner._define(aqt, detach=True)
                acc = learner._metric([w.detach() for w in
                                       learner.pln.clone_params()],
                                      aqt, m=m)[1]
            aq_eval_acc['aq%d_%s' % (i, perm)] = round(acc, 4)
        aux['aq_eval_acc'] = aq_eval_acc
    if getattr(learner, 'bank', None) is not None:
        aux['atom_bank'] = learner.bank.stats()
    if getattr(learner, 'compose', None) is not None:
        cp = learner.compose
        aux['compose_params'] = sum(p.numel() for p in cp.parameters())
    if learner._use_fractal:
        aux['fractal_gate_params'] = sum(
            p.numel() for p in learner._fractal_gate_params)
    if learner._adaptive_k:
        kus = learner._k_used
        aux['k_used'] = {'mean': round(sum(kus) / len(kus), 2),
                         'min': min(kus), 'max': max(kus)}
    if learner._aq is not None:
        aux['aq_gen'] = len(learner._aq.gen_history)
        aux['aq_combo_lens'] = [h[2] for h in learner._aq.gen_history[-10:]]
    if getattr(learner, 'think_net', None) is not None:
        aux['think_params'] = sum(p.numel()
                                  for p in learner.think_net.parameters())
        tr = learner.think_rounds
        aux['think_rounds'] = {'mean': round(sum(tr) / len(tr), 2),
                               'min': min(tr), 'max': max(tr)} if tr else None
        # 收敛验证 (T8): 最后若干次思考的检验误差轨迹是否下降
        traces = [t for t in learner.think_trace if len(t) >= 2]
        if traces:
            drops = [t[0] - t[-1] for t in traces]
            aux['think_err_drop'] = {
                'mean': round(sum(drops) / len(drops), 4),
                'frac_positive': round(
                    sum(1 for d in drops if d > 0) / len(drops), 3)}
        else:
            aux['think_err_drop'] = None
    if getattr(learner, 'perm_head', None) is not None:
        pa = learner.perm_acc
        aux['perm_acc'] = {
            'seen': (round(sum(pa['seen']) / len(pa['seen']), 3)
                     if pa['seen'] else None),
            'unseen': (round(sum(pa['unseen']) / len(pa['unseen']), 3)
                       if pa['unseen'] else None),
            'n_seen': len(pa['seen']), 'n_unseen': len(pa['unseen'])}
        ca = learner.compose_acc
        aux['compose_acc'] = (round(sum(ca) / len(ca), 3) if ca else None)
    # 学习效率参数观测 (SwiftTDHead β/θ/h): 缺失就补 (V32_1/V33 有, V35 曾丢)
    try:
        hs = learner.head.step_size_stats()
        aux['head_beta'] = {k: (round(v, 6) if isinstance(v, float)
                                else v) for k, v in hs.items()}
    except Exception:
        pass
    n_params = sum(p.numel() for p in rln.parameters()) + sum(p.numel() for p in pln.parameters())
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'world': cfg.get('world', 'new'),
           'factors': {'LM': lm, 'alpha': cfg.get('alpha', 1.0),
                       'use_compose': use_compose,
                       'use_fractal': cfg.get('fractal', False),
                       'use_hopfield': cfg.get('hopfield', False),
                       'adaptive_k': cfg.get('adaptive_k', False),
                       'active_query': cfg.get('active_query', False)},
           'time_s': round(time.time() - t0, 1),
           'n_params': n_params,
           'params_kb': round(n_params * 4 / 1024, 1),
           'seen_acc': {n: st['acc_after'][n] for n in train_names},
           'unseen_acc': {n: st['acc_after'][n] for n in unseen_names},
           'unseen_by_struct': struct_mean,
           'aux': aux}
    u = rec['unseen_acc']; ua = sum(u.values()) / len(u)
    print((u"  [%s] unseen=%.4f struct=%s n_params=%d aux=%s" %
           (key, ua, struct_mean, n_params, aux)), flush=True)
    return rec


CELLS = {
    'B':   dict(world='new', lm='base', label='B-base'),
    'C':   dict(world='new', lm='atom', alpha=1.0, label='C-atom-pure'),
    'M':   dict(world='new', lm='compose', alpha=1.0, label='M-compose-learn'),
    'F':   dict(world='new', lm='compose', alpha=1.0, fractal=True,
                label='F-fractal'),
    'FH':  dict(world='new', lm='compose', alpha=1.0, fractal=True,
                hopfield=True, label='FH-fractal-hopfield'),
    'FHA': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                hopfield=True, adaptive_k=True, label='FHA-fractal-hopfield-adapK'),
    'FHAQ': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                 hopfield=True, adaptive_k=True, active_query=True,
                 label='FHAQ-full'),
    # ── V35.1 (②③) ──
    'FHR': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                hopfield=True, adaptive_k=True, train_retrieval=True,
                label='FHR-train-retrieval'),
    'FHAQ2': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                  hopfield=True, adaptive_k=True, active_query=True,
                  train_retrieval=True, aq_p3=0.7, aq_eval=True,
                  label='FHAQ2-full-v2'),
    # ── V35.2 学习效率诊断 (方案 A: τ 维度归一) ──
    'FN': dict(world='new', lm='compose', alpha=1.0, fractal=True,
               hopfield=True, adaptive_k=True, tau_norm='mean',
               label='FN-tau-norm-mean'),
    'FN2': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                hopfield=True, adaptive_k=True, tau_norm='mean',
                train_retrieval=True, label='FN2-tau-norm-train-retr'),
    # V35.3: β_init 提高恢复 IDBD 分化 (V31.1 配置: 0.05)
    'FB': dict(world='new', lm='compose', alpha=1.0, fractal=True,
               hopfield=True, adaptive_k=True, tau_norm='mean',
               beta_init=0.05, label='FB-tau-norm-beta05'),
    'FBT': dict(world='new', lm='compose', alpha=1.0, fractal=True,
                hopfield=True, adaptive_k=True, tau_norm='mean',
                beta_init=0.05, train_retrieval=True,
                label='FBT-tau-norm-beta05-train-retr'),
    # ── V35.4: 4 层分形 RLN (FractalRLN4, 等比宽度 128/102/81/64) ──
    'F4': dict(world='new', lm='compose', alpha=1.0, fractal4=True,
               label='F4-fractal4-rln'),
    'F4P': dict(world='new', lm='compose', alpha=1.0, fractal4=True,
                fractal_pln=True, label='F4P-fractal4-rln+pln'),
    # ── V35.5: 潜空间思考循环 (E-V 闭环内部化) ──
    'TH': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               label='TH-think-loop'),
    'THF': dict(world='new', lm='compose', alpha=1.0,
                hopfield=True, adaptive_k=True, use_think=True, n_think=6,
                train_retrieval=True, active_query=True, aq_p3=0.6,
                label='THF-think-full'),
    # ── V35.6: 思考监督直接化 (学修正假设, 非讨好训练) ──
    # lambda_icl 复用为思考监督权重 (V33.meta_step 用它加权 aux loss)
    'TH2': dict(world='new', lm='compose', alpha=1.0,
                hopfield=True, adaptive_k=True, use_think=True, n_think=6,
                lambda_icl=0.5, label='TH2-think-direct'),
    'TH2F': dict(world='new', lm='compose', alpha=1.0,
                 hopfield=True, adaptive_k=True, use_think=True, n_think=6,
                 lambda_icl=0.5, train_retrieval=True, active_query=True,
                 aq_p3=0.6, label='TH2F-think-direct-full'),
    # ── V35.7: 思考 + unseen 组合检验监督 (最后一环) ──
    'TH3': dict(world='new', lm='compose', alpha=1.0,
                hopfield=True, adaptive_k=True, use_think=True, n_think=6,
                lambda_icl=0.5, lambda_combo=0.5, active_query=True,
                aq_p3=0.6, label='TH3-think-combo'),
    'TH3F': dict(world='new', lm='compose', alpha=1.0,
                 hopfield=True, adaptive_k=True, use_think=True, n_think=6,
                 lambda_icl=0.5, lambda_combo=0.5, train_retrieval=True,
                 active_query=True, aq_p3=0.6,
                 label='TH3F-think-combo-full'),
    # ── V35.8 (④): S4 群代数 — perm 回读 + 复合表 ──
    'G0': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, use_perm=True,
               lambda_perm=0.5, lambda_table=0.0,
               label='G0-perm-readback'),
    'G1': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, use_perm=True,
               lambda_perm=0.5, lambda_table=0.5,
               label='G1-perm+table'),
    'G2': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, use_perm=True,
               lambda_perm=1.0, lambda_table=0.5, train_retrieval=True,
               active_query=True, aq_p3=0.6,
               label='G2-perm-full'),
    # ── V35.10: 回读闭环 (probe 解耦 + 组合任务注入) ──
    'R0': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5,
               label='R0-probe-fixed'),
    'R1': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=2, aq_p3=0.7,
               label='R1-probe+combo-inject'),
    'R2': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=2, aq_p3=0.7, train_retrieval=True,
               label='R2-probe+inject+retrieval'),
    # ── V35.11: 回读驱动预测 (部署时 probe perm 重映射 logits) ──
    'R3': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=2, aq_p3=0.7, use_remap=True,
               label='R3-remap-readback'),
    'R4': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=2, aq_p3=0.7, train_retrieval=True, use_remap=True,
               label='R4-remap+retrieval'),
    # ── V35.12: 组合样本密度 + probe 强化 ──
    'S0': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0,
               label='S0-dense-combo+probe'),
    'S1': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0, use_remap=True,
               label='S1-dense+remap'),
    # ── V35.13: 密度再放大 + probe 加倍 ──
    'S2': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=10, aq_p3=0.7, lambda_probe=2.0,
               label='S2-dense10-probe2'),
    'S3': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=14, aq_p3=0.7, lambda_probe=2.0, train_retrieval=True,
               label='S3-dense14-probe2-retr'),
    # ── V35.14: S0 微调 (组合权重/思考轮数) + 多 seed ──
    'T0': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.3, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0,
               label='T0-combo-soft'),
    'T1': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=8,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0,
               label='T1-think8'),
    'T2': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True,
               lambda_icl=0.5, lambda_combo=0.3, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0, n_think=8,
               label='T2-combo-soft-think8'),
    # ── V35.15: 组合课程化 (前期低密度打底, 后期高密度冲刺) ──
    'U0': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0, aq_curriculum=True,
               label='U0-curriculum'),
    'U1': dict(world='new', lm='compose', alpha=1.0,
               hopfield=True, adaptive_k=True, use_think=True, n_think=6,
               lambda_icl=0.5, lambda_combo=0.5, active_query=True,
               aq_n=6, aq_p3=0.7, lambda_probe=1.0, aq_curriculum=True,
               train_retrieval=True,
               label='U1-curriculum+retr'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='B,F,FH,FHA,FHAQ')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35 分形持续学习引擎 (lm-principle 83 定理重设计)")
    print("=" * 70)
    tasks = build_v32_tasks(200, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
    s1 = None
    if not args.no_stage1:
        need_f4 = any(CELLS[c.strip()].get('fractal4') for c in
                      args.cells.split(','))
        cands = (['/tmp/v32_s1_f4_state.pt'] if need_f4
                 else ['/tmp/v32_s1_state.pt', '/tmp/v31_2_s1_state.pt'])
        for cand in cands:
            if os.path.exists(cand):
                s1 = torch.load(cand)
                s1 = {k: v.cpu().clone() for k, v in s1.items()}
                print("Stage1: %s" % cand); break
        else:
            print(">>> Stage1 缺失, 请先跑 run_v32_stage1.py"
                  " 或 run_v35_4_fractal4.py"); return
    res = []
    for key in [c.strip() for c in args.cells.split(',')]:
        cfg = CELLS[key]
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell(key, cfg, s1, tasks, U_STRUCT,
                         meta_iters=args.meta_iters, seed=args.seed)
        except Exception:
            import traceback; traceback.print_exc()
            print("!!! %s 失败" % key); continue
        res.append(r)
    out = ROOT / 'results' / 'v35_fractal_report.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s params=%d %s" %
              (r['cell'], ua, r['unseen_by_struct'], r['n_params'], r['aux']))


if __name__ == '__main__':
    main()
