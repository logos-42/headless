"""
V32.1-32.9: 表示瓶颈攻坚 — 矩阵五方向消融 (regs-only / 规则身份 / 对比 / EHS / 换世界)
========================================================================================
计划: docs/wiki/v32-1-matrix-plan.md (预注册: 因子矩阵 / 假设 H1-H6 / 阈值)

立论 (V31.0 → V32 四层一致):
  head 侧一切干预 (标量/BPTT/SwiftTD/意图注入/EMA 潜目标) 都撞 ~0.39 未见天花板;
  V32 证明零步 goal readout ≥ 60 步适应 → 瓶颈在 RLN 表示本身, 不在 head 接口。
  → 本矩阵动**表示的学习目标与架构**, 五方向:
    R regs-only   : head n_resp=4 (只预测寄存器), m = Δregs 嵌入 (去指令 token 稀释)
    P 规则身份    : aux_head 组成性预测 4×4 寄存器重映射 (perm[i] 而非 24 类 ID)
    C 对比        : prototypical 对比 loss (同规则 query 近原型, 异规则远), λ=0.3
    E EHS         : 独立 rule_encoder (Linear→Tanh→Linear), 完全替换 intent_net
                    (论文主线: 显式假设分离), head 输入 [h; h_rule; h⊙h_rule]
    W 换世界      : window=4 (ctx=32, 观察历史更长 → 规则可辨识性更强)

细胞 (10):
  C0   8-token 基线 (≡ V32 C1, 验证实现等价) + regs_acc 统一 4-token 报告
  32.1 R / 32.2 R+P / 32.3 R+C / 32.4 R+E / 32.5 R+W
  32.6 R+P+C / 32.7 R+P+E / 32.8 R+C+E / 32.9 R+P+C+E

控制变量 (与 V32 完全一致): 世界 S4 24 排列 · Stage1 同缓存 · meta 80 iters
  K=20 · 2 tasks/step · Adam 1e-3 · SwiftTD β_init=1e-3 κ=0.1 η=0.1 ϵ=0.99
  长流 16×60 步 · sleep_iters=2 · seed=42 · n_support=40 n_query=20

主指标: 未见规则 mean regs_acc (60 步段后) + 零步 readout regs_acc (rd0)
  结构分组: unseen c3 vs c4 vs dbl 分别报告 (V32 §4.3 教训)
  aux: remap 预测 acc (P cells), 对比 loss 收敛 (C cells)

实现要点:
  - V32_1_Learner(V32_Learner): n_resp 参数化贯穿 (head/PLN/评估), Δregs intent
    (goal detached + local attached 双条件), rule_encoder / aux_head / 对比原型
  - C0 = n_resp=8 + use_intent=False → 与 V32 C1 同构 (活跃基线验证)
  - regs 任务预处理: tgt 切片 [:4] (make_regs_tasks), 世界与 ctx 不变
  - 32.5 用 build_v32_tasks(window=4), 其余 window=3 (单一变量)
"""

import os, sys, time, json, math, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hibs_lnn.swifttd_head import SwiftTDHead
from hibs_lnn.meta_rule_world import build_rule_splits, s4_groups
from run_v31_meta_learning import (
    batch_pairs, head_forward, stage1_pretrain_world,
    V31_RLN, V31_PLN,
)
from run_v31_2_long_stream import build_v32_tasks
from run_v32_intent_head import V32_Learner

PAD_ID = 0

# ============================================================
# 细胞定义 (9 cell × 5 因子; W 只进 32.5, 32.9 与 32.1 同世界)
#   C0: V32 C1 原样 (n_resp=8, 无 m) + regs_acc 报告
# ============================================================
CELLS = {
    'C0':   dict(n_resp=8,  use_intent=False, use_rule_enc=False,
                 use_perm_aux=False, use_contrast=False, window=3,
                 label='C0-8token-baseline'),
    '32.1': dict(n_resp=4,  use_intent=True,  use_rule_enc=False,
                 use_perm_aux=False, use_contrast=False, window=3,
                 label='32.1-regs-only'),
    '32.2': dict(n_resp=4,  use_intent=True,  use_rule_enc=False,
                 use_perm_aux=True,  use_contrast=False, window=3,
                 label='32.2-rule-id'),
    '32.3': dict(n_resp=4,  use_intent=True,  use_rule_enc=False,
                 use_perm_aux=False, use_contrast=True,  window=3,
                 label='32.3-contrast'),
    '32.4': dict(n_resp=4,  use_intent=True,  use_rule_enc=True,
                 use_perm_aux=False, use_contrast=False, window=3,
                 label='32.4-EHS'),
    '32.5': dict(n_resp=4,  use_intent=True,  use_rule_enc=False,
                 use_perm_aux=False, use_contrast=False, window=4,
                 label='32.5-window4'),
    '32.6': dict(n_resp=4,  use_intent=True,  use_rule_enc=False,
                 use_perm_aux=True,  use_contrast=True,  window=3,
                 label='32.6-ruleid+contrast'),
    '32.7': dict(n_resp=4,  use_intent=True,  use_rule_enc=True,
                 use_perm_aux=True,  use_contrast=False, window=3,
                 label='32.7-ruleid+EHS'),
    '32.8': dict(n_resp=4,  use_intent=True,  use_rule_enc=True,
                 use_perm_aux=False, use_contrast=True,  window=3,
                 label='32.8-contrast+EHS'),
    '32.9': dict(n_resp=4,  use_intent=True,  use_rule_enc=True,
                 use_perm_aux=True,  use_contrast=True,  window=3,
                 label='32.9-all'),
}

FACTOR_NAMES = ['R', 'P', 'C', 'E', 'W']


# ============================================================
# 模型构建 (build_model 的 n_response 参数化版本, 不碰共享文件)
# ============================================================
def build_model_resp(V, n_response, s1_state=None, d_model=128,
                    d_state=8, n_layers=2):
    """与 run_v31_meta_learning.build_model 同构, 但 n_response 参数化。
    d_model/d_state/n_layers 可配 (LM1 生产级规模扩展, 默认保持 V32 兼容)."""
    rln = V31_RLN(V + 2, d_model=d_model, d_state=d_state, n_layers=n_layers)
    if s1_state is not None:
        own = rln.state_dict()
        stripped = {}
        for k, v in s1_state.items():
            kk = k[4:] if k.startswith('rln.') else k
            if kk in own and own[kk].shape == v.shape:
                stripped[kk] = v
        own.update(stripped)
        rln.load_state_dict(own)
    pln = V31_PLN(d_model, V + 2, n_response=n_response)
    return rln, pln


def make_regs_tasks(tasks):
    """regs-only 预处理: target 只留前 4 个寄存器 token (规则相关);
    世界/ctx 不变。C0 用原 8-token 任务, 32.x 用切片后的。"""
    out = {}
    for n, t in tasks.items():
        out[n] = {
            'support': [(c, tg[:4]) for (c, tg) in t['support']],
            'query': [(c, tg[:4]) for (c, tg) in t['query']],
            'perm': t['perm'], 'seen': t['seen'],
        }
    return out


# ============================================================
# V32_1_Learner
# ============================================================
class V32_1_Learner(V32_Learner):
    """V32 + 五方向机制:
    R: n_resp=4, m = Δregs 嵌入 (intent_net)
    P: aux_head 组成性 4×4 remap 预测
    C: prototypical 对比 loss (12 原型, no_grad)
    E: 独立 rule_encoder 完全替换 intent_net (head 输入 [h;h_rule;h⊙h_rule])
    W: 由外部任务数据 (window=4) 控制, 学习器无感知
    """

    def __init__(self, rln, pln, V, K=20, outer_lr=1e-3,
                 n_tasks_per_step=2, seed=42,
                 beta_init=1e-3, kappa=0.1, eta=0.1, eps=0.99,
                 init_lr=0.0, sleep_iters=2,
                 n_resp=4, use_intent=None, use_rule_enc=False,
                 use_perm_aux=False, use_contrast=False,
                 lambda_perm=0.1, lambda_contra=0.3, n_local_sup=8):
        if use_intent is None:
            use_intent = (n_resp < 8)          # C0 (n_resp=8) 无 m 注入
        d = rln.d_model
        self.lambda_perm = lambda_perm
        self.lambda_contra = lambda_contra

        super().__init__(rln, pln, V, K=K, outer_lr=outer_lr,
                         n_tasks_per_step=n_tasks_per_step, seed=seed,
                         beta_init=beta_init, kappa=kappa, eta=eta, eps=eps,
                         init_lr=init_lr, sleep_iters=sleep_iters,
                         use_intent=use_intent, use_ema=False,
                         n_local_sup=n_local_sup)

        # R: n_resp 贯穿 head (V32 基类在 super 里建了 n_resp=8 的 head)
        self.n_resp = n_resp
        self.use_rule_enc = use_rule_enc
        self.use_perm_aux = use_perm_aux
        self.use_contrast = use_contrast
        self.head = SwiftTDHead(self.d_model, self.world_vocab, self.n_resp,
                                beta_init=beta_init, kappa=kappa, eta=eta,
                                eps=eps, input_dim=self.intent_dim)
        self.head.reset_state()

        # P/E 附加网络; E 完全替换 intent_net (冻结, 不进 optimizer)
        extra = list(self.intent_net.parameters())
        if use_rule_enc:
            self.rule_encoder = nn.Sequential(
                nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))
            extra = list(self.rule_encoder.parameters())
        if use_perm_aux:
            self.aux_head = nn.Linear(d, 16)
            extra = extra + list(self.aux_head.parameters())
        self.outer_opt = torch.optim.Adam(
            list(rln.parameters()) + list(pln.parameters()) + extra,
            lr=outer_lr)
        self.loss_trace = {'perm': [], 'contra': []}

    # ── 规则向量 (E 或 intent_net) ──────────────────────────
    def _rule_vec(self, agg):
        if self.use_rule_enc:
            return self.rule_encoder(agg)
        return self.intent_net(agg)

    def _dregs(self, cs, ts):
        """Δregs 聚合嵌入: mean(embed(tgt[:4]) − embed(ctx[-8:-4]))。
        ctx_regs = ctx 最后 8 个 token 的前 4 个 (最近观察的寄存器部分)。"""
        e_t = self.rln.embed(ts[:, :4])        # (B,4,d)
        e_c = self.rln.embed(cs[:, -8:-4])     # (B,4,d)
        return (e_t - e_c).mean(dim=1).mean(dim=0)   # (d,)

    def compute_goal_intent(self, task):
        """goal call (部署): Δregs 聚合, stop-grad anchor。"""
        sup = task['support']
        B = min(len(sup), 16)
        with torch.no_grad():
            cs = torch.stack([s[0] for s in sup[:B]])
            ts = torch.stack([s[1] for s in sup[:B]])
            delta = self._dregs(cs, ts)
            m = self.m_norm(self._rule_vec(delta))
        return m.detach()

    def _local_rule_intent(self, c, tg):
        """local call (训练): per-sample Δregs, attached (梯度到 RLN+编码器)。"""
        e_t = self.rln.embed(tg[:4].unsqueeze(0))
        e_c = self.rln.embed(c[-8:-4].unsqueeze(0))
        delta = (e_t - e_c).mean(dim=1).mean(dim=0)
        return self.m_norm(self._rule_vec(delta))

    # ── 对比原型 (12 条训练规则, no_grad; 只 query h 反传) ──
    def _compute_prototypes(self, tasks, train_rules):
        with torch.no_grad():
            protos = []
            for rn in train_rules:
                hs = [self.rln(c.unsqueeze(0))[:, -1, :]
                      for (c, _) in tasks[rn]['support'][:8]]
                protos.append(torch.cat(hs, 0).mean(0))
            P = torch.stack(protos)                    # (12,d)
            return P / (P.norm(dim=1, keepdim=True) + 1e-8)

    # ── 外循环 (BPTT): 主 loss + local call + P + C ────────
    def meta_step(self, tasks, train_rules, K=None):
        self.outer_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        Ws = []
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        protos = (self._compute_prototypes(tasks, train_rules)
                  if self.use_contrast else None)
        for rname in chosen:
            t = tasks[rname]
            m = self.compute_goal_intent(t) if self.use_intent else None
            W = self.adapt_graph(t['support'], m=m, K=K)
            Ws.append([w.detach() for w in W])
            ctx, tgt = batch_pairs(t['query'], max_batch=16)
            hidden = self.rln(ctx)
            h = hidden[:, -1, :]
            h_in = self._h_in_batch(h, m)
            pred = head_forward(W, h_in, self.intent_dim, self.world_vocab,
                                self.n_resp)
            loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                   tgt.view(-1).clamp(0, self.V + 1))
            total_loss = total_loss + loss / len(chosen)

            # 双条件共享算子: local call (attached), per-sample Δregs
            if self.use_intent and self.lambda_local > 0:
                for (c, tg) in t['support'][:self.n_local_sup]:
                    hsv = self.rln(c.unsqueeze(0))[:, -1, :]
                    m_loc = self._local_rule_intent(c, tg)
                    hin = self._h_in_batch(hsv, m_loc)
                    pl = head_forward(W, hin, self.intent_dim,
                                      self.world_vocab, self.n_resp)
                    total_loss = total_loss + self.lambda_local * \
                        F.cross_entropy(pl.view(-1, self.world_vocab),
                                        tg.view(-1).clamp(0, self.V + 1)) \
                        / (len(chosen) * self.n_local_sup)

            # P: 规则身份 (perm) 组成性 aux — 预测 perm[i] 而非 24 类 ID
            if self.use_perm_aux:
                aux = self.aux_head(h)                  # (B,16)
                perm = torch.tensor(t['perm'], dtype=torch.long)  # (4,)
                tl = perm.unsqueeze(0).expand(h.shape[0], -1).reshape(-1)
                l_perm = F.cross_entropy(aux.view(-1, 4), tl)
                total_loss = total_loss + self.lambda_perm * l_perm \
                    / len(chosen)
                self.loss_trace['perm'].append(l_perm.item())

            # C: prototypical 对比 — query h 近同规则原型, 异规则远
            if protos is not None:
                qn = h / (h.norm(dim=1, keepdim=True) + 1e-8)
                sim = qn @ protos.t()                   # (B,12)
                tid = train_rules.index(rname)
                tl = torch.full((h.shape[0],), tid, dtype=torch.long)
                l_con = F.cross_entropy(sim, tl)
                total_loss = total_loss + self.lambda_contra * l_con \
                    / len(chosen)
                self.loss_trace['contra'].append(l_con.item())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.rln.parameters() if p.requires_grad] +
            [p for p in self.pln.parameters() if p.requires_grad], 1.0)
        self.outer_opt.step()
        if self.init_lr > 0 and Ws:
            with torch.no_grad():
                scale = self.init_lr / len(Ws)
                for wa in Ws:
                    for p, w in zip(self.init_params, wa):
                        p.add_(scale * (w - p.detach()))
        self._meta_iters += 1
        return total_loss.item()

    # ── 评估 (统一 regs_acc 主指标; C0 兼报 full acc) ──────
    def _metric(self, W, task, m=None, max_batch=16):
        """返回 (full_acc, regs_acc)。regs_acc = 前 4 个响应 token 精确匹配。
        32.x (n_resp=4, tgt 4-token) 两者相等; C0 (n_resp=8) 分离报告。"""
        ctx, tgt = batch_pairs(task['query'], max_batch=max_batch)
        with torch.no_grad():
            hidden = self.rln(ctx)
        h = hidden[:, -1, :]
        h_in = self._h_in_batch(h, m)
        pred = head_forward([w.detach() for w in W], h_in,
                            self.intent_dim, self.world_vocab, self.n_resp)
        pt = pred.argmax(dim=-1)                       # (B, n_resp)
        full = (pt == tgt[:, :self.n_resp]).sum().item() / \
            max(tgt[:, :self.n_resp].numel(), 1)
        regs = (pt[:, :4] == tgt[:, :4]).sum().item() / \
            max(tgt[:, :4].numel(), 1)
        return full, regs

    def eval_goal_readout(self, task, head_state=None):
        """INTACT 零步 readout: goal call, 不做任何 head 适应。"""
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self.compute_goal_intent(task) if self.use_intent else None
        return self._metric(W, task, m=m)

    def eval_adaptation_curve(self, task, max_steps=40, eval_every=10,
                              head_state=None):
        """off-stream 适应曲线 (0/10/20/30/40), regs_acc。"""
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self.compute_goal_intent(task) if self.use_intent else None
        curve = [self._metric(W, task, m=m)[1]]
        sup = task['support']
        for j in range(max_steps):
            if j < len(sup):
                with torch.no_grad():
                    hidden = self.rln(sup[j][0].unsqueeze(0))
                h = hidden[:, -1, :]
                hin = self._h_in_batch(h, m)
                x = torch.tanh(hin @ W[0].t() + W[1])
                logits = (x @ W[2].t() + W[3]).view(
                    1, self.n_resp, self.world_vocab)
                p = F.softmax(logits, dim=-1)
                onehot = F.one_hot(sup[j][1], self.world_vocab).float()
                errs = (p - onehot).reshape(-1)
                self.head.step(W, hin[0], x[0], errs)
            if (j + 1) % eval_every == 0:
                curve.append(self._metric(W, task, m=m)[1])
        return curve

    def long_stream(self, tasks, stream, steps_per_task=60,
                    sleep_every=1, sleep_iters=2, head_state=None,
                    eval_every=20):
        """长流: W/θ/h 跨任务持续; sleep = BPTT meta_step 于已见任务。
        修正 V32 细节: 遗忘行按任务各自 goal intent 评估 (V32 误用当前任务 m)。"""
        self._load_head(head_state)
        W = [w.detach() for w in self.pln.clone_params()]
        seen = []
        forget_rows = []
        acc_after = {}
        acc_after_full = {}
        curves = {}
        m_cache = {}
        for tname in stream:
            t = tasks[tname]
            if tname not in m_cache:
                m_cache[tname] = (self.compute_goal_intent(t)
                                  if self.use_intent else None)
            m = m_cache[tname]
            seg = [self._metric(W, t, m=m)[1]]
            for j in range(steps_per_task):
                if j < len(t['support']):
                    with torch.no_grad():
                        hidden = self.rln(t['support'][j][0].unsqueeze(0))
                    h = hidden[:, -1, :]
                    hin = self._h_in_batch(h, m)
                    x = torch.tanh(hin @ W[0].t() + W[1])
                    logits = (x @ W[2].t() + W[3]).view(
                        1, self.n_resp, self.world_vocab)
                    p = F.softmax(logits, dim=-1)
                    onehot = F.one_hot(t['support'][j][1],
                                       self.world_vocab).float()
                    errs = (p - onehot).reshape(-1)
                    self.head.step(W, hin[0], x[0], errs)
                if (j + 1) % eval_every == 0:
                    seg.append(self._metric(W, t, m=m)[1])
            seen.append(tname)
            curves[tname] = seg
            acc_after[tname] = seg[-1]
            acc_after_full[tname] = self._metric(W, t, m=m)[0]
            row = {}
            for s in seen:
                if s not in m_cache:
                    m_cache[s] = (self.compute_goal_intent(tasks[s])
                                  if self.use_intent else None)
                row[s] = round(self._metric(W, tasks[s], m=m_cache[s])[1], 4)
            forget_rows.append(row)
            if sleep_iters > 0 and len(seen) >= sleep_every:
                saved = self.head.state()
                for _ in range(sleep_iters):
                    self.meta_step(tasks, seen)
                self.head.load_state(saved)
        return {'acc_after': acc_after, 'acc_after_full': acc_after_full,
                'curves': curves, 'forget': forget_rows}


# ============================================================
# 运行一个 cell
# ============================================================
def eval_remap_acc(learner, tasks, train_names, unseen_names):
    """P 因子 aux 指标: 组成性 remap 预测 acc (4 位置逐位正确)。"""
    def _acc(names):
        ok = tot = 0
        for n in names:
            t = tasks[n]
            ctx, _ = batch_pairs(t['query'], max_batch=16)
            with torch.no_grad():
                h = learner.rln(ctx)[:, -1, :]
            aux = learner.aux_head(h)                       # (B,16)
            pred = aux.view(h.shape[0], 4, 4).argmax(dim=-1)  # (B,4)
            perm = torch.tensor(t['perm'], dtype=torch.long).unsqueeze(0)
            ok += (pred == perm).sum().item()
            tot += pred.numel()
        return round(ok / max(tot, 1), 4)
    return _acc(train_names), _acc(unseen_names)


def run_cell(cell, cfg, V, s1_state, tasks_w3, tasks_w4, U_STRUCT,
             n_meta_iters=80, K=20, seed=42, steps_per_task=60,
             verbose=True):
    label = cfg['label']
    window = cfg['window']
    base = tasks_w4 if window == 4 else tasks_w3
    n_resp = cfg['n_resp']
    tasks = base if n_resp == 8 else make_regs_tasks(base)
    train_rules = [n for n in tasks if tasks[n]['seen']]
    train_names = train_rules[:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']][:8]
    stream = train_names + unseen_names

    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(V, n_resp, s1_state)
    learner = V32_1_Learner(rln, pln, V, K=K, seed=seed, sleep_iters=2,
                            n_resp=n_resp, use_intent=cfg['use_intent'],
                            use_rule_enc=cfg['use_rule_enc'],
                            use_perm_aux=cfg['use_perm_aux'],
                            use_contrast=cfg['use_contrast'])
    t0 = time.time()
    for it in range(n_meta_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_rules)
        if verbose and (it + 1) % 20 == 0:
            print(f"    [{label}] meta_iter {it+1}/{n_meta_iters} "
                  f"loss={loss:.4f}", flush=True)
    trained_head = learner.head.state()
    dt = time.time() - t0
    print(f"    [{label}] meta 训练 {n_meta_iters} iters 完成 ({dt:.0f}s)",
          flush=True)

    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    t_ls = time.time() - t0
    print(f"    [{label}] 长流完成 ({t_ls:.0f}s)", flush=True)

    curves = {}
    rd0 = {}
    rd0_full = {}
    for n in unseen_names:
        curves[n] = learner.eval_adaptation_curve(
            tasks[n], max_steps=40, eval_every=10, head_state=trained_head)
        f, r = learner.eval_goal_readout(tasks[n], head_state=trained_head)
        rd0[n] = r
        rd0_full[n] = f
    rd0_train = {}
    for n in train_names:
        f, r = learner.eval_goal_readout(tasks[n], head_state=trained_head)
        rd0_train[n] = r

    # 结构分组 (未见: c3 三轮换 / c4 四轮换 / dbl 双对换)
    by_struct = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen_names:
        s = U_STRUCT[n]
        if s in by_struct:
            by_struct[s].append(st['acc_after'][n])
    struct_mean = {k: (sum(v) / len(v) if v else None)
                   for k, v in by_struct.items()}

    aux = {}
    if cfg['use_perm_aux']:
        a_tr, a_un = eval_remap_acc(learner, tasks, train_names, unseen_names)
        aux['remap_acc_train'] = a_tr
        aux['remap_acc_unseen'] = a_un
    if cfg['use_contrast']:
        tr = learner.loss_trace['contra']
        last10 = tr[-10:] if len(tr) >= 10 else tr
        aux['contra_loss_mean_last10'] = \
            round(sum(last10) / len(last10), 4) if last10 else None
        aux['contra_loss_final'] = round(tr[-1], 4) if tr else None

    head_stats = learner.head.step_size_stats()
    result = {
        'cell': cell, 'label': label, 'seed': seed, 'time_s': round(dt, 1),
        'meta_iters': n_meta_iters, 'K': K,
        'window': window, 'n_resp': n_resp,
        'factors': {
            'R': n_resp == 4,
            'P': cfg['use_perm_aux'],
            'C': cfg['use_contrast'],
            'E': cfg['use_rule_enc'],
            'W': window == 4,
        },
        'stream': stream,
        'train_acc_after': {n: st['acc_after'][n] for n in train_names},
        'unseen_acc_after': {n: st['acc_after'][n] for n in unseen_names},
        'unseen_acc_after_full': {n: st['acc_after_full'][n]
                                  for n in unseen_names},
        'readout0_unseen': rd0,
        'readout0_unseen_full': rd0_full,
        'readout0_train': rd0_train,
        'unseen_by_struct': struct_mean,
        'stream_curves': st['curves'],
        'unseen_curve_offstream': curves,
        'forget': st['forget'],
        'head_stats': head_stats,
        'aux': aux,
    }
    ua = result['unseen_acc_after']
    u_mean = sum(ua.values()) / len(ua)
    r_mean = sum(rd0.values()) / len(rd0)
    dt = time.time() - t0
    result['time_s'] = round(dt, 1)
    print(f"  [{label}] 全流程完成 ({dt:.0f}s)", flush=True)
    print(f"     训练 regs_acc: {result['train_acc_after']}", flush=True)
    print(f"     未见 regs_acc: {ua}  mean={u_mean:.4f}", flush=True)
    print(f"     未见 rd0: {rd0}  mean={r_mean:.4f}", flush=True)
    print(f"     结构分组: {struct_mean}", flush=True)
    if aux:
        print(f"     aux: {aux}", flush=True)
    return result


# ============================================================
# 报告
# ============================================================
def _mean(d):
    return sum(d.values()) / max(len(d), 1)


def build_t_struct(seed=42):
    """训练任务结构分组: T0-T11 ↔ build_rule_splits(seed) 的 train_perms 顺序。
    结构 = ident / swap / c3 (V33.1 模式库分组检索用)。"""
    train_perms, test_perms = build_rule_splits(seed)
    identity, swaps, c3, c4, dbl = s4_groups()
    struct_of = {}
    for p in c3:
        struct_of[p] = 'c3'
    for p in c4:
        struct_of[p] = 'c4'
    for p in dbl:
        struct_of[p] = 'dbl'
    for p in swaps:
        struct_of[p] = 'swap'
    struct_of[(0, 1, 2, 3)] = 'ident'
    return {f'T{i}': struct_of[p] for i, p in enumerate(train_perms)}


def build_u_struct(seed=42):
    """未见任务结构分组: U0-U11 ↔ build_rule_splits(seed) 的 test_perms 顺序。
    结构 = c3 (三轮换) / c4 (四轮换) / dbl (双对换) / swap / ident。"""
    train_perms, test_perms = build_rule_splits(seed)
    identity, swaps, c3, c4, dbl = s4_groups()
    struct_of = {}
    for p in c3:
        struct_of[p] = 'c3'
    for p in c4:
        struct_of[p] = 'c4'
    for p in dbl:
        struct_of[p] = 'dbl'
    for p in swaps:
        struct_of[p] = 'swap'
    struct_of[(0, 1, 2, 3)] = 'ident'
    return {f'U{i}': struct_of[p] for i, p in enumerate(test_perms)}


def tome_fraction(tasks):
    """TOME 地板: 程序结束后 target 后 4 token 全为 TOME (201) 的比例。
    full-8 acc 的地板 = tome_frac/2 (8 token 中 4 个白拿) — V31.2/V32
    "~0.39 未见天花板" 的机制解释 (≈0.775/2 = 0.3875)。"""
    tot = tome = 0
    for n, t in tasks.items():
        for (ctx, tgt) in t['query']:
            tot += 1
            if int(tgt[4]) == 201:
                tome += 1
    return round(tome / max(tot, 1), 4)


def summarize(results):
    rows = {}
    for r in results:
        ua = r['unseen_acc_after']
        rd = r['readout0_unseen']
        by = r['unseen_by_struct']
        rows[r['cell']] = {
            'label': r['label'], 'time_s': r['time_s'],
            'factors': r['factors'],
            'train_mean': round(_mean(r['train_acc_after']), 4),
            'unseen_mean': round(_mean(ua), 4),
            'rd0_mean': round(_mean(rd), 4),
            'unseen_minus_rd0': round(_mean(ua) - _mean(rd), 4),
            'c3': by.get('c3'), 'c4': by.get('c4'), 'dbl': by.get('dbl'),
            'c3_minus_c4': round(by['c3'] - by['c4'], 4)
            if by.get('c3') is not None and by.get('c4') is not None else None,
            'beta_mean': r['head_stats']['beta_mean'],
            'aux': r['aux'],
        }
    return rows


def hypothesis_verdicts(rows):
    """按预注册阈值判定 H1-H6 (缺失 cell 返回 N/A)。"""
    def _g(k):
        return rows.get(k, {}).get('unseen_mean')

    c0, r1 = _g('C0'), _g('32.1')
    v = {}
    if c0 is None or r1 is None:
        v['H1'] = ('N/A', 'N/A', None, 0.05)
    else:
        v['H1'] = _threshold(r1 - c0, 0.02, 0.05,
                             '32.1 regs_acc vs C0 regs_acc')
    for h, a, b in (('H2', '32.2', '32.1'), ('H3', '32.3', '32.1'),
                    ('H4', '32.4', '32.1')):
        va, vb = _g(a), _g(b)
        if va is None or vb is None:
            v[h] = ('N/A', 'N/A', None, 0.03)
        else:
            v[h] = _threshold(va - vb, 0.01, 0.03, f'{a} vs {b} regs_acc')
    g1 = rows.get('32.1', {}).get('c3_minus_c4')
    g5 = rows.get('32.5', {}).get('c3_minus_c4')
    dg = (g5 - g1) if (g1 is not None and g5 is not None) else None
    if dg is None:
        v['H5'] = ('N/A', 'N/A', None, 0.03)
    else:
        v['H5'] = _threshold(dg, 0.0, 0.03,
                             '32.5 c3−c4 差距收窄 (vs 32.1)')
    cands = [rows.get(k, {}).get('unseen_mean')
             for k in ('32.1', '32.2', '32.3', '32.4')]
    best = max([x for x in cands if x is not None], default=None)
    r9 = _g('32.9')
    if best is None or r9 is None:
        v['H6'] = ('N/A', 'N/A', None, 0.02)
    else:
        v['H6'] = _threshold(r9 - best, 0.0, 0.02,
                             '32.9 ≥ max(单因子)')
    return v


def _threshold(d, lo, hi, desc):
    if d > hi:
        return (desc, '成功', round(d, 4), hi)
    if d >= lo:
        return (desc, '部分成功', round(d, 4), hi)
    return (desc, '失败', round(d, 4), hi)


def write_report(results, args):
    rp = ROOT / "results" / "v32_1_matrix_report.json"
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                  encoding='utf-8')
    rows = summarize(results)
    hv = hypothesis_verdicts(rows)

    # TOME 地板 (机制发现, 数据实时计算)
    from run_v31_2_long_stream import build_v32_tasks
    _tw = build_v32_tasks(200, n_support=40, n_query=20, seed=42, window=3)
    _tome = tome_fraction(_tw)
    _floor = round(_tome / 2, 4)
    U_STRUCT = build_u_struct(42)

    L = []
    A = L.append
    A("# V32.1-32.9 矩阵验证报告 — 表示瓶颈五方向消融")
    A("")
    A("> 计划: `docs/wiki/v32-1-matrix-plan.md` (预注册) · 运行时间: "
      f"{time.strftime('%Y-%m-%d %H:%M')}")
    A(f"> 参数: meta_iters={args.meta_iters} K={args.K} "
      f"steps_per_task={args.steps_per_task} seed=42 · 同 Stage1 缓存")
    A("")
    A("## 0. 大白话总结")
    A("")
    A("- **做了什么**: 把 V32 的 8-token 预测目标换成纯寄存器目标 (regs-only), "
      "并叠加四个表示级机制 (规则身份预测 / 原型对比 / 独立规则编码器 EHS / 更长观察窗口), "
      "10 个 cell 消融, 直攻 V32 撞墙的 ~0.39 未见规则天花板。")
    A("- **发现什么**: (结果落地后填)")
    A("- **为什么重要**: 前四个版本家族已排除 head 侧一切机制; 本矩阵是第一次直接改"
      " RLN 的学习目标与架构, 决定论文主线 EHS 是否成立。")
    A("")
    A(f"## 0.5 重要机制发现: 8-token full acc 的 TOME 地板 (冒烟阶段)")
    A("")
    A(f"- 数据统计: 世界任务里 **{_tome:.1%}** 的 target 后 4 个 token 是 TOME(201) "
      "(程序结束后指令位置填充)。")
    A(f"- 推论: 只要模型学会输出 \"201\", 8 个 token 里就有 4 个\"白拿\" → "
      f"**full-8 acc 地板 = {_floor:.4f}**。")
    A(f"- 与历史对照: V31.2 S2 unseen 0.386 / V32 C1 0.376 ≈ {_floor:.4f} — "
      "**V31.2/V32 的 \"~0.39 未见天花板\" 主要就是 TOME 地板, 不是规则学习水平**。")
    A("- 真实信号: 用 regs_acc (前 4 寄存器 token) 度量, C0 (≡V32 C1 协议) 只有 "
      "~0.05-0.10 → 表示瓶颈比 V31/V32 报告的数字更严重; 本矩阵全部用 regs_acc "
      "统一度量, 预注册阈值 (C0+0.05 / 32.1+0.03) 正是在这个真实尺度上。")
    A("")
    A("## 1. 背景与立论")
    A("")
    A("- V31.0/1/2 + V32 四层一致: head 侧一切干预撞 ~0.39 未见天花板 "
      "(`results/v32_intent_report.md`, `docs/wiki/rln-mechanism.md`)")
    A("- V32 关键发现: 零步 goal readout ≥ 60 步适应 (所有 cell 60s−rd0 ≤ 0) "
      "→ 表示在适应前就应可读, 瓶颈在 RLN 表示本身")
    A("- 机制归因 (rln-mechanism.md 事实 1-3): 线性递推 + 单 last-hidden 出口装不下"
      " 状态内容+规则索引双信息; tgt 4 规则 token + 4 指令 token 混合稀释监督")
    A("")
    A("## 2. 实验设计 (预注册)")
    A("")
    A("| cell | 因子 | 核心改动 |")
    A("|:--|:--|:--|")
    for cell, cfg in CELLS.items():
        fac = '+'.join(f for f, on in {
            'R': cfg['n_resp'] == 4,
            'P': cfg['use_perm_aux'], 'C': cfg['use_contrast'],
            'E': cfg['use_rule_enc'], 'W': cfg['window'] == 4,
        }.items() if on)
        A(f"| {cell} | {fac or '—'} | {cfg['label']} |")
    A("")
    A("| 指标 | 失败 | 部分成功 | 成功 |")
    A("|:--|:--|:--|:--|")
    A("| 32.1 unseen regs_acc vs C0 | ≤ C0 | +0.02~0.05 | > +0.05 |")
    A("| 32.2/3/4 vs 32.1 | ≤ 32.1 | +0.01~0.03 | > +0.03 |")
    A("| 32.5 c3−c4 差距 vs 32.1 | 不缩窄 | <3pp | ≥3pp |")
    A("| 32.9 ≥ max(单因子) | 否 | 持平 | 严格更大 |")
    A("")
    A("## 3. 结果总表 (regs_acc 统一 4-token 协议)")
    A("")
    A("| cell | 因子 | Train | Unseen | rd0 | 60s−rd0 | c3 | c4 | dbl | c3−c4 | time_s |")
    A("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cell in CELLS:
        if cell not in rows:
            continue
        s = rows[cell]
        fac = '+'.join(f for f in ('R', 'P', 'C', 'E', 'W')
                       if s['factors'].get(f))
        c34 = f"{s['c3_minus_c4']:.4f}" if s['c3_minus_c4'] is not None else "-"
        cc = lambda x: f"{x:.4f}" if x is not None else "-"
        A(f"| {cell} | {s['label']} | {s['train_mean']:.4f} | "
          f"{s['unseen_mean']:.4f} | {s['rd0_mean']:.4f} | "
          f"{s['unseen_minus_rd0']:+.4f} | {cc(s['c3'])} | {cc(s['c4'])} | "
          f"{cc(s['dbl'])} | {c34} | {s['time_s']:.0f} |")
    A("")
    A("### 假设判定 (预注册阈值)")
    A("")
    A("| 假设 | 判定 | 差值 | 阈值 |")
    A("|:--|:--|--:|--:|")
    for k in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
        desc, verdict, d, hi = hv[k]
        ds = f"{d:+.4f}" if d is not None else "-"
        A(f"| {k} | **{verdict}** | {ds} | {hi} |")
    A("")
    A("## 4. 逐任务明细 (未见规则 regs_acc)")
    A("")
    for cell in CELLS:
        if cell not in rows:
            continue
        s = rows[cell]
        A(f"### {cell} ({s['label']})")
        A("")
        r = {x['cell']: x for x in results}[cell]
        A("| 任务 | 结构 | 60步段后 | rd0 | off-stream 曲线 0/10/20/30/40 |")
        A("|:--|:--|--:|--:|:--|")
        for n in r['unseen_acc_after']:
            struct = U_STRUCT.get(n, '?')
            curve = r['unseen_curve_offstream'][n]
            A(f"| {n} | {struct} | {r['unseen_acc_after'][n]:.4f} | "
              f"{r['readout0_unseen'][n]:.4f} | "
              f"{'→'.join(f'{x:.2f}' for x in curve)} |")
        A("")
    A("## 5. 辅助指标")
    A("")
    for cell in CELLS:
        r = {x['cell']: x for x in results}.get(cell)
        if not r:
            continue
        aux = r['aux']
        if aux:
            A(f"- **{cell}**: {json.dumps(aux, ensure_ascii=False)}")
    A("")
    A("## 6. 假阳性检查")
    A("")
    A("- **活跃基线 (实现等价)**: C0 = V32 C1 同协议同缓存 (80 iters/K=20/seed 42), "
      "应复现 V32 C1 full-8 acc (0.4346/0.3867/rd0 0.3779, `results/v32_C1.json` 重跑) — "
      "full acc 的 TOME 地板解释见 §0.5; regs_acc 为本矩阵新增统一主指标")
    A("- **评估独立性**: 主指标 regs_acc 与训练 loss 同源但取 argmax 精确匹配; "
      "rd0 用纯共享 W (零适应)")
    A("- **单一变量**: 32.2/3/4/6/7/8/9 与 32.1 同世界同 window; 仅 32.5 换 window=4")
    A("")
    A("## 7. 结论")
    A("")
    A("(结果落地后填: 每条 证据→解读→通俗)")
    A("")
    A("## 8. 复现")
    A("")
    A("```bash")
    A(f"python tests/run_v32_1_matrix.py --meta-iters {args.meta_iters} "
      f"--K {args.K} --steps-per-task {args.steps_per_task}")
    A("```")
    A("")
    A("- Stage1 缓存: /tmp/v32_s1_state.pt (优先) / /tmp/v31_2_s1_state.pt; "
      "缺则自动重训 (2 epochs, ~15 min)")
    A("- 结果 JSON 可再生: `results/v32_1_matrix_report.json` (gitignored)")

    mp = ROOT / "results" / "v32_1_matrix_report.md"
    mp.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"\nJSON 报告: {rp}\nMD 报告: {mp}")
    return rows, hv


# ============================================================
# 主流程
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-iters', type=int, default=80)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--steps-per-task', type=int, default=60)
    parser.add_argument('--cells', type=str,
                        default='C0,32.1,32.2,32.3,32.4,32.5,32.6,32.7,32.8,32.9')
    parser.add_argument('--result', type=str, default='v32_1_matrix_report.json')
    parser.add_argument('--no-stage1', action='store_true',
                        help='管线冒烟用: 随机 RLN 初始化, 不加载/重训 Stage1')
    args = parser.parse_args()

    print("=" * 70)
    print("V32.1-32.9 矩阵验证 — 表示瓶颈五方向消融")
    print("=" * 70)

    WORLD_V = 200
    tasks_w3 = build_v32_tasks(WORLD_V, n_support=40, n_query=20,
                               seed=42, window=3)
    tasks_w4 = build_v32_tasks(WORLD_V, n_support=40, n_query=20,
                               seed=42, window=4)
    train_rules_all = [n for n in tasks_w3 if tasks_w3[n]['seen']]
    unseen_all = [n for n in tasks_w3 if not tasks_w3[n]['seen']]
    print(f"V={WORLD_V}, 训练规则={len(train_rules_all)}, "
          f"未见规则={len(unseen_all)}")

    # 未见任务结构分组 (build_rule_splits 确定性: U0-U11 ↔ test_perms 顺序)
    U_STRUCT = build_u_struct(42)
    print(f"未见结构: {U_STRUCT}")

    # Stage1 缓存: 兼容 v32/v31_2 两个路径
    s1_state = None
    if args.no_stage1:
        print("\n--no-stage1: 随机 RLN 初始化 (仅管线冒烟)")
    else:
        S1_CACHE = None
        for cand in ("/tmp/v32_s1_state.pt", "/tmp/v31_2_s1_state.pt"):
            if os.path.exists(cand):
                S1_CACHE = cand
                break
        if S1_CACHE is None:
            S1_CACHE = "/tmp/v32_s1_state.pt"
            os.makedirs(os.path.dirname(S1_CACHE), exist_ok=True)
            print("\n>>> Stage1 预训练 (12 条训练规则 token 流)...")
            random.seed(42); torch.manual_seed(42)
            ppl, s1_model = stage1_pretrain_world(tasks_w3, train_rules_all,
                                                  WORLD_V, n_epochs=2)
            s1_state = {k: v.cpu().clone()
                        for k, v in s1_model.state_dict().items()}
            torch.save(s1_state, S1_CACHE)
            print(f"    Stage1 PPL = {ppl:.2f}, 已缓存到 {S1_CACHE}")
        else:
            s1_state = torch.load(S1_CACHE)
            s1_state = {k: v.cpu().clone() for k, v in s1_state.items()}
            print(f"\nStage1: 从缓存加载 ({S1_CACHE})")

    keys = [c.strip() for c in args.cells.split(',')]
    results = []
    # 增量 checkpoint: 崩溃后已完成 cell 不丢, 支持续跑
    ckpt = ROOT / "results" / "v32_1_matrix_checkpoint.json"
    if ckpt.exists():
        try:
            done = {r['cell']: r for r in
                    json.loads(ckpt.read_text(encoding='utf-8'))}
        except Exception:
            done = {}
        print(f"checkpoint: {sorted(done.keys())} 已完成")
    else:
        done = {}
    for key in keys:
        if key in done:
            print(f"跳过已完成 cell: {key}")
            results.append(done[key])
            continue
        if key not in CELLS:
            print(f"未知 cell: {key}")
            continue
        cfg = CELLS[key]
        print(f"\n{'='*60}\n  实验: {cfg['label']} "
              f"(n_resp={cfg['n_resp']}, window={cfg['window']}, "
              f"P={cfg['use_perm_aux']}, C={cfg['use_contrast']}, "
              f"E={cfg['use_rule_enc']})\n{'='*60}")
        try:
            r = run_cell(key, cfg, WORLD_V, s1_state, tasks_w3, tasks_w4,
                         U_STRUCT, n_meta_iters=args.meta_iters, K=args.K,
                         seed=42, steps_per_task=args.steps_per_task)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! cell {key} 失败: {e} (继续后续 cell)")
            continue
        results.append(r)
        ckpt.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
        print(f"checkpoint 更新: {key} 完成 ({len(results)}/{len(keys)})",
              flush=True)

    if results:
        rows, hv = write_report(results, args)
        print(f"\n{'='*60}\n汇总:\n{'='*60}")
        print(f"{'cell':<6}{'Train':<8}{'Unseen':<10}{'rd0':<8}"
              f"{'60s-rd0':<10}{'c3-c4':<10}{'time_s':<8}")
        for cell in CELLS:
            if cell not in rows:
                continue
            s = rows[cell]
            c34 = f"{s['c3_minus_c4']:.4f}" if s['c3_minus_c4'] is not None else "-"
            print(f"{cell:<6}{s['train_mean']:<8.4f}{s['unseen_mean']:<10.4f}"
                  f"{s['rd0_mean']:<8.4f}{s['unseen_minus_rd0']:<10.4f}"
                  f"{c34:<10}{s['time_s']:<8.0f}")
        print("\n假设判定:")
        for k in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
            desc, verdict, d, hi = hv[k]
            print(f"  {k}: {verdict} ({d} vs 阈值 {hi})")
    return results


if __name__ == "__main__":
    main()
