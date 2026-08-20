"""
V33: Define-Route 循环 — OaK 式持续学习 × INTACT 意图路由
==========================================================
计划: docs/wiki/v33-plan.md (预注册 H-A..H-E)

立论 (用户重新规划 2026-08-07):
  h 的问题不是容量, 是"循环和路由没有确立":
  - Define 层缺失: 经验没有显式的"抽象成可学习结构"环节 (单次聚合 ≠ define,
    define 必须可迭代: 定义 → 重解释经验 → 按误差修正定义)
  - Route 层缺失: 信息没有路由分派, 规则与内容挤在同一向量
  INTACT 重定位: 意图插入 = define, 选择动作 = route; V32 失败根因 =
  只做 define 输出注入 (head 侧被 W 适应洗掉), 没做路由进表示 (递推侧)

三圈循环:
  循环1 (任务内, 表示级): define 的 m 调制 SSM 递推 dt/B (路由进递推)
  循环2 (任务内, 定义级): define-refine — m₀ 解释经验 → 检验误差 → 修正 m₁
  循环3 (跨任务, 能力级): 模式库 — 入库/检索匹配/叠加 (持续学习)

细胞 (8):
  B0  基线 旧世界 (≡ V32.1 32.1, 复现 0.1016)
  W0  基线 新世界 (TOME 截断, 世界修复单独效果)
  A   +define-refine (循环2)
  B   +路由进递推 (循环1)
  C   +动作路由 top-k (INTACT 选择语义)
  D   +模式库 (循环3)
  E   全闭环 (旧世界)
  EW  全闭环 (新世界)

控制变量 (与 V32.1 一致): 世界 S4 24 排列 · Stage1 同缓存 (V32.1 重训, PPL 12.50)
  meta 80 iters · K=20 · 2 tasks/step · Adam 1e-3 · clip 1.0 · seed 42
  SwiftTD β_init=1e-3 κ=0.1 η=0.1 ϵ=0.99 · 长流 16×60 · sleep_iters=2
主指标: unseen regs_acc (4-token) · rd0 · 结构分组 c3/c4/dbl · 遗忘 · β
  aux: define 探针 (m → perm 组成性 acc) · 路由一致性 (同规则 m 近) · 模式库统计

实现要点:
  - 路由进递推: SSM_Layer_V30_3.forward(x, cond) — cond_dt/cond_b 零初始化,
    Stage1 缓存直接复用 (无需重训)
  - define-refine: 检验用 pln.clone_params() (共享头) 的预测误差对 m 的梯度
    (detached), refine_net([m; g]) → 修正; goal detached / local attached 双条件
  - 动作路由: top-k 候选 logit += λ·(m·embed(cand)) — 意图-动作匹配度 (INTACT 选择语义)
  - 模式库: cosine 检索 τ=0.5, 匹配则叠加 α=0.5, 否则入库
  - B0/W0 用 V32_1_Learner 原样 (严格复现 32.1), A-E 用 V33_Learner
"""

import os, sys, time, json, math, random
from typing import Optional
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
from run_v32_1_matrix import (
    V32_1_Learner, build_model_resp, make_regs_tasks, build_u_struct,
    build_t_struct,
    tome_fraction, CELLS as V321_CELLS,
)

# ============================================================
# 细胞定义
# ============================================================
CELLS = {
    'B0': dict(world='old', define_refine=0, route_recur=False,
               route_action=False, bank=False,
               label='B0-baseline-old'),
    'W0': dict(world='new', define_refine=0, route_recur=False,
               route_action=False, bank=False,
               label='W0-baseline-new'),
    'A':  dict(world='old', define_refine=2, route_recur=False,
               route_action=False, bank=False,
               label='A-define-refine'),
    'B':  dict(world='old', define_refine=0, route_recur=True,
               route_action=False, bank=False,
               label='B-route-recur'),
    'C':  dict(world='old', define_refine=0, route_recur=False,
               route_action=True, bank=False,
               label='C-route-action'),
    'D':  dict(world='old', define_refine=0, route_recur=False,
               route_action=False, bank=True,
               label='D-pattern-bank'),
    'E':  dict(world='old', define_refine=2, route_recur=True,
               route_action=True, bank=True,
               label='E-full-loop'),
    'EW': dict(world='new', define_refine=2, route_recur=True,
               route_action=True, bank=True,
               label='EW-full-loop-new'),
    # ── V33.1: OaK 放大成功 (世界修复 × define-refine) ──
    'WA':  dict(world='new', define_refine=2, route_recur=False,
                route_action=False, bank=False,
                label='WA-world-fix-x-refine'),
    'WAC': dict(world='new', define_refine=2, route_recur=False,
                route_action=True, bank=False,
                label='WAC-world-fix-x-refine-x-route-act'),
    'WADb': dict(world='new', define_refine=2, route_recur=False,
                 route_action=False, bank=True, bank_tau=0.7,
                 bank_struct=True,
                 label='WADb-world-fix-x-refine-x-bank-struct'),
    # ── V33.2: redefine MLP 参数扩展 (宽/深) ──
    'WAw': dict(world='new', define_refine=2, route_recur=False,
                route_action=False, bank=False, refine_hidden=512,
                label='WAw-refine-wide-512'),
    'WAd': dict(world='new', define_refine=2, route_recur=False,
                route_action=False, bank=False, refine_layers=3,
                label='WAd-refine-deep-3'),
    'WAwd': dict(world='new', define_refine=2, route_recur=False,
                 route_action=False, bank=False, refine_hidden=512,
                 refine_layers=3,
                 label='WAwd-refine-wide-deep'),
    # ── V33.3: DRLN 魔改 — 不冻结 / 多频更新 (基线待 V33.1/33.2 定, 先挂 WA 配置) ──
    'WAf': dict(world='new', define_refine=2, route_recur=False,
                route_action=False, bank=False, rln_online='all',
                rln_lr=1e-4, rln_every=10,
                label='WAf-rln-unfrozen'),
    'WAmf': dict(world='new', define_refine=2, route_recur=False,
                 route_action=False, bank=False, rln_online='multi',
                 rln_fast_lr=5e-4, rln_fast_every=5,
                 rln_slow_lr=1e-4, rln_slow_every=20,
                 label='WAmf-rln-multifreq'),
    # ── V33.4: 训练饱和测试 (W0 长训练, 验证 0.0977 是否未饱和) ──
    'W0l': dict(world='new', define_refine=0, route_recur=False,
                route_action=False, bank=False, meta_iters=160,
                label='W0l-longtrain-160'),
}


# ============================================================
# 模式库 (循环3)
# ============================================================
class PatternBank:
    """跨任务能力叠加: 显式存储 define 的模式, 检索匹配则融合, 否则新增。
    struct 分组 (V33.1 反坍缩): 只检索同规则结构 (identity/swaps/c3/c4/dbl) 的键,
    避免所有任务 fuse 到同一个原型 (V33 D cell n_keys=1-5 的坍缩修复)。"""

    def __init__(self, d, tau=0.5, alpha=0.5, struct_groups=False):
        self.d = d
        self.tau = tau
        self.alpha = alpha
        self.struct_groups = struct_groups
        self.keys = []          # list of (归一化 m, 任务名, struct)
        self.n_add = 0
        self.n_fuse = 0

    def _norm(self, m):
        return m / (m.norm() + 1e-8)

    def retrieve(self, m, struct=None):
        if not self.keys:
            return None, -1.0
        mn = self._norm(m)
        best, best_s = None, -1.0
        for k, _, ks in self.keys:
            if self.struct_groups and struct is not None and ks != struct:
                continue
            s = float((mn * k).sum())
            if s > best_s:
                best_s, best = s, k
        if best_s > self.tau:
            return best, best_s
        return None, best_s

    def add(self, m, name, struct=None):
        self.keys.append((self._norm(m.detach()), name, struct))
        self.n_add += 1

    def fuse(self, m, matched):
        self.n_fuse += 1
        return self.alpha * matched.detach() + (1 - self.alpha) * m

    def stats(self):
        return {'n_keys': len(self.keys), 'n_add': self.n_add,
                'n_fuse': self.n_fuse}


# ============================================================
# V33_Learner
# ============================================================
class V33_Learner(V32_1_Learner):
    """V32.1 + DRL 循环: define-refine (循环2) × 路由进递推 (循环1) ×
    动作路由 (选择语义) × 模式库 (循环3)。"""

    def __init__(self, rln, pln, V, K=20, outer_lr=1e-3,
                 n_tasks_per_step=2, seed=42,
                 beta_init=1e-3, kappa=0.1, eta=0.1, eps=0.99,
                 init_lr=0.0, sleep_iters=2,
                 n_resp=4, use_intent=True, use_rule_enc=False,
                 use_perm_aux=False, use_contrast=False,
                 n_define_refine=0, use_route_recur=False,
                 use_route_action=False, use_pattern_bank=False,
                 lambda_route=0.3, lambda_probe=0.3,
                 refine_sup=4, bank_tau=0.5, bank_alpha=0.5,
                 bank_struct=False, refine_hidden=None,
                 refine_layers=2):
        super().__init__(rln, pln, V, K=K, outer_lr=outer_lr,
                         n_tasks_per_step=n_tasks_per_step, seed=seed,
                         beta_init=beta_init, kappa=kappa, eta=eta, eps=eps,
                         init_lr=init_lr, sleep_iters=sleep_iters,
                         n_resp=n_resp, use_intent=use_intent,
                         use_rule_enc=use_rule_enc,
                         use_perm_aux=use_perm_aux,
                         use_contrast=use_contrast)
        d = rln.d_model
        self.n_define_refine = n_define_refine
        self.use_route_recur = use_route_recur
        self.use_route_action = use_route_action
        self.use_pattern_bank = use_pattern_bank
        self.lambda_route = lambda_route
        self.lambda_probe = lambda_probe
        self.refine_sup = refine_sup
        self.lambda_icl = 0.0
        # 循环2: 定义修正网络 + define 探针 (诊断 + 定义质量监督)
        # V33.2: refine MLP 参数扩展 (hidden 宽度 / 层数)
        rh = refine_hidden or d
        self.refine_net = nn.Sequential()
        self.refine_net.append(nn.Linear(2 * d, rh))
        self.refine_net.append(nn.Tanh())
        for _ in range(refine_layers - 2):
            self.refine_net.append(nn.Linear(rh, rh))
            self.refine_net.append(nn.Tanh())
        self.refine_net.append(nn.Linear(rh, d))
        self.refine_hidden = rh
        self.refine_layers = refine_layers
        self.probe_head = nn.Linear(d, 16)
        extra = (list(self.refine_net.parameters())
                 + list(self.probe_head.parameters()))
        # 循环3: 模式库
        if use_pattern_bank:
            self.bank = PatternBank(d, tau=bank_tau, alpha=bank_alpha,
                                    struct_groups=bank_struct)
        else:
            self.bank = None
        # 任务结构标签 (bank 分组检索用): {任务名: struct}
        self._task_struct = {}
        # 重建 outer_opt (含 refine/probe; intent_net 已含在 V32_1 的 extra)
        base_params = (list(rln.parameters()) + list(pln.parameters())
                       + list(self.intent_net.parameters()))
        if use_rule_enc:
            base_params = (list(rln.parameters()) + list(pln.parameters())
                           + list(self.rule_encoder.parameters()))
        self.outer_opt = torch.optim.Adam(
            base_params + extra, lr=outer_lr)
        self.probe_trace = []

    # ── 前向路由 (循环1) ──────────────────────────────────
    def _rln_fwd(self, x, m=None):
        if self.use_route_recur and m is not None:
            return self.rln(x, cond=m)
        return self.rln(x)

    # ── Define 层 (循环2) ─────────────────────────────────
    def _post_define_aux_loss(self, m, t) -> Optional[torch.Tensor]:
        """ICL 偏置 hook. 默认关闭 (返回 None). 子类覆写返回附加 loss 张量."""
        m, t = m, t
        return None

    def _define(self, task, detach=True, use_bank=True):
        """goal define: 支持集聚合 Δregs → define_net → refine 迭代。
        detach=True (部署): 全 stop-grad; detach=False (训练): 带图。"""
        sup = task['support']
        B = min(len(sup), 16)
        cs = torch.stack([s[0] for s in sup[:B]])
        ts = torch.stack([s[1] for s in sup[:B]])
        delta = self._dregs(cs, ts)                # (d,)
        m = self.m_norm(self._rule_vec(delta))
        if self.n_define_refine > 0:
            m = self._refine(m, task)
        if use_bank and self.bank is not None:
            m = self._bank_fuse(m, task.get('name', '?'))
        if detach:
            m = m.detach()
        return m

    def _refine(self, m, task):
        """循环2: 定义 → 用定义重解释经验 → 检验误差修正定义。
        检验: 共享头 (clone_params) 在 support 前几个样本的预测误差对 m 的梯度
        (detached 一阶信号, 不做二阶图); refine_net([m; g]) → 修正量。"""
        W = [w.detach() for w in self.pln.clone_params()]
        for _ in range(self.n_define_refine):
            g = 0.0
            for (c, tg) in task['support'][:self.refine_sup]:
                h = self._rln_fwd(c.unsqueeze(0), m)[:, -1, :]
                hin = self._h_in_batch(h, m)
                pred = head_forward(W, hin, self.intent_dim,
                                    self.world_vocab, self.n_resp)
                loss = F.cross_entropy(
                    pred.view(-1, self.world_vocab),
                    tg.view(-1).clamp(0, self.V + 1))
                gi = torch.autograd.grad(loss, m, create_graph=True)[0]
                g = g + gi.detach()
            corr = self.refine_net(torch.cat([m.detach(), g], dim=-1))
            m = self.m_norm(m + corr)
        return m

    def _local_define(self, c, tg):
        """local call (attached): per-sample Δregs 单步定义。"""
        e_t = self.rln.embed(tg[:4].unsqueeze(0))
        e_c = self.rln.embed(c[-8:-4].unsqueeze(0))
        delta = (e_t - e_c).mean(dim=1).mean(dim=0)
        return self.m_norm(self._rule_vec(delta))

    def _bank_fuse(self, m, name):
        if self.bank is None:
            return m
        struct = self._task_struct.get(name)
        matched, sim = self.bank.retrieve(m, struct=struct)
        if matched is not None:
            return self.bank.fuse(m, matched)
        self.bank.add(m, name, struct=struct)
        return m

    # ── 动作路由 (INTACT 选择语义) ───────────────────────
    def _route_logits(self, logits, m, perm_pred=None):
        """top-k 候选重加权: logit += λ·(m·embed(cand)) — 意图-动作匹配度。
        只作用于每位置 top-8 候选 (低概率候选不受影响).
        V35.10: perm_pred (4,4) 置换矩阵 — 若给出, 按 perm 重映射 logits
        的寄存器位置 (静态代数: 组合 perm 已知时, 预测 = 重映射后的
        常规预测). 部署时用 probe 回读的 perm."""
        if not self.use_route_action or m is None:
            return logits
        with torch.no_grad():
            ce = self.rln.embed.weight.detach()    # (vocab, d)
        route = m @ ce.t()                          # (vocab,)
        out = logits.clone()
        for p in range(self.n_resp):
            _, idx = logits[:, p, :].topk(8, dim=-1)
            out[:, p, :].scatter_add_(
                1, idx, self.lambda_route * route[idx])
        return out

    # ── 可微内循环 (cond 透传) ────────────────────────────
    def adapt_graph(self, support, K=None, m=None):
        K = K or self.K
        W = self.pln.clone_params()
        for (ctx, tgt) in support[:K]:
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
        return W

    def adapt_online(self, W, support, n_steps, m=None):
        for (ctx, tgt) in support[:n_steps]:
            with torch.no_grad():
                hidden = self._rln_fwd(ctx.unsqueeze(0), m)
            h = hidden[:, -1, :]
            hin = self._h_in_batch(h, m)
            x = torch.tanh(hin @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(
                1, self.n_resp, self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            self.head.step(W, hin[0], x[0], errs)
        return W

    # ── 外循环 (BPTT) ────────────────────────────────────
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
            # Define (attached, 训练): 意图定义 + 模式库融合
            m = self._define(t, detach=False) if self.use_intent else None
            W = self.adapt_graph(t['support'], m=m, K=K)
            Ws.append([w.detach() for w in W])
            ctx, tgt = batch_pairs(t['query'], max_batch=16)
            hidden = self._rln_fwd(ctx, m)
            h = hidden[:, -1, :]
            h_in = self._h_in_batch(h, m)
            pred = head_forward(W, h_in, self.intent_dim,
                                self.world_vocab, self.n_resp)
            pred = self._route_logits(pred, m)
            loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                   tgt.view(-1).clamp(0, self.V + 1))
            total_loss = total_loss + loss / len(chosen)

            # ICL 偏置 hook (默认 None = 不启用; compose 子类覆写):
            # 用未适配共享头 W0 直接读组合后的 m, 逼 m 在纯 ICL 下暴露组合规则
            iclx = self._post_define_aux_loss(m, t)
            if iclx is not None:
                total_loss = total_loss + self.lambda_icl * iclx / len(chosen)

            # local call (attached, 单步定义)
            if self.use_intent and self.lambda_local > 0:
                for (c, tg) in t['support'][:self.n_local_sup]:
                    hsv = self._rln_fwd(c.unsqueeze(0), m)[:, -1, :]
                    m_loc = self._local_define(c, tg)
                    hin = self._h_in_batch(hsv, m_loc)
                    pl = head_forward(W, hin, self.intent_dim,
                                      self.world_vocab, self.n_resp)
                    total_loss = total_loss + self.lambda_local * \
                        F.cross_entropy(pl.view(-1, self.world_vocab),
                                        tg.view(-1).clamp(0, self.V + 1)) \
                        / (len(chosen) * self.n_local_sup)

            # Define 质量监督: probe 从 m (单向量聚合定义) 预测 perm (4×4 组成性)
            # V35.10: 触发条件与 n_define_refine 解耦 (V33 bug: refine=0 时
            # probe 监督静默失效); 目标 = 任务真实 perm (含 AQ 组合任务
            # 的 unseen c4 perm — 回读学会组合置换结构)
            if m is not None:
                perm = torch.tensor(t['perm'], dtype=torch.long)
                l_probe = F.cross_entropy(
                    self.probe_head(m).view(4, 4), perm)
                total_loss = total_loss + self.lambda_probe * l_probe \
                    / len(chosen)
                self.probe_trace.append(l_probe.item())

            # P/C 兼容 (V32.1 因子, 默认关)
            if self.use_perm_aux:
                aux = self.aux_head(h)
                perm = torch.tensor(t['perm'], dtype=torch.long)
                tl = perm.unsqueeze(0).expand(h.shape[0], -1).reshape(-1)
                total_loss = total_loss + self.lambda_perm * \
                    F.cross_entropy(aux.view(-1, 4), tl) / len(chosen)
            if protos is not None:
                qn = h / (h.norm(dim=1, keepdim=True) + 1e-8)
                sim = qn @ protos.t()
                tid = train_rules.index(rname)
                tl = torch.full((h.shape[0],), tid, dtype=torch.long)
                total_loss = total_loss + self.lambda_contra * \
                    F.cross_entropy(sim, tl) / len(chosen)

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

    # ── 评估 (统一 regs_acc) ──────────────────────────────
    def _metric(self, W, task, m=None, max_batch=16):
        ctx, tgt = batch_pairs(task['query'], max_batch=max_batch)
        with torch.no_grad():
            hidden = self._rln_fwd(ctx, m)
        h = hidden[:, -1, :]
        h_in = self._h_in_batch(h, m)
        pred = head_forward([w.detach() for w in W], h_in,
                            self.intent_dim, self.world_vocab, self.n_resp)
        pred = self._route_logits(pred, m)
        pt = pred.argmax(dim=-1)
        full = (pt == tgt[:, :self.n_resp]).sum().item() / \
            max(tgt[:, :self.n_resp].numel(), 1)
        regs = (pt[:, :4] == tgt[:, :4]).sum().item() / \
            max(tgt[:, :4].numel(), 1)
        return full, regs

    def eval_goal_readout(self, task, head_state=None):
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self._define(task, detach=True) if self.use_intent else None
        return self._metric(W, task, m=m)

    def eval_adaptation_curve(self, task, max_steps=40, eval_every=10,
                              head_state=None):
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self._define(task, detach=True) if self.use_intent else None
        curve = [self._metric(W, task, m=m)[1]]
        sup = task['support']
        for j in range(max_steps):
            if j < len(sup):
                with torch.no_grad():
                    hidden = self._rln_fwd(sup[j][0].unsqueeze(0), m)
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
                m_cache[tname] = (self._define(t, detach=True)
                                  if self.use_intent else None)
            m = m_cache[tname]
            seg = [self._metric(W, t, m=m)[1]]
            for j in range(steps_per_task):
                if j < len(t['support']):
                    with torch.no_grad():
                        hidden = self._rln_fwd(
                            t['support'][j][0].unsqueeze(0), m)
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
                    m_cache[s] = (self._define(tasks[s], detach=True)
                                  if self.use_intent else None)
                row[s] = round(self._metric(W, tasks[s],
                                            m=m_cache[s])[1], 4)
            forget_rows.append(row)
            if sleep_iters > 0 and len(seen) >= sleep_every:
                saved = self.head.state()
                for _ in range(sleep_iters):
                    self.meta_step(tasks, seen)
                self.head.load_state(saved)
        return {'acc_after': acc_after, 'acc_after_full': acc_after_full,
                'curves': curves, 'forget': forget_rows}

    # ── Define 探针诊断 ───────────────────────────────────
    def probe_perm_acc(self, tasks, names):
        """从 define 的 m 预测 perm (4×4 组成性), train/unseen 分报。"""
        ok = tot = 0
        for n in names:
            t = tasks[n]
            m = self._define(t, detach=True) if self.use_intent else None
            if m is None:
                continue
            with torch.no_grad():
                logits = self.probe_head(m)
                pred = logits.view(4, 4).argmax(dim=-1)
            perm = torch.tensor(t['perm'], dtype=torch.long)
            ok += (pred == perm).sum().item()
            tot += 4
        return round(ok / max(tot, 1), 4) if tot else None


# ============================================================
# V33_3_Learner: DRLN 魔改 — 不冻结 / 多频更新 (V33.3)
# ============================================================
class V33_3_Learner(V33_Learner):
    """V33.2 最佳组合 + RLN 在线多频更新 (ANML 分层更新思想 + 大脑多频节奏)。

    - 冻结基线: 长流只更新 head (SwiftTD), RLN 冻结 (V33 现状)
    - 不冻结 (WAf): 长流中 RLN 全部参数每 rln_every 步 SGD 小步长更新
    - 多频 (WAmf): 参数分两群 — embed (快, 每 n_fast 步, lr_fast) /
      ssm layers (慢, 每 n_slow 步, lr_slow) — 模拟大脑不同子系统更新节奏
    - 遗忘风险由遗忘矩阵监控 (在线更新可能破坏已学表示)
    """

    def __init__(self, rln, pln, V, K=20, outer_lr=1e-3,
                 n_tasks_per_step=2, seed=42,
                 beta_init=1e-3, kappa=0.1, eta=0.1, eps=0.99,
                 init_lr=0.0, sleep_iters=2,
                 n_resp=4, use_intent=True,
                 n_define_refine=0, use_route_recur=False,
                 use_route_action=False, use_pattern_bank=False,
                 lambda_route=0.3, lambda_probe=0.3,
                 refine_sup=4, bank_tau=0.5, bank_alpha=0.5,
                 bank_struct=False, refine_hidden=None,
                 refine_layers=2,
                 rln_online='off', rln_lr=1e-4, rln_every=10,
                 rln_fast_lr=5e-4, rln_fast_every=5,
                 rln_slow_lr=1e-4, rln_slow_every=20):
        super().__init__(rln, pln, V, K=K, outer_lr=outer_lr,
                         n_tasks_per_step=n_tasks_per_step, seed=seed,
                         beta_init=beta_init, kappa=kappa, eta=eta, eps=eps,
                         init_lr=init_lr, sleep_iters=sleep_iters,
                         n_resp=n_resp, use_intent=use_intent,
                         n_define_refine=n_define_refine,
                         use_route_recur=use_route_recur,
                         use_route_action=use_route_action,
                         use_pattern_bank=use_pattern_bank,
                         lambda_route=lambda_route,
                         lambda_probe=lambda_probe,
                         refine_sup=refine_sup,
                         bank_tau=bank_tau, bank_alpha=bank_alpha,
                         bank_struct=bank_struct,
                         refine_hidden=refine_hidden,
                         refine_layers=refine_layers)
        self.rln_online = rln_online      # 'off' | 'all' (不冻结) | 'multi' (多频)
        self.rln_lr = rln_lr
        self.rln_every = rln_every
        self.rln_fast_lr = rln_fast_lr
        self.rln_fast_every = rln_fast_every
        self.rln_slow_lr = rln_slow_lr
        self.rln_slow_every = rln_slow_every
        self._rln_opt = None
        self._rln_fast_opt = None
        self._rln_slow_opt = None
        self.rln_online_steps = 0

    def _setup_online_opts(self):
        if self.rln_online == 'all':
            self._rln_opt = torch.optim.SGD(
                [p for p in self.rln.parameters() if p.requires_grad],
                lr=self.rln_lr)
        elif self.rln_online == 'multi':
            self._rln_fast_opt = torch.optim.SGD(
                self.rln.embed.parameters(), lr=self.rln_fast_lr)
            slow = [p for lyr in self.rln.layers
                    for p in lyr.parameters() if p.requires_grad]
            self._rln_slow_opt = torch.optim.SGD(slow, lr=self.rln_slow_lr)

    def _online_rln_step(self, ctx, tgt, step_idx):
        """按频率更新 RLN (不冻结/多频)。返回是否执行了更新。"""
        if self.rln_online == 'off':
            return False
        if self.rln_online == 'all':
            if step_idx % self.rln_every != 0:
                return False
            if self._rln_opt is None:
                self._setup_online_opts()
            self._rln_opt.zero_grad()
            h = self._rln_fwd(ctx.unsqueeze(0))[:, -1, :]
            hin = self._h_in_batch(h, None if not self.use_intent
                                   else self.m_norm(
                                       self._rule_vec(self._dregs_single(ctx))))
            pred = head_forward([w.detach() for w in
                                 self.pln.clone_params()], hin,
                                self.intent_dim, self.world_vocab,
                                self.n_resp)
            loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                   tgt.view(-1).clamp(0, self.V + 1))
            loss.backward()
            self._rln_opt.step()
            self.rln_online_steps += 1
            return True
        if self.rln_online == 'multi':
            if self._rln_fast_opt is None:
                self._setup_online_opts()
            did = False
            if step_idx % self.rln_fast_every == 0:
                self._rln_fast_opt.zero_grad()
                h = self._rln_fwd(ctx.unsqueeze(0))[:, -1, :]
                m_ctx = (None if not self.use_intent else self.m_norm(
                    self._rule_vec(self._dregs_single(ctx))))
                hin = self._h_in_batch(h, m_ctx)
                pred = head_forward([w.detach() for w in
                                     self.pln.clone_params()], hin,
                                    self.intent_dim, self.world_vocab,
                                    self.n_resp)
                loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                       tgt.view(-1).clamp(0, self.V + 1))
                loss.backward()
                self._rln_fast_opt.step()
                did = True
            if step_idx % self.rln_slow_every == 0:
                # 独立重建计算图 (fast 分支的 backward 已释放共享中间值)
                self._rln_slow_opt.zero_grad()
                h = self._rln_fwd(ctx.unsqueeze(0))[:, -1, :]
                m_ctx = (None if not self.use_intent else self.m_norm(
                    self._rule_vec(self._dregs_single(ctx))))
                hin = self._h_in_batch(h, m_ctx)
                pred = head_forward([w.detach() for w in
                                     self.pln.clone_params()], hin,
                                    self.intent_dim, self.world_vocab,
                                    self.n_resp)
                loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                       tgt.view(-1).clamp(0, self.V + 1))
                loss.backward()
                self._rln_slow_opt.step()
                did = True
            self.rln_online_steps += 1
            return did
        return False

    def _dregs_single(self, ctx):
        """单样本 Δregs: 最近观察的寄存器 vs 目标寄存器 (在线更新用)。"""
        c = ctx[-8:-4].unsqueeze(0)
        e_c = self.rln.embed(c)
        return e_c.mean(dim=1).mean(dim=0)

    def long_stream(self, tasks, stream, steps_per_task=60,
                    sleep_every=1, sleep_iters=2, head_state=None,
                    eval_every=20):
        self._load_head(head_state)
        W = [w.detach() for w in self.pln.clone_params()]
        seen = []
        forget_rows = []
        acc_after = {}
        acc_after_full = {}
        curves = {}
        m_cache = {}
        step_idx = 0
        for tname in stream:
            t = tasks[tname]
            if tname not in m_cache:
                m_cache[tname] = (self._define(t, detach=True)
                                  if self.use_intent else None)
            m = m_cache[tname]
            seg = [self._metric(W, t, m=m)[1]]
            for j in range(steps_per_task):
                if j < len(t['support']):
                    with torch.no_grad():
                        hidden = self._rln_fwd(
                            t['support'][j][0].unsqueeze(0), m)
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
                    # V33.3: RLN 在线更新 (不冻结/多频)
                    self._online_rln_step(t['support'][j][0],
                                          t['support'][j][1], step_idx)
                    step_idx += 1
                if (j + 1) % eval_every == 0:
                    seg.append(self._metric(W, t, m=m)[1])
            seen.append(tname)
            curves[tname] = seg
            acc_after[tname] = seg[-1]
            acc_after_full[tname] = self._metric(W, t, m=m)[0]
            row = {}
            for s in seen:
                if s not in m_cache:
                    m_cache[s] = (self._define(tasks[s], detach=True)
                                  if self.use_intent else None)
                row[s] = round(self._metric(W, tasks[s],
                                            m=m_cache[s])[1], 4)
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
def run_cell(cell, cfg, V, s1_state, tasks_old, tasks_new, U_STRUCT,
             n_meta_iters=80, K=20, seed=42, steps_per_task=60,
             verbose=True):
    label = cfg['label']
    base = tasks_new if cfg['world'] == 'new' else tasks_old
    tasks = make_regs_tasks(base)          # 全部 regs-only (R 基线)
    train_rules = [n for n in tasks if tasks[n]['seen']]
    train_names = train_rules[:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']][:8]
    stream = train_names + unseen_names
    for n in tasks:
        tasks[n]['name'] = n
    # 结构标签 (bank 分组检索 / 报告)
    T_STRUCT = build_t_struct(42)
    for n in tasks:
        tasks[n]['struct'] = (U_STRUCT.get(n) if not tasks[n]['seen']
                              else T_STRUCT.get(n, '?'))

    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(V, 4, s1_state)
    is_base = (cfg['define_refine'] == 0 and not cfg['route_recur']
               and not cfg['route_action'] and not cfg['bank'])
    if is_base:
        # B0/W0: V32_1_Learner 原样 (严格复现 32.1)
        learner = V32_1_Learner(rln, pln, V, K=K, seed=seed, sleep_iters=2,
                                n_resp=4, use_intent=True,
                                use_rule_enc=False, use_perm_aux=False,
                                use_contrast=False)
    else:
        ron = cfg.get('rln_online', 'off')
        if ron != 'off':
            learner = V33_3_Learner(
                rln, pln, V, K=K, seed=seed, sleep_iters=2,
                n_resp=4, use_intent=True,
                n_define_refine=cfg['define_refine'],
                use_route_recur=cfg['route_recur'],
                use_route_action=cfg['route_action'],
                use_pattern_bank=cfg['bank'],
                bank_tau=cfg.get('bank_tau', 0.5),
                bank_struct=cfg.get('bank_struct', False),
                refine_hidden=cfg.get('refine_hidden'),
                refine_layers=cfg.get('refine_layers', 2),
                rln_online=ron, rln_lr=cfg.get('rln_lr', 1e-4),
                rln_every=cfg.get('rln_every', 10),
                rln_fast_lr=cfg.get('rln_fast_lr', 5e-4),
                rln_fast_every=cfg.get('rln_fast_every', 5),
                rln_slow_lr=cfg.get('rln_slow_lr', 1e-4),
                rln_slow_every=cfg.get('rln_slow_every', 20))
        else:
            learner = V33_Learner(
                rln, pln, V, K=K, seed=seed, sleep_iters=2,
                n_resp=4, use_intent=True,
                n_define_refine=cfg['define_refine'],
                use_route_recur=cfg['route_recur'],
                use_route_action=cfg['route_action'],
                use_pattern_bank=cfg['bank'],
                bank_tau=cfg.get('bank_tau', 0.5),
                bank_struct=cfg.get('bank_struct', False),
                refine_hidden=cfg.get('refine_hidden'),
                refine_layers=cfg.get('refine_layers', 2))
    learner._task_struct = {n: tasks[n]['struct'] for n in tasks}
    t0 = time.time()
    n_iters = cfg.get('meta_iters', n_meta_iters)
    for it in range(n_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_rules)
        if verbose and (it + 1) % 20 == 0:
            print(f"    [{label}] meta_iter {it+1}/{n_iters} "
                  f"loss={loss:.4f}", flush=True)
    trained_head = learner.head.state()
    dt = time.time() - t0
    print(f"    [{label}] meta 训练完成 ({dt:.0f}s)", flush=True)

    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    t_ls = time.time() - t0
    print(f"    [{label}] 长流完成 ({t_ls:.0f}s)", flush=True)

    curves = {}
    rd0 = {}
    for n in unseen_names:
        curves[n] = learner.eval_adaptation_curve(
            tasks[n], max_steps=40, eval_every=10, head_state=trained_head)
        f, r = learner.eval_goal_readout(tasks[n], head_state=trained_head)
        rd0[n] = r

    by_struct = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen_names:
        s = U_STRUCT[n]
        if s in by_struct:
            by_struct[s].append(st['acc_after'][n])
    struct_mean = {k: (sum(v) / len(v) if v else None)
                   for k, v in by_struct.items()}

    aux = {}
    if hasattr(learner, 'probe_head'):
        aux['probe_train'] = learner.probe_perm_acc(tasks, train_names)
        aux['probe_unseen'] = learner.probe_perm_acc(tasks, unseen_names)
        if learner.probe_trace:
            aux['probe_loss_last10'] = round(
                sum(learner.probe_trace[-10:]) / 10, 4)
    if cfg['bank'] and learner.bank is not None:
        aux['bank'] = learner.bank.stats()

    head_stats = learner.head.step_size_stats()
    result = {
        'cell': cell, 'label': label, 'seed': seed,
        'time_s': round(time.time() - t0, 1),
        'meta_iters': cfg.get('meta_iters', n_meta_iters), 'K': K,
        'world': cfg['world'],
        'factors': {'DefineRefine': cfg['define_refine'] > 0,
                    'RouteRecur': cfg['route_recur'],
                    'RouteAction': cfg['route_action'],
                    'PatternBank': cfg['bank'],
                    'WorldFix': cfg['world'] == 'new'},
        'stream': stream,
        'train_acc_after': {n: st['acc_after'][n] for n in train_names},
        'unseen_acc_after': {n: st['acc_after'][n] for n in unseen_names},
        'unseen_acc_after_full': {n: st['acc_after_full'][n]
                                  for n in unseen_names},
        'readout0_unseen': rd0,
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
    print(f"  [{label}] 全流程完成 ({result['time_s']:.0f}s)", flush=True)
    print(f"     未见 regs_acc: {ua}  mean={u_mean:.4f}", flush=True)
    print(f"     未见 rd0: mean={r_mean:.4f}", flush=True)
    print(f"     结构分组: {struct_mean}", flush=True)
    if aux:
        print(f"     aux: {aux}", flush=True)
    return result


# ============================================================
# 报告
# ============================================================
def _mean(d):
    return sum(d.values()) / max(len(d), 1)


def summarize(results):
    rows = {}
    for r in results:
        ua = r['unseen_acc_after']
        rd = r['readout0_unseen']
        by = r['unseen_by_struct']
        rows[r['cell']] = {
            'label': r['label'], 'time_s': r['time_s'],
            'factors': r['factors'],
            'world': r['world'],
            'unseen_mean': round(_mean(ua), 4),
            'rd0_mean': round(_mean(rd), 4),
            'unseen_minus_rd0': round(_mean(ua) - _mean(rd), 4),
            'c3': by.get('c3'), 'c4': by.get('c4'), 'dbl': by.get('dbl'),
            'aux': r['aux'],
        }
    return rows


def hypothesis_verdicts(rows):
    b0 = rows.get('B0', {}).get('unseen_mean')
    v = {}

    def _t(a, b, hi, desc):
        va = rows.get(a, {}).get('unseen_mean')
        vb = rows.get(b, {}).get('unseen_mean')
        if va is None or vb is None:
            return (desc, 'N/A', None, hi)
        d = va - vb
        return (desc, '成功' if d > hi else
                ('部分成功' if d >= hi * 0.4 else '失败'), round(d, 4), hi)

    if b0 is not None:
        w0 = rows.get('W0', {}).get('unseen_mean')
        v['HW'] = _t('W0', 'B0', 0.02, '世界修复单独效果 vs B0')
        v['HA'] = _t('A', 'B0', 0.02, 'A define-refine vs B0')
        v['HB'] = _t('B', 'B0', 0.02, 'B 路由进递推 vs B0')
        v['HC'] = _t('C', 'B0', 0.02, 'C 动作路由 vs B0')
        v['HD'] = _t('D', 'B0', 0.02, 'D 模式库 vs B0')
        v['HE'] = _t('E', 'B0', 0.02, 'E 全闭环 vs B0')
        v['HEW'] = _t('EW', 'B0', 0.02, 'EW 全闭环+新世界 vs B0')
        # V33.1 (OaK 放大成功, 基线 = W0)
        v['HA1'] = _t('WA', 'W0', 0.02,
                      'WA 世界修复×define-refine vs W0')
        v['HA2'] = _t('WAC', 'WA', 0.02, 'WAC +动作路由 vs WA')
        v['HA3'] = _t('WADb', 'WA', 0.02, 'WADb +结构化模式库 vs WA')
        wa = rows.get('WA', {}).get('unseen_mean')
        if wa is not None:
            v['HA0'] = ('WA > 0.10 (突破 W0 后继续)', 
                        '成功' if wa > 0.10 else '未突破', round(wa, 4), 0.10)
        # V33.2 (redefine MLP 扩展, 基线 = WA)
        v['HB1'] = _t('WAw', 'WA', 0.02, 'WAw refine 加宽(512) vs WA')
        v['HB2'] = _t('WAd', 'WA', 0.02, 'WAd refine 加深(3层) vs WA')
        v['HB3'] = _t('WAwd', 'WA', 0.02, 'WAwd 宽+深 vs WA')
        # V33.3 (DRLN 魔改, 基线 = WA)
        v['HC1'] = _t('WAf', 'WA', 0.02, 'WAf RLN 不冻结 vs WA')
        v['HC2'] = _t('WAmf', 'WA', 0.02, 'WAmf RLN 多频更新 vs WA')
        # 可加性: E vs max(A,B,C,D)
        cands = [rows.get(k, {}).get('unseen_mean')
                 for k in ('A', 'B', 'C', 'D')]
        best = max([x for x in cands if x is not None], default=None)
        e = rows.get('E', {}).get('unseen_mean')
        if best is not None and e is not None:
            v['HADD'] = ('E ≥ max(A,B,C,D)',
                         '成功' if e > best + 0.01 else
                         ('持平' if abs(e - best) <= 0.01 else '失败'),
                         round(e - best, 4), 0.01)
        else:
            v['HADD'] = ('E ≥ max(A,B,C,D)', 'N/A', None, 0.01)
    return v


def write_report(results, args):
    rp = ROOT / "results" / "v33_drl_report.json"
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                  encoding='utf-8')
    rows = summarize(results)
    hv = hypothesis_verdicts(rows)

    from run_v31_2_long_stream import build_v32_tasks as _b
    _to = tome_fraction(_b(200, seed=42, window=3))
    _tn = tome_fraction(_b(200, seed=42, window=3, stop_at_halt=True))

    L = []
    A = L.append
    A("# V33 Define-Route 循环报告 — OaK 式持续学习 × INTACT 意图路由")
    A("")
    A("> 计划: `docs/wiki/v33-plan.md` (预注册) · 运行时间: "
      f"{time.strftime('%Y-%m-%d %H:%M')}")
    A(f"> 参数: meta_iters={args.meta_iters} K={args.K} "
      f"steps_per_task={args.steps_per_task} seed=42 · 同 Stage1 缓存")
    A("")
    A("## 0. 大白话总结")
    A("")
    A("- **做了什么**: 按 OaK 的 Define 层视角重建循环 — 意图定义可迭代 (define-refine)、"
      "定义路由进递推 (调制 SSM dt/B)、定义路由到动作 (top-k 重加权, INTACT 选择语义)、"
      "跨任务模式库叠加; 8 cell 消融, 含世界修复 (TOME 截断) 因子。")
    A("- **发现什么**: (结果落地后填)")
    A("- **为什么重要**: V32.1 证明外部机制 (aux/对比/EHS) 全失败 → 本版把机制"
      "从\"加监督\"改为\"立循环\": 定义结果回流到表示与动作。若全闭环成功, "
      "论文主线 EHS 循环化 (Define=假设提取, Route=假设路由) 就有了实验 1 数据。")
    A("")
    A("## 1. 立论")
    A("")
    A("- h 的问题 = 循环和路由没确立 (用户诊断): Define 层缺失 (经验→抽象结构需可迭代), "
      "Route 层缺失 (规则与内容挤在同一向量)")
    A("- INTACT 重定位: 意图插入 = define, 选择动作 = route; V32 失败根因 = "
      "只做 define 输出注入 (head 侧, 被 W 适应洗掉), 没做路由进表示 (递推侧)")
    A("- 三圈循环: 循环1 表示级 (define 调制递推) · 循环2 定义级 (define-refine) · "
      "循环3 能力级 (模式库)")
    A("")
    A("## 2. 细胞设计")
    A("")
    A("| cell | 环节 | 世界 |")
    A("|:--|:--|:--|")
    for cell, cfg in CELLS.items():
        fac = '+'.join(f for f, on in {
            'DefRef': cfg['define_refine'] > 0, 'RteRecur': cfg['route_recur'],
            'RteAct': cfg['route_action'], 'Bank': cfg['bank'],
        }.items() if on)
        A(f"| {cell} | {fac or '—'} | {'新(TOME截断)' if cfg['world']=='new' else '旧'} |")
    A("")
    A("## 3. 结果总表 (unseen regs_acc)")
    A("")
    A("| cell | 因子 | Unseen | rd0 | 60s−rd0 | c3 | c4 | dbl | time_s |")
    A("|:--|:--|--:|--:|--:|--:|--:|--:|--:|")
    for cell in CELLS:
        if cell not in rows:
            continue
        s = rows[cell]
        fac = '+'.join(f for f, on in s['factors'].items() if on)
        cc = lambda x: f"{x:.4f}" if x is not None else "-"
        A(f"| {cell} | {fac or '—'} | {s['unseen_mean']:.4f} | "
          f"{s['rd0_mean']:.4f} | {s['unseen_minus_rd0']:+.4f} | "
          f"{cc(s['c3'])} | {cc(s['c4'])} | {cc(s['dbl'])} | {s['time_s']:.0f} |")
    A("")
    A("### 假设判定")
    A("")
    A("| 假设 | 判定 | 差值 | 阈值 |")
    A("|:--|:--|--:|--:|")
    for k, (desc, verdict, d, hi) in hv.items():
        ds = f"{d:+.4f}" if d is not None else "-"
        A(f"| {k} | **{verdict}** | {ds} | {hi} |")
    A("")
    A(f"## 4. 世界修复 (TOME): 旧 {_to:.4f} → 新 {_tn:.4f} (query 集 tgt[4]==201 占比)")
    A("")
    A("## 5. 辅助指标")
    A("")
    for cell in CELLS:
        r = {x['cell']: x for x in results}.get(cell)
        if not r or not r['aux']:
            continue
        A(f"- **{cell}**: {json.dumps(r['aux'], ensure_ascii=False)}")
    A("")
    A("## 6. 假阳性检查")
    A("")
    A("- **活跃基线**: B0 ≡ V32.1 32.1 (regs-only, 旧世界, 同缓存同协议) — "
      "应复现 unseen regs_acc 0.1016 (`results/v32_1_matrix_report.json`)")
    A("- **评估独立性**: regs_acc 4-token 精确匹配; rd0 纯共享 W (零适应)")
    A("- **单一循环变量**: A/B/C/D 各自只开一个环节, E 才全开; "
      "W0/EW 用新世界 (数据量减少 ~65% 是世界因子的固有属性)")
    A("- **路由参数零初始化**: cond_dt/cond_b 初始 ≡ 无条件 → B 与 B0 在训练前等价")
    A("")
    A("## 7. 结论")
    A("")
    A("(结果落地后填)")
    A("")
    A("## 8. 复现")
    A("")
    A("```bash")
    A(f"python tests/run_v33_drl.py --meta-iters {args.meta_iters} "
      f"--K {args.K} --steps-per-task {args.steps_per_task}")
    A("```")
    mp = ROOT / "results" / "v33_drl_report.md"
    mp.write_text("\n".join(L) + "\n", encoding='utf-8')
    print(f"\nJSON: {rp}\nMD: {mp}")
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
                        default='B0,W0,WA,WAC,WADb')
    parser.add_argument('--no-stage1', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("V33 Define-Route 循环 — OaK × INTACT")
    print("=" * 70)

    WORLD_V = 200
    tasks_old = build_v32_tasks(WORLD_V, n_support=40, n_query=20,
                                seed=42, window=3)
    tasks_new = build_v32_tasks(WORLD_V, n_support=40, n_query=20,
                                seed=42, window=3, stop_at_halt=True)
    train_rules_all = [n for n in tasks_old if tasks_old[n]['seen']]
    print(f"V={WORLD_V}, 训练规则={len(train_rules_all)}; "
          f"旧世界 TOME={tome_fraction(tasks_old):.4f} / "
          f"新世界 TOME={tome_fraction(tasks_new):.4f}")

    U_STRUCT = build_u_struct(42)

    s1_state = None
    if args.no_stage1:
        print("\n--no-stage1: 随机初始化 (冒烟用)")
    else:
        S1_CACHE = None
        for cand in ("/tmp/v32_s1_state.pt", "/tmp/v31_2_s1_state.pt"):
            if os.path.exists(cand):
                S1_CACHE = cand
                break
        if S1_CACHE is None:
            print("\n>>> Stage1 缓存缺失, 请先跑 tests/run_v32_stage1.py")
            return
        s1_state = torch.load(S1_CACHE)
        s1_state = {k: v.cpu().clone() for k, v in s1_state.items()}
        print(f"\nStage1: 从缓存加载 ({S1_CACHE})")

    keys = [c.strip() for c in args.cells.split(',')]
    results = []
    ckpt = ROOT / "results" / "v33_checkpoint.json"
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
        print(f"\n{'='*60}\n  实验: {cfg['label']} (世界={cfg['world']}, "
              f"refine={cfg['define_refine']}, RteR={cfg['route_recur']}, "
              f"RteA={cfg['route_action']}, Bank={cfg['bank']})\n{'='*60}")
        try:
            r = run_cell(key, cfg, WORLD_V, s1_state, tasks_old, tasks_new,
                         U_STRUCT, n_meta_iters=args.meta_iters, K=args.K,
                         seed=42, steps_per_task=args.steps_per_task)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! cell {key} 失败: {e} (继续)")
            continue
        results.append(r)
        ckpt.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
        print(f"checkpoint 更新: {key} ({len(results)}/{len(keys)})",
              flush=True)

    if results:
        rows, hv = write_report(results, args)
        print(f"\n{'='*60}\n汇总:\n{'='*60}")
        print(f"{'cell':<6}{'Unseen':<10}{'rd0':<8}{'60s-rd0':<10}{'time_s':<8}")
        for cell in CELLS:
            if cell not in rows:
                continue
            s = rows[cell]
            print(f"{cell:<6}{s['unseen_mean']:<10.4f}{s['rd0_mean']:<8.4f}"
                  f"{s['unseen_minus_rd0']:<10.4f}{s['time_s']:<8.0f}")
        print("\n假设判定:")
        for k, (desc, verdict, d, hi) in hv.items():
            print(f"  {k}: {verdict} ({d} vs 阈值 {hi})")
    return results


if __name__ == "__main__":
    main()
