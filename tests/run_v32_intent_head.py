"""
V32: INTACT-JEPA 意图注入 head 如何打破 V31.2 RLN 平坦适应瓶颈
====================================================================
立论 (docs/wiki/v31-2-vs-jepa.md + docs/wiki/v32-plan.md):
  V31.2 结论: 各种 head 适应 (~0.4-0.5 未见规则 acc 封顶), 流内 60 步
  适应曲线平坦 → 瓶颈在 RLN 表示 / head readout 接口。
  INTACT (arXiv 2607.26056v1) 主张: forward-only 学习器的瓶颈是
  "uncalibrated funnel" (只学预测, 几何附带) → 解法是用**意图条件动作律**:
  同一个共享算子解释两个条件族 (non-call + goal call), 非对称梯度
  (物理后继 attached / 未来目标 stop-grad anchor), 且意图在 adaptation
  之前就可用 → 表示在适应前就变好 ("search-free direct readout").

映射到 LMT 代码世界 (OML 元学习):
  - "动作" = 预测 8-token 观察 (规则确定下的确定性结果)
  - non-call (local, attached) = 支持样本自身的 target → m_local (训练监督)
  - goal call (可部署, stop-grad) = 支持集聚合得到的规则意图 → m_goal (anchor)
  - 共享算子 = PLN head, 注入语法 [h; m; h⊙m] (头输入 3d)

2×2 factorial:
  A = 意图注入 (intent): off (V31.2 原样 [h]) / on ([h;m;h⊙m] + 双条件)
  B = JEPA EMA 潜目标辅助: off / on (预测 EMA_RLN(tgt).last)
  C1 base / C2 +intent / C3 +ema-aux / C4 +intent+ema-aux

控制变量 (与 V31.2 完全一致): 世界 · 数据 · Stage1 缓存 · meta 80 iters
  K=20 · Adam 1e-3 · SwiftTD β_init=1e-3 κ=0.1 η=0.1 ϵ=0.99 · 长流 16×60 步
  sleep_iters=2 · seed=42 · n_support=40 n_query=20

主指标: 未见规则 60 步段后 acc; 段内 0/20/40/60 适应曲线;
  新增 INTACT 零步 readout acc (goal call, 不适应) —— 表示是否在适应前就变好。
"""

import os, sys, time, json, math, random, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hibs_lnn.swifttd_head import SwiftTDHead
from run_v31_meta_learning import (
    batch_pairs, head_forward, build_model, stage1_pretrain_world,
)
from run_v31_2_long_stream import V31_2_Learner, build_v32_tasks


class V32_Learner(V31_2_Learner):
    """V31.2 + 意图条件 head (INTACT 双条件共享算子) + I-JEPA EMA 潜目标。"""

    def __init__(self, rln, pln, V, K=20, outer_lr=1e-3,
                 n_tasks_per_step=2, seed=42,
                 beta_init=1e-3, kappa=0.1, eta=0.1, eps=0.99,
                 init_lr=0.0, sleep_iters=2,
                 use_intent=True, use_ema=False,
                 lambda_local=1.0, lambda_aux=0.1, ema_decay=0.99,
                 n_local_sup=8):
        d = rln.d_model
        self.use_intent = use_intent
        self.use_ema = use_ema
        self.lambda_local = lambda_local
        self.lambda_aux = lambda_aux
        self.ema_decay = ema_decay
        self.n_local_sup = n_local_sup

        # head 输入语法: [h; m; h⊙m] → w1=(d,3d); C1/C3 保持 (d,d)
        if use_intent:
            old = pln.predictor[0]
            new = nn.Linear(d * 3, d)
            with torch.no_grad():
                new.weight[:, :d].copy_(old.weight)
                new.bias.copy_(old.bias)
            pln.predictor[0] = new
            pln.step_beta = nn.Parameter(
                torch.full((pln._n_params(),), math.log(beta_init)))

        super().__init__(rln, pln, V, K=K, outer_lr=outer_lr,
                         n_tasks_per_step=n_tasks_per_step, seed=seed,
                         beta_init=beta_init, kappa=kappa, eta=eta, eps=eps,
                         init_lr=init_lr, sleep_iters=sleep_iters)

        self.intent_dim = d * 3 if use_intent else d
        # INTACT 意图编码器: 局部/目标共享 (meta 参数, 外循环训练)
        self.m_norm = nn.LayerNorm(d)
        self.intent_net = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))
        # JEPA EMA 潜目标: EMA 拷贝 + 辅助预测器
        if use_ema:
            self.ema_rln = copy.deepcopy(rln)
            self.ema_rln.requires_grad_(False)
            self.aux_pred = nn.Linear(d, d)
        else:
            self.ema_rln = None
            self.aux_pred = None
        if use_intent:
            # 头换成输入 3d 的 SwiftTDHead (保持 β_init 等一致)
            self.head = SwiftTDHead(self.d_model, self.world_vocab,
                                    self.n_resp, beta_init=beta_init,
                                    kappa=kappa, eta=eta, eps=eps,
                                    input_dim=self.intent_dim)
            self.head.reset_state()
        extra = list(self.intent_net.parameters())
        if self.aux_pred is not None:
            extra += list(self.aux_pred.parameters())
        self.outer_opt = torch.optim.Adam(
            list(rln.parameters()) + list(pln.parameters()) + extra,
            lr=outer_lr)

    # ── 意图 ──────────────────────────────────────────────────
    def _embed_mean(self, tgt):
        e = self.rln.embed(tgt)             # (B, n_resp, d)
        return e.mean(dim=(0, 1))

    def compute_intent(self, tgts):
        """tgts: (B, n_resp) → m (d,) 经 intent_net + LN (带图)。"""
        emb = self.rln.embed(tgts).mean(dim=(0, 1))
        return self.m_norm(self.intent_net(emb))

    def compute_goal_intent(self, task):
        """goal call 聚合 (INTACT: 未来目标 anchor, stop-grad; 部署接口)。"""
        sup = task['support']
        B = min(len(sup), 16)
        with torch.no_grad():
            tgts = torch.stack([s[1] for s in sup[:B]])
            m = self.compute_intent(tgts)
        return m.detach()

    def _h_in_batch(self, h, m):
        if not self.use_intent or m is None:
            return h
        mb = m.unsqueeze(0).expand(h.shape[0], -1)
        return torch.cat([h, mb, h * mb], dim=-1)   # (B, 3d)

    # ── 可微内循环 ────────────────────────────────────────────
    def adapt_graph(self, support, K=None, m=None):
        K = K or self.K
        W = self.pln.clone_params()
        for (ctx, tgt) in support[:K]:
            hidden = self.rln(ctx.unsqueeze(0))
            h = hidden[:, -1, :]
            hin = self._h_in_batch(h, m)
            x = torch.tanh(hin @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(1, self.n_resp,
                                                self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            W = self.head.step_graph(W, hin[0], x[0], errs)
        return W

    # ── 无图在线适应 ──────────────────────────────────────────
    def adapt_online(self, W, support, n_steps, m=None):
        for (ctx, tgt) in support[:n_steps]:
            with torch.no_grad():
                hidden = self.rln(ctx.unsqueeze(0))
            h = hidden[:, -1, :]
            hin = self._h_in_batch(h, m)
            x = torch.tanh(hin @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(1, self.n_resp,
                                                self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            self.head.step(W, hin[0], x[0], errs)
        return W

    # ── 外循环 (BPTT) ────────────────────────────────────────
    def meta_step(self, tasks, train_rules, K=None):
        self.outer_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        Ws = []
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        n_loc = n_aux = 0
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

            # 双条件共享算子 (INTACT): local call, attached —
            # per-sample target → 塑造 intent_net + RLN (与 goal 同一算子)
            if self.use_intent and self.lambda_local > 0:
                for (c, tg) in t['support'][:self.n_local_sup]:
                    hsv = self.rln(c.unsqueeze(0))[:, -1, :]
                    m_loc = self.compute_intent(tg.unsqueeze(0))
                    hin = self._h_in_batch(hsv, m_loc)
                    pl = head_forward(W, hin, self.intent_dim,
                                      self.world_vocab, self.n_resp)
                    total_loss = total_loss + self.lambda_local * \
                        F.cross_entropy(pl.view(-1, self.world_vocab),
                                        tg.view(-1).clamp(0, self.V + 1)) \
                        / (len(chosen) * self.n_local_sup)
                    n_loc += 1

            # JEPA EMA 潜目标辅助 (因子 B)
            if self.use_ema and self.aux_pred is not None \
                    and self.ema_rln is not None:
                with torch.no_grad():
                    h_tgt = self.ema_rln(tgt)[:, -1, :]
                aux_loss = F.mse_loss(self.aux_pred(h), h_tgt)
                total_loss = total_loss + self.lambda_aux * aux_loss \
                    / len(chosen)
                n_aux += 1

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
        if self.use_ema and self.ema_rln is not None:
            with torch.no_grad():
                for p, ep in zip(self.rln.parameters(),
                                 self.ema_rln.parameters()):
                    ep.mul_(self.ema_decay).add_(p.detach() *
                                                 (1 - self.ema_decay))
        self._meta_iters += 1
        return total_loss.item()

    # ── 评估 ──────────────────────────────────────────────────
    def _query_acc(self, W, task, max_batch=16, m=None):
        ctx, tgt = batch_pairs(task['query'], max_batch=max_batch)
        with torch.no_grad():
            hidden = self.rln(ctx)
        h = hidden[:, -1, :]
        h_in = self._h_in_batch(h, m)
        pred = head_forward([w.detach() for w in W], h_in,
                            self.intent_dim, self.world_vocab, self.n_resp)
        pt = pred.argmax(dim=-1)
        return (pt == tgt).sum().item() / max(tgt.numel(), 1)

    def _load_head(self, head_state):
        if head_state is not None:
            self.head.load_state(head_state)
        else:
            self.head.reset_state()

    def eval_goal_readout(self, task, head_state=None):
        """INTACT 零步 readout: goal call, 不做任何 head 适应。
        表示是否在 adaptation 之前就可用 (search-free direct)。"""
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self.compute_goal_intent(task) if self.use_intent else None
        return self._query_acc(W, task, m=m)

    def evaluate_task(self, task, K=10, head_state=None):
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self.compute_goal_intent(task) if self.use_intent else None
        self.adapt_online(W, task['support'], K, m=m)
        return self._query_acc(W, task, m=m)

    def eval_adaptation_curve(self, task, max_steps=40, eval_every=10,
                              head_state=None):
        W = [w.detach() for w in self.pln.clone_params()]
        self._load_head(head_state)
        m = self.compute_goal_intent(task) if self.use_intent else None
        curve = [self._query_acc(W, task, m=m)]
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
                curve.append(self._query_acc(W, task, m=m))
        return curve

    def long_stream(self, tasks, stream, steps_per_task=60,
                    sleep_every=1, sleep_iters=2, head_state=None,
                    eval_every=20):
        self._load_head(head_state)
        W = [w.detach() for w in self.pln.clone_params()]
        seen = []
        forget_rows = []
        acc_after = {}
        curves = {}
        for tname in stream:
            t = tasks[tname]
            m = self.compute_goal_intent(t) if self.use_intent else None
            seg = [self._query_acc(W, t, m=m)]
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
                    seg.append(self._query_acc(W, t, m=m))
            seen.append(tname)
            curves[tname] = seg
            acc_after[tname] = seg[-1]
            row = {s: round(self._query_acc(W, tasks[s], m=m), 4)
                   for s in seen}
            forget_rows.append(row)
            if sleep_iters > 0 and len(seen) >= sleep_every:
                saved = self.head.state()
                for _ in range(sleep_iters):
                    self.meta_step(tasks, seen)
                self.head.load_state(saved)
        return {'acc_after': acc_after, 'curves': curves,
                'forget': forget_rows}


def run_cell(label, use_intent, use_ema, V, tasks, train_rules,
             s1_state, n_meta_iters=80, K=20, seed=42, verbose=True,
             steps_per_task=60):
    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model(V, use_stage1=True, s1_state=s1_state)
    learner = V32_Learner(rln, pln, V, K=K, seed=seed, sleep_iters=2,
                          use_intent=use_intent, use_ema=use_ema)
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

    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']][:8]
    stream = train_names + unseen_names
    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    dt = time.time() - t0

    curve_curves = {}
    readout0 = {}
    for n in unseen_names:
        curve_curves[n] = learner.eval_adaptation_curve(
            tasks[n], max_steps=40, eval_every=10, head_state=trained_head)
        readout0[n] = learner.eval_goal_readout(tasks[n],
                                                head_state=trained_head)
    readout0_train = {n: learner.eval_goal_readout(tasks[n],
                                                   head_state=trained_head)
                      for n in train_names}

    head_stats = learner.head.step_size_stats()
    result = {
        'label': label, 'seed': seed, 'time_s': round(dt, 1),
        'meta_iters': n_meta_iters, 'K': K,
        'use_intent': use_intent, 'use_ema': use_ema,
        'stream': stream,
        'train_acc_after': {n: st['acc_after'][n] for n in train_names},
        'unseen_acc_after': {n: st['acc_after'][n] for n in unseen_names},
        'readout0_unseen': readout0,
        'readout0_train': readout0_train,
        'stream_curves': st['curves'],
        'unseen_curve_offstream': curve_curves,
        'forget': st['forget'],
        'head_stats': head_stats,
    }
    print(f"  [{label}] 长流 {len(stream)} 任务完成 ({dt:.0f}s)", flush=True)
    print(f"     训练规则 acc: {result['train_acc_after']}", flush=True)
    print(f"     未见规则 acc: {result['unseen_acc_after']}", flush=True)
    print(f"     未见零步 readout: {readout0}", flush=True)
    return result


def print_report(results, args):
    print(f"\n{'='*120}")
    print("V32 INTACT-JEPA 意图注入 — 2×2 factorial 报告")
    print(f"{'='*120}")
    summary_rows = []
    for r in results:
        ta = r['train_acc_after']
        ua = r['unseen_acc_after']
        ro = r['readout0_unseen']
        tmean = sum(ta.values()) / len(ta)
        umean = sum(ua.values()) / len(ua)
        rmean = sum(ro.values()) / len(ro)
        hs = r['head_stats']
        summary_rows.append({
            'label': r['label'], 'intent': r['use_intent'],
            'ema': r['use_ema'], 'time_s': r['time_s'],
            'train_acc': round(tmean, 4), 'unseen_acc': round(umean, 4),
            'readout0': round(rmean, 4),
            'beta_mean': round(hs['beta_mean'], 5),
            'unseen_60step_minus_readout0': round(umean - rmean, 4),
        })
        print(f"\n[{r['label']}] ({r['time_s']}s) "
              f"intent={r['use_intent']} ema={r['use_ema']}")
        print(f"  训练规则 (流内 60 步): 均值={tmean:.4f}  {ta}")
        print(f"  未见规则 (流内 60 步): 均值={umean:.4f}  {ua}")
        print(f"  未见零步 readout (goal): 均值={rmean:.4f}  {ro}")
        print(f"  未见 off-stream 适应曲线 (0/10/20/30/40):")
        for n in r['unseen_curve_offstream']:
            print(f"    {n}: {r['unseen_curve_offstream'][n]}")
        hs = r['head_stats']
        print(f"  β: mean={hs['beta_mean']:.5f} std={hs['beta_std']:.5f} "
              f"max={hs['beta_max']:.4f} bound={hs['n_bound_triggers']}")
        if r['forget']:
            print(f"  遗忘 (16 任务后已见 acc): {r['forget'][-1]}")

    print(f"\n{'='*60}")
    print(f"{'cell':<6}{'intent':<8}{'ema':<6}{'Train':<8}{'Unseen':<10}"
          f"{'rd0':<8}{'60s-rd0':<10}{'time_s':<8}")
    for s in summary_rows:
        print(f"{s['label']:<6}{str(s['intent']):<8}{str(s['ema']):<6}"
              f"{s['train_acc']:<8.4f}{s['unseen_acc']:<10.4f}"
              f"{s['readout0']:<8.4f}{s['unseen_60step_minus_readout0']:<10.4f}"
              f"{s['time_s']:<8.0f}")

    rp = ROOT / "results" / "v32_intent_report.json"
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                  encoding='utf-8')
    md = ROOT / "results" / "v32_intent_report.md"
    lines = [
        "# V32 INTACT-JEPA 意图注报报告 (2×2)",
        "",
        f"- 时间: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- meta_iters={args.meta_iters} K={args.K} sleep_iters=2 "
        f"steps_per_task={args.steps_per_task}",
        "",
        "| cell | intent | ema | Train(60步) | Unseen(60步) | rd0(0步) | 60s-rd0 | time_s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summary_rows:
        lines.append(
            f"| {s['label']} | {s['intent']} | {s['ema']} | "
            f"{s['train_acc']:.4f} | {s['unseen_acc']:.4f} | "
            f"{s['readout0']:.4f} | {s['unseen_60step_minus_readout0']:.4f} | "
            f"{s['time_s']:.0f} |")
    md.write_text("\n".join(lines), encoding='utf-8')
    print(f"\nJSON 报告: {rp}\nMD 报告: {md}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-iters', type=int, default=80)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--steps-per-task', type=int, default=60)
    parser.add_argument('--cells', type=str,
                        default='C1,C2,C3,C4')
    parser.add_argument('--result', type=str, default='v32_intent_report.json')
    args = parser.parse_args()

    print("=" * 70)
    print("V32 INTACT-JEPA 意图注入 head — 2×2 factorial")
    print("=" * 70)

    WORLD_V = 200
    tasks = build_v32_tasks(WORLD_V, n_support=40, n_query=20, seed=42)
    train_rules = [n for n in tasks if tasks[n]['seen']]
    unseen = [n for n in tasks if not tasks[n]['seen']]
    print(f"V={WORLD_V}, 训练规则={len(train_rules)}, 未见规则={len(unseen)}")

    # Stage1 缓存: 兼容 v32/v31_2 两个路径
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
        ppl, s1_model = stage1_pretrain_world(tasks, train_rules, WORLD_V,
                                              n_epochs=2)
        s1_state = {k: v.cpu().clone()
                    for k, v in s1_model.state_dict().items()}
        torch.save(s1_state, S1_CACHE)
        print(f"    Stage1 PPL = {ppl:.2f}, 已缓存到 {S1_CACHE}")
    else:
        s1_state = torch.load(S1_CACHE)
        s1_state = {k: v.cpu().clone() for k, v in s1_state.items()}
        print(f"\nStage1: 从缓存加载 ({S1_CACHE})")

    cell_map = {
        'C1': ('C1-base', False, False),
        'C2': ('C2-intent', True, False),
        'C3': ('C3-ema', False, True),
        'C4': ('C4-both', True, True),
    }
    keys = [c.strip().upper() for c in args.cells.split(',')]
    results = []
    for key in keys:
        if key not in cell_map:
            print(f"未知 cell: {key}")
            continue
        label, use_intent, use_ema = cell_map[key]
        print(f"\n{'='*60}")
        print(f"  实验: {label} (intent={use_intent}, ema={use_ema})")
        print(f"{'='*60}")
        r = run_cell(label, use_intent, use_ema, WORLD_V, tasks, train_rules,
                     s1_state, n_meta_iters=args.meta_iters, K=args.K,
                     seed=42, steps_per_task=args.steps_per_task)
        results.append(r)

    print_report(results, args)
    return results


if __name__ == "__main__":
    main()