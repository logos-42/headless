"""
V31.2: SwiftTD 内循环 + BPTT 外循环 (不隔离) + 长流持续学习
============================================================
研究问题 (用户设定):
  1. 长流测试: 8-16 任务 × 长步数, θ/h 迹与 W 跨任务持续
  2. 是否可以学会没有学过的学习方法: 流中混入**未见规则**
     (结构不同的排列: 四轮换/双对换 vs 训练的对换/三轮换)
  3. 持续学习能力调整: sleep (BPTT meta_step 于已见任务) 在线演进 RLN
  4. 不隔离外循环: 内循环可微 (SwiftTD 速率 detached, 直通 BPTT),
     外循环用完整 BPTT (V31.0 OML 风格, Adam 更新 RLN+PLN init)

世界: S4 对称群 24 条排列规则
  训练: identity + 6 对换 + 5 三轮换 = 12
  未见: 3 三轮换 + 6 四轮换 + 3 双对换 = 12 (结构不同)

注意: V31.2 修复了 V31.0 世界的潜伏 bug (立即数被 remap mod 4),
  V31.0 的规则语义是被污染的 (其自测 remap 项一直失败, 计数器显示 7/7),
  V31.0 数字与 V31.2 不可直接对比。

实验:
  meta 训练: BPTT K=20 于 12 条训练规则 (80 iters)
  长流: [8 训练规则] + [8 未见规则], 每任务 60 步在线适应, W/θ/h 持续
  S1-no-sleep / S2-sleep (每任务后 2 次 BPTT meta_step 于已见任务)
  指标: 遗忘矩阵 (已见任务 acc), 未见规则段内适应 (每 20 步 checkpoint),
        零样本 vs 段后 acc, β 统计
"""

import os, sys, time, json, math, random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hibs_lnn.meta_rule_world import (
    build_rule_splits, gen_task_data_split,
)
from hibs_lnn.swifttd_head import SwiftTDHead
from run_v31_meta_learning import (
    batch_pairs, head_forward, build_model, stage1_pretrain_world,
)


def build_v32_tasks(V, n_support=40, n_query=20, seed=42, window=3,
                    stop_at_halt=False):
    """24 条排列规则的任务数据: {name: {'support': [...], 'query': [...]}}
    window (V32.1 因子 W): 观察窗口长度, 默认 3 (ctx=24), 4 → ctx=32。
    stop_at_halt (V33 世界修复): 程序结束后截断, 去 TOME 填充 target。"""
    train_perms, test_perms = build_rule_splits(seed)
    tasks = {}
    for i, perm in enumerate(train_perms):
        sup, qry = gen_task_data_split(None, V, n_support=n_support,
                                       n_query=n_query, seed=seed + i,
                                       remap=perm, window=window,
                                       stop_at_halt=stop_at_halt)
        tasks[f'T{i}'] = {'support': sup, 'query': qry, 'perm': perm,
                          'seen': True}
    for i, perm in enumerate(test_perms):
        sup, qry = gen_task_data_split(None, V, n_support=n_support,
                                       n_query=n_query,
                                       seed=seed + 100 + i, remap=perm,
                                       window=window,
                                       stop_at_halt=stop_at_halt)
        tasks[f'U{i}'] = {'support': sup, 'query': qry, 'perm': perm,
                          'seen': False}
    return tasks


class V31_2_Learner:
    """
    SwiftTD 可微内循环 (速率 detached) + 完整 BPTT 外循环 + 长流。

    内循环 (adapt_graph): W 从共享 init 克隆 (带图), 每样本一步
      head.step_graph (SwiftTD 速率 s,β detached, 权重链保持计算图)
      → 外循环 query loss 梯度穿过 K 步内循环链 (RLN 特征 + 共享 init)
    外循环 (meta_step): BPTT, Adam 更新 rln+pln; Reptile 拉 init
    长流 (long_stream): W 跨任务持续, θ/h 迹持续 (learner-level 步长),
      sleep 用 BPTT meta_step 于已见任务集 → RLN 随流演进
    """

    def __init__(self, rln, pln, V, K=20, outer_lr=1e-3,
                 n_tasks_per_step=2, seed=42,
                 beta_init=1e-3, kappa=0.1, eta=0.1, eps=0.99,
                 init_lr=0.0, sleep_iters=2):
        self.rln = rln
        self.pln = pln
        self.V = V
        self.world_vocab = V + 2
        self.n_resp = 8
        self.d_model = rln.d_model
        self.K = K
        self.outer_lr = outer_lr
        self.n_tasks_per_step = n_tasks_per_step
        self.seed = seed
        self.head = SwiftTDHead(self.d_model, self.world_vocab, self.n_resp,
                                beta_init=beta_init, kappa=kappa,
                                eta=eta, eps=eps)
        self.init_lr = init_lr
        self.sleep_iters = sleep_iters
        self.outer_opt = torch.optim.Adam(
            list(rln.parameters()) + list(pln.parameters()), lr=outer_lr)
        self.init_params = [self.pln.predictor[0].weight,
                            self.pln.predictor[0].bias,
                            self.pln.predictor[2].weight,
                            self.pln.predictor[2].bias]
        self._meta_iters = 0

    @property
    def meta_iters(self):
        return self._meta_iters

    @meta_iters.setter
    def meta_iters(self, v):
        self._meta_iters = v

    # ── 可微内循环 (BPTT 用) ───────────────────────────────
    def adapt_graph(self, support, K=None):
        """从共享 init 可微适应 K 步 (step_graph), 返回带图 W。"""
        K = K or self.K
        W = self.pln.clone_params()
        for (ctx, tgt) in support[:K]:
            hidden = self.rln(ctx.unsqueeze(0))       # 带图 (RLN 前向)
            h = hidden[:, -1, :]
            x = torch.tanh(h @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(1, self.n_resp,
                                                self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            W = self.head.step_graph(W, h[0], x[0], errs)
        return W

    # ── 无图在线适应 (长流用) ──────────────────────────────
    def adapt_online(self, W, support, n_steps):
        """在 W (可 detached) 上在线适应 n_steps 步, 无图。"""
        for (ctx, tgt) in support[:n_steps]:
            with torch.no_grad():
                hidden = self.rln(ctx.unsqueeze(0))
            h = hidden[:, -1, :]
            x = torch.tanh(h @ W[0].t() + W[1])
            logits = (x @ W[2].t() + W[3]).view(1, self.n_resp,
                                                self.world_vocab)
            p = F.softmax(logits, dim=-1)
            onehot = F.one_hot(tgt, self.world_vocab).float()
            errs = (p - onehot).reshape(-1)
            self.head.step(W, h[0], x[0], errs)
        return W

    # ── 外循环: BPTT meta 更新 ─────────────────────────────
    def meta_step(self, tasks, train_rules, K=None):
        """一个 BPTT 元更新: 采样 n_tasks_per_step 个任务, 内循环 K 步
        (带图), query loss backward → Adam 更新 rln+pln + Reptile init。"""
        self.outer_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        Ws = []
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        for rname in chosen:
            t = tasks[rname]
            W = self.adapt_graph(t['support'], K=K)
            Ws.append([w.detach() for w in W])
            ctx, tgt = batch_pairs(t['query'], max_batch=16)
            hidden = self.rln(ctx)
            h = hidden[:, -1, :]
            pred = head_forward(W, h, self.d_model, self.world_vocab,
                                self.n_resp)
            loss = F.cross_entropy(
                pred.view(-1, self.world_vocab),
                tgt.view(-1).clamp(0, self.V + 1))
            total_loss = total_loss + loss / len(chosen)
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

    # ── 评估 ───────────────────────────────────────────────
    def _query_acc(self, W, task, max_batch=16):
        ctx, tgt = batch_pairs(task['query'], max_batch=max_batch)
        with torch.no_grad():
            hidden = self.rln(ctx)
        h = hidden[:, -1, :]
        pred = head_forward([w.detach() for w in W], h, self.d_model,
                            self.world_vocab, self.n_resp)
        pt = pred.argmax(dim=-1)
        return (pt == tgt).sum().item() / max(tgt.numel(), 1)

    def evaluate_task(self, task, K=10, head_state=None):
        """从共享 init 适应 K 步后 query acc。"""
        W = [w.detach() for w in self.pln.clone_params()]
        if head_state is not None:
            self.head.load_state(head_state)
        else:
            self.head.reset_state()
        self.adapt_online(W, task['support'], K)
        return self._query_acc(W, task)

    def eval_adaptation_curve(self, task, max_steps=40, eval_every=10,
                              head_state=None):
        """适应曲线: [0步, ...] — 未见规则段内学习速度。"""
        W = [w.detach() for w in self.pln.clone_params()]
        if head_state is not None:
            self.head.load_state(head_state)
        else:
            self.head.reset_state()
        curve = [self._query_acc(W, task)]
        sup = task['support']
        for j in range(max_steps):
            if j < len(sup):
                with torch.no_grad():
                    hidden = self.rln(sup[j][0].unsqueeze(0))
                h = hidden[:, -1, :]
                x = torch.tanh(h @ W[0].t() + W[1])
                logits = (x @ W[2].t() + W[3]).view(
                    1, self.n_resp, self.world_vocab)
                p = F.softmax(logits, dim=-1)
                onehot = F.one_hot(sup[j][1], self.world_vocab).float()
                errs = (p - onehot).reshape(-1)
                self.head.step(W, h[0], x[0], errs)
            if (j + 1) % eval_every == 0:
                curve.append(self._query_acc(W, task))
        return curve

    # ── 长流 (V31.2 核心) ──────────────────────────────────
    def long_stream(self, tasks, stream, steps_per_task=60,
                    sleep_every=1, sleep_iters=2, head_state=None,
                    eval_every=20):
        """
        长流持续学习:
          W 从共享 init 开始, 跨任务持续 (不重置)
          θ/h 迹跨任务持续 (learner-level 步长, SwiftTD 核心主张)
          sleep_every 个任务后做 sleep_iters 次 BPTT meta_step (已见任务)
          → RLN 随流演进; sleep 前后保存/恢复 θ/h 迹

        返回:
          {'acc_after': {task: acc 段后},
           'curves': {task: [段内每 eval_every 步的 acc]},
           'forget': [[行=学完任务, 列=评估任务] 矩阵]}
        """
        if head_state is not None:
            self.head.load_state(head_state)
        else:
            self.head.reset_state()
        W = [w.detach() for w in self.pln.clone_params()]
        seen = []
        forget_rows = []
        acc_after = {}
        curves = {}
        for tname in stream:
            t = tasks[tname]
            seg = [self._query_acc(W, t)]
            for j in range(steps_per_task):
                if j < len(t['support']):
                    with torch.no_grad():
                        hidden = self.rln(t['support'][j][0].unsqueeze(0))
                    h = hidden[:, -1, :]
                    x = torch.tanh(h @ W[0].t() + W[1])
                    logits = (x @ W[2].t() + W[3]).view(
                        1, self.n_resp, self.world_vocab)
                    p = F.softmax(logits, dim=-1)
                    onehot = F.one_hot(t['support'][j][1],
                                       self.world_vocab).float()
                    errs = (p - onehot).reshape(-1)
                    self.head.step(W, h[0], x[0], errs)
                if (j + 1) % eval_every == 0:
                    seg.append(self._query_acc(W, t))
            seen.append(tname)
            curves[tname] = seg
            acc_after[tname] = seg[-1]
            # 遗忘行: 学完 tname 后所有已见任务 acc
            row = {s: round(self._query_acc(W, tasks[s]), 4) for s in seen}
            forget_rows.append(row)
            # sleep: 用已见任务集做 BPTT meta_step (RLN 演进)
            if sleep_iters > 0 and len(seen) >= sleep_every:
                saved = self.head.state()
                for _ in range(sleep_iters):
                    self.meta_step(tasks, seen)
                self.head.load_state(saved)
        return {'acc_after': acc_after, 'curves': curves,
                'forget': forget_rows}


def run_experiment(label, V, tasks, train_rules, n_meta_iters=80, K=20,
                   seed=42, s1_state=None, sleep_iters=2, verbose=True):
    rln, pln = build_model(V, use_stage1=True, s1_state=s1_state)
    learner = V31_2_Learner(rln, pln, V, K=K, seed=seed,
                            sleep_iters=sleep_iters)
    t0 = time.time()
    for it in range(n_meta_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_rules)
        if verbose and (it + 1) % 20 == 0:
            print(f"    [{label}] meta_iter {it+1}/{n_meta_iters} "
                  f"loss={loss:.4f}", flush=True)
    trained_head = learner.head.state()
    dt = time.time() - t0
    print(f"    [{label}] meta 训练 {n_meta_iters} iters 完成 ({dt:.0f}s)")

    # 长流: 8 训练规则 + 8 未见规则
    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']][:8]
    stream = train_names + unseen_names
    st = learner.long_stream(tasks, stream, steps_per_task=60,
                             sleep_iters=sleep_iters,
                             head_state=trained_head)
    dt = time.time() - t0

    # 未见规则的独立适应曲线 (对比: 流外从头适应)
    curve_curves = {}
    for n in unseen_names:
        curve_curves[n] = learner.eval_adaptation_curve(
            tasks[n], max_steps=40, eval_every=10, head_state=trained_head)

    head_stats = learner.head.step_size_stats()
    result = {
        'label': label, 'seed': seed, 'time_s': round(dt, 1),
        'meta_iters': n_meta_iters, 'K': K,
        'stream': stream,
        'train_acc_after': {n: st['acc_after'][n] for n in train_names},
        'unseen_acc_after': {n: st['acc_after'][n] for n in unseen_names},
        'stream_curves': st['curves'],
        'unseen_curve_offstream': curve_curves,
        'forget': st['forget'],
        'head_stats': head_stats,
    }
    print(f"  [{label}] 长流 {len(stream)} 任务完成 ({dt:.0f}s)")
    print(f"     训练规则 acc: {result['train_acc_after']}")
    print(f"     未见规则 acc: {result['unseen_acc_after']}")
    return result


def print_report(results, args):
    print(f"\n{'='*120}")
    print("V31.2 SwiftTD + BPTT 长流持续学习 — 实验报告")
    print(f"{'='*120}")
    for r in results:
        print(f"\n[{r['label']}] ({r['time_s']}s)")
        ta = r['train_acc_after']
        ua = r['unseen_acc_after']
        print(f"  训练规则 (流内, 60 步): "
              f"均值={sum(ta.values())/len(ta):.4f}  {ta}")
        print(f"  未见规则 (流内, 60 步): "
              f"均值={sum(ua.values())/len(ua):.4f}  {ua}")
        hs = r['head_stats']
        print(f"  β: mean={hs['beta_mean']:.5f} std={hs['beta_std']:.5f} "
              f"max={hs['beta_max']:.4f} "
              f"bound={hs['n_bound_triggers']}")
        print("  未见规则流内段内曲线 (0/20/40/60 步):")
        for n in r['stream_curves']:
            if not n.startswith('U'):
                continue
            print(f"    {n}: {r['stream_curves'][n]}")
        # 遗忘: 最后一行 (全部 16 任务后) vs 学完即测
        if r['forget']:
            last = r['forget'][-1]
            print(f"  遗忘 (16 任务全部学完后的已见任务 acc): {last}")

    rp = ROOT / "results" / "v31_2_long_stream_report.json"
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                  encoding='utf-8')
    print(f"\nJSON 报告: {rp}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-iters', type=int, default=80)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--sleep-iters', type=int, default=2)
    parser.add_argument('--steps-per-task', type=int, default=60)
    parser.add_argument('--modes', type=str, default='S1,S2')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("V31.2 SwiftTD 内循环 + BPTT 外循环 + 长流持续学习")
    print("=" * 70)

    WORLD_V = 200
    tasks = build_v32_tasks(WORLD_V, n_support=40, n_query=20, seed=42)
    train_rules = [n for n in tasks if tasks[n]['seen']]
    unseen = [n for n in tasks if not tasks[n]['seen']]
    print(f"V={WORLD_V}, 训练规则={len(train_rules)}, "
          f"未见规则={len(unseen)}")

    S1_CACHE = "/tmp/v31_2_s1_state.pt"
    os.makedirs(os.path.dirname(S1_CACHE), exist_ok=True)
    if os.path.exists(S1_CACHE):
        s1_state = torch.load(S1_CACHE)
        s1_state = {k: v.cpu().clone() for k, v in s1_state.items()}
        print("\nStage1: 从缓存加载")
    else:
        print("\n>>> Stage1 预训练 (12 条训练规则 token 流)...")
        random.seed(42); torch.manual_seed(42)
        ppl, s1_model = stage1_pretrain_world(tasks, train_rules, WORLD_V,
                                              n_epochs=2)
        s1_state = {k: v.cpu().clone()
                    for k, v in s1_model.state_dict().items()}
        torch.save(s1_state, S1_CACHE)
        print(f"    Stage1 PPL = {ppl:.2f}, 已缓存到 {S1_CACHE}")

    mode_map = {
        'S1': ('S1-no-sleep', 0),
        'S2': ('S2-sleep', args.sleep_iters),
    }
    keys = [m.strip().upper() for m in args.modes.split(',')]
    if args.quick:
        keys = ['S2']

    results = []
    for key in keys:
        if key not in mode_map:
            print(f"未知 mode: {key}")
            continue
        label, sleep = mode_map[key]
        print(f"\n{'='*60}\n  实验: {label} (sleep_iters={sleep})\n{'='*60}")
        r = run_experiment(label, WORLD_V, tasks, train_rules,
                           n_meta_iters=args.meta_iters, K=args.K,
                           seed=42, s1_state=s1_state,
                           sleep_iters=sleep)
        results.append(r)

    print_report(results, args)
    return results


if __name__ == "__main__":
    main()
