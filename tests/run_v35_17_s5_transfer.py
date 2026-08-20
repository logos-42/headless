"""V35.17 L2: S4→S5 群外推 — 迁移泛化真测试 (2026-08-15, 用户发起)

问题: 表示层学到的"对换复合规则"能否跨群迁移?
  S4 训练 (4 元素/6 对换/24 perm, C2 配置家族同进程最强)
  → S5 迁移 (5 元素/10 对换/120 perm)

协议 (同进程, 排除机制噪声):
  1. S4 训练 (C2 配置, 40 iters) → 保存 rln/pln
  2. head 扩维: predictor[2] (V+2)*4→(V+2)*5 (前 4 块复制, 新块零), step_beta 重建
  3. 迁移评估 (禁用 think/compose/bank — 直测表示迁移):
     a. 零样本: adapt_graph(S5 support) → _metric(query), 按 cycle type 分桶
     b. 微调: S5 seen 继续 meta_step 20 iters → 同评估
  4. 对照: scratch (S5 从零, base 配置, 40 iters) → 同评估
  5. 迁移增益 = 迁移模型 unseen - scratch unseen (同进程)
"""
import os, sys, time, json, math, random
import itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_v31_meta_learning import batch_pairs, head_forward
from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import generator_signature, cycle_type, IDENT, _apply
from run_v35_fractal import V35_Learner, compose_swaps
from run_v35_16a_ablation import CELLS14, V35_Learner_A, run_cell_v14
from hibs_lnn.meta_rule_world import gen_task_data_split
import hibs_lnn.code_world as cw
import hibs_lnn.meta_rule_world as mrw

V = 200
N_RESP = 4          # S4
N_RESP5 = 5         # S5

# ============================================================
# S5 代数: 10 对换 / 120 perm / cycle type 5
# ============================================================
TRANS5 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
          (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
PERMS5 = list(itertools.permutations(range(5)))


def _apply5(p, g):
    q = list(p)
    a, b = g
    q[a], q[b] = q[b], q[a]
    return tuple(q)


def cycle_type5(perm):
    """S5 循环类型 → 类 id: 1=trans(2) 2=c3(3) 3=c4(4,1) 4=dbl(2,2) 5=c5(5) 6=(3,2)"""
    seen = set()
    lens = []
    for i in range(5):
        if i in seen:
            continue
        c = []
        j = i
        while j not in seen:
            seen.add(j)
            c.append(j)
            j = perm[j]
        if len(c) > 1:
            lens.append(len(c))
    lens.sort()
    if not lens:
        return 0
    if lens == [2]:
        return 1
    if lens == [3]:
        return 2
    if lens == [4]:
        return 3
    if lens == [2, 2]:
        return 4
    if lens == [5]:
        return 5
    if lens == [3, 2]:
        return 6
    return 0


def s5_groups():
    out = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    for p in PERMS5:
        out[cycle_type5(p)].append(p)
    return out


def make_s5_tasks(n_sup=24, n_qry=12, seed=42):
    """S5 seen/unseen 任务: seen = identity+8 对换+4 c3; unseen = c3 余+c4+c5+dbl."""
    g = s5_groups()
    rng = random.Random(seed)
    tasks = {}
    seen_perms = [(0, 1, 2, 3, 4)] + rng.sample(g[1], 8) + rng.sample(g[2], 4)
    unseen_perms = (rng.sample([p for p in g[2] if p not in seen_perms], 3)
                    + rng.sample(g[3], 6) + rng.sample(g[5], 4)
                    + rng.sample(g[4], 3))
    for i, p in enumerate(seen_perms):
        sup, qry = gen_task_data_split(
            None, V, n_support=n_sup, n_query=n_qry, seed=seed + i,
            remap=p, stop_at_halt=True)
        tasks['S5T%d' % i] = {
            'support': [(c, tg[:N_RESP5]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP5]) for (c, tg) in qry],
            'perm': p, 'seen': True}
    for i, p in enumerate(unseen_perms):
        sup, qry = gen_task_data_split(
            None, V, n_support=n_sup, n_query=n_qry, seed=1000 + i,
            remap=p, stop_at_halt=True)
        tasks['S5U%d' % i] = {
            'support': [(c, tg[:N_RESP5]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP5]) for (c, tg) in qry],
            'perm': p, 'seen': False}
    return tasks


# ============================================================
# head 扩维: n_resp 4→5 (predictor[2] + step_beta + SwiftTDHead)
# ============================================================
def extend_head(pln, learner):
    old = pln.predictor[2]
    new = nn.Linear(old.in_features, (V + 2) * N_RESP5)
    with torch.no_grad():
        new.weight[:(V + 2) * N_RESP].copy_(old.weight)
        new.bias[:(V + 2) * N_RESP].copy_(old.bias)
        new.weight[(V + 2) * N_RESP:].zero_()
        new.bias[(V + 2) * N_RESP:].zero_()
    pln.predictor[2] = new
    pln.n_response = N_RESP5
    pln.step_beta = nn.Parameter(
        torch.full((pln._n_params(),), math.log(pln.step_beta.exp().mean().item())))
    # SwiftTDHead: 输出 (V+2)*5, 状态重置 (与 scratch 公平)
    hd = learner.head
    from hibs_lnn.swifttd_head import SwiftTDHead
    learner.head = SwiftTDHead(
        hd.d_model, V + 2, N_RESP5, beta_init=hd.beta_init,
        kappa=hd.kappa, eta=hd.eta, eps=hd.eps, beta_min=hd.beta_min,
        input_dim=learner.intent_dim, tau_norm=hd.tau_norm)
    learner.head.reset_state()
    learner.n_resp = N_RESP5
    return pln, learner


# ============================================================
# 迁移评估: 纯表示路径 (禁用 think/compose/bank)
# ============================================================
def s5_eval(learner, tasks, names, k_adapt=20):
    """每任务: adapt_graph(support) → _metric(query). 返回 per-task regs_acc."""
    learner._use_think = False
    learner._use_compose = False
    learner.use_atoms = False
    accs = {}
    for n in names:
        t = tasks[n]
        with torch.no_grad():
            m = learner._define(t, detach=True)
        W = learner.adapt_graph(t['support'], K=k_adapt, m=m)
        with torch.no_grad():
            acc = learner._metric([w.detach() for w in W], t, m=m)[1]
        accs[n] = acc
    return accs


def struct_mean5(accs, tasks):
    by = {'c3': [], 'c4': [], 'c5': [], 'dbl': []}
    for n, a in accs.items():
        ct = cycle_type5(tasks[n]['perm'])
        key = {2: 'c3', 3: 'c4', 5: 'c5', 4: 'dbl'}.get(ct)
        if key:
            by[key].append(a)
    return {k: (sum(v) / len(v) if v else None) for k, v in by.items()}


def oml_step(learner, tasks, names, it, n_tasks=4):
    """纯 OML 外循环步: define(带图) → adapt_graph → CE(adapted W on support)
    → outer_opt 更新。绕过 V33.meta_step 的 probe 段 (view(4,4) 写死, S5 不兼容),
    微调/scratch 统一用此协议 (公平)。"""
    rng = random.Random(learner.seed + it)
    chosen = rng.sample(names, min(n_tasks, len(names)))
    loss = 0.0
    for rname in chosen:
        t = tasks[rname]
        m = learner._define(t, detach=False)   # 带图 (think/compose/bank 已禁)
        W = learner.adapt_graph(t['support'], K=learner.K, m=m)
        ctx, tg = batch_pairs(t['support'], max_batch=16)
        h = learner._rln_fwd(ctx, m)[:, -1, :]
        hin = learner._h_in_batch(h, m)
        pred = head_forward([w for w in W], hin, learner.intent_dim,
                            learner.world_vocab, learner.n_resp)
        loss = loss + F.cross_entropy(
            pred.view(-1, learner.world_vocab),
            tg.view(-1).clamp(0, learner.V + 1)) / len(chosen)
    learner.outer_opt.zero_grad()
    loss.backward()
    learner.outer_opt.step()
    return loss.item()


# ============================================================
# 驱动
# ============================================================
def train_s4(cfg, s1_state, meta_iters=40):
    """S4 训练 (C2 配置), 返回 (rln, pln, learner)."""
    from run_v35_16a_ablation import run_cell_v14 as rc
    tasks = build_v32_tasks(V, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
    tasks = make_regs_tasks(tasks)
    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']]
    T_STRUCT = build_t_struct(42)
    for n in tasks:
        tasks[n]['name'] = n
        tasks[n]['struct'] = (U_STRUCT.get(n) if not tasks[n]['seen']
                              else T_STRUCT.get(n, '?'))
        sig = generator_signature(tasks[n]['perm'])
        tasks[n]['gen_sig'] = sig[0] if sig else None
        tasks[n]['gen_len'] = sig[1] if sig else 0
    random.seed(42); torch.manual_seed(42)
    rln, pln = build_model_resp(V, N_RESP, s1_state)
    learner = V35_Learner_A(
        rln, pln, V, K=20, seed=42, sleep_iters=2, n_resp=N_RESP,
        use_intent=True, use_compose=True, use_atoms=True,
        use_hopfield=cfg.get('hopfield', False),
        adaptive_k=cfg.get('adaptive_k', False),
        active_query=cfg.get('active_query', False),
        aq_p3=cfg.get('aq_p3', 0.7),
        use_think=cfg.get('use_think', False), n_think=6,
        lambda_icl=cfg.get('lambda_icl', 0.5),
        lambda_combo=cfg.get('lambda_combo', 0.5),
        lambda_probe=cfg.get('lambda_probe', 1.0),
        aq_n=cfg.get('aq_n', 6), p_target=cfg.get('p_target', 0.5))
    for it in range(meta_iters):
        learner.meta_iters = it
        learner.meta_step(tasks, train_names)
    return rln, pln, learner


def train_s5_scratch(s1_state, meta_iters=40):
    """S5 从零训练 (base 配置: 无机制), 返回 per-task 评估."""
    tasks = make_s5_tasks(seed=42)
    seen = [n for n in tasks if tasks[n]['seen']][:13]
    unseen = [n for n in tasks if not tasks[n]['seen']]
    random.seed(42); torch.manual_seed(42)
    rln, pln = build_model_resp(V, N_RESP5, s1_state)
    learner = V35_Learner_A(
        rln, pln, V, K=20, seed=42, sleep_iters=2, n_resp=N_RESP5,
        use_intent=True, use_compose=False, use_atoms=False)
    t0 = time.time()
    for it in range(meta_iters):
        learner.meta_iters = it
        oml_step(learner, tasks, seen, it)
    print("    [scratch-S5] 训练 %.0fs" % (time.time() - t0), flush=True)
    accs = s5_eval(learner, tasks, unseen)
    return accs, tasks


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--s4-iters', type=int, default=40)
    ap.add_argument('--finetune-iters', type=int, default=20)
    ap.add_argument('--scratch-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.17 L2: S4→S5 群外推 (迁移泛化真测试)")
    print("=" * 70)
    s1 = None
    if not args.no_stage1:
        for cand in ['/tmp/v32_s1_state.pt', '/tmp/v31_2_s1_state.pt']:
            if os.path.exists(cand):
                s1 = torch.load(cand)
                s1 = {k: v.cpu().clone() for k, v in s1.items()}
                print("Stage1: %s" % cand)
                break
        else:
            print(">>> Stage1 缺失")
            return
    res = {}

    # 1) S4 训练 (C2 配置)
    print("\n== S4 训练 (C2) ==")
    t0 = time.time()
    rln, pln, learner = train_s4(CELLS14['C2'], s1, meta_iters=args.s4_iters)
    print("    S4 训练 %.0fs" % (time.time() - t0), flush=True)

    # 2) head 扩维 4→5
    print("== head 扩维 n_resp 4→5 ==")
    extend_head(pln, learner)

    # 3) S5 零样本迁移评估
    print("== S5 零样本迁移 (禁用机制, 纯表示) ==")
    s5_tasks = make_s5_tasks(seed=42)
    s5_seen = [n for n in s5_tasks if s5_tasks[n]['seen']]
    s5_unseen = [n for n in s5_tasks if not s5_tasks[n]['seen']]
    acc0 = s5_eval(learner, s5_tasks, s5_unseen)
    res['s5_zero_shot'] = {
        'unseen': round(sum(acc0.values()) / len(acc0), 4),
        'by_struct': struct_mean5(acc0, s5_tasks)}
    print("    zero-shot unseen=%.4f struct=%s" % (
        res['s5_zero_shot']['unseen'], res['s5_zero_shot']['by_struct']))

    # 4) 迁移微调: S5 seen 继续 meta_step 20 iters
    print("== 迁移微调 (S5 seen, %d iters) ==" % args.finetune_iters)
    # 保持纯表示路径 (think/compose/bank 均禁用, 与 scratch 公平)
    t0 = time.time()
    for it in range(args.finetune_iters):
        learner.meta_iters = 1000 + it
        oml_step(learner, s5_tasks, s5_seen, 1000 + it)
    print("    微调 %.0fs" % (time.time() - t0), flush=True)
    acc1 = s5_eval(learner, s5_tasks, s5_unseen)
    res['s5_finetuned'] = {
        'unseen': round(sum(acc1.values()) / len(acc1), 4),
        'by_struct': struct_mean5(acc1, s5_tasks)}
    print("    finetuned unseen=%.4f struct=%s" % (
        res['s5_finetuned']['unseen'], res['s5_finetuned']['by_struct']))

    # 5) scratch 对照: S5 从零 40 iters (同进程)
    print("== scratch 对照 (S5 从零, base, %d iters) ==" % args.scratch_iters)
    acc2, s5_tasks2 = train_s5_scratch(s1, meta_iters=args.scratch_iters)
    res['s5_scratch'] = {
        'unseen': round(sum(acc2.values()) / len(acc2), 4),
        'by_struct': struct_mean5(acc2, s5_tasks2)}
    print("    scratch unseen=%.4f struct=%s" % (
        res['s5_scratch']['unseen'], res['s5_scratch']['by_struct']))

    # 汇总
    print("\n" + "=" * 70)
    print("L2 迁移汇总 (同进程):")
    for k in ['s5_zero_shot', 's5_finetuned', 's5_scratch']:
        r = res[k]
        print("  %-14s unseen=%.4f %s" % (k, r['unseen'], r['by_struct']))
    out = ROOT / 'results' / 'v35_17_s5_transfer.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("JSON:", out)


if __name__ == '__main__':
    main()
