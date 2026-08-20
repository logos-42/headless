"""V35.22 — S6 递推: 群外推一般性验证 (2026-08-16)

问题: 价值函数训练的表示能否继续外推到 S6 (6 元素/15 对换/720 perm)?
递推链: S4→S5 已验证 (V35.20/21), 现在 S5-value → S6。

协议 (同进程):
  1. S5-value 训练 (价值函数提议, 40 iters, 5 寄存器世界)
  2. patch 世界到 6 寄存器 → head 扩维 5→6 (前块复制新块零)
  3. S6 零样本迁移评估 (纯表示路径, 按 cycle type 分桶)
  4. 对照: S6-scratch (从零 oml_step 40 iters, 同进程)

通用 cycle_type_n: 返回 (lens...) 精确循环类型, 分桶 c3/c4/c5/c6/dbl。
"""
import os, sys, time, json, math, random
import itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from run_v31_meta_learning import batch_pairs, head_forward
from run_v32_1_matrix import build_model_resp
from run_v35_16a_ablation import V35_Learner_A
from run_v35_17_s5_transfer import (
    make_s5_tasks, s5_eval, N_RESP5)
from run_v35_21_s5_value import train_s5, oml_step, S5ValueProposer
import hibs_lnn.code_world as cw
import hibs_lnn.meta_rule_world as mrw
from hibs_lnn.meta_rule_world import gen_task_data_split

V = 200
N_RESP6 = 6

# ============================================================
# S6 代数
# ============================================================
TRANS6 = [(i, j) for i in range(6) for j in range(i + 1, 6)]
IDENT6 = tuple(range(6))
PERMS6 = list(itertools.permutations(range(6)))


def cycle_type_n(perm):
    n = len(perm)
    seen = set()
    lens = []
    for i in range(n):
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
    return tuple(lens)


def struct_key6(ct):
    """循环类型 → 分桶键."""
    if not ct:
        return 'id'
    m = max(ct)
    if m == 6:
        return 'c6'
    if m == 5:
        return 'c5'
    if m == 4:
        return 'c4'
    if m == 3:
        return 'c3'
    return 'dbl'   # (2,2,2) / (2,2) 等


def make_s6_tasks(n_sup=16, n_qry=8, seed=42):
    g = {}
    for p in PERMS6:
        k = struct_key6(cycle_type_n(p))
        g.setdefault(k, []).append(p)
    rng = random.Random(seed)
    seen_perms = ([IDENT6] + rng.sample(g['dbl'][:10], 6)  # 6 对换作原子
                  + rng.sample(g['c3'], 4))
    # 去重
    seen_perms = list(dict.fromkeys(seen_perms))
    unseen_pool = [p for p in PERMS6 if p not in seen_perms
                   and p != IDENT6]
    rng.shuffle(unseen_pool)
    c3s = [p for p in unseen_pool if struct_key6(cycle_type_n(p)) == 'c3'][:4]
    c4s = [p for p in unseen_pool if struct_key6(cycle_type_n(p)) == 'c4'][:6]
    c5s = [p for p in unseen_pool if struct_key6(cycle_type_n(p)) == 'c5'][:6]
    c6s = [p for p in unseen_pool if struct_key6(cycle_type_n(p)) == 'c6'][:6]
    dbls = [p for p in unseen_pool if struct_key6(cycle_type_n(p)) == 'dbl'][:4]
    unseen_perms = c3s + c4s + c5s + c6s + dbls
    tasks = {}
    for i, p in enumerate(seen_perms):
        sup, qry = gen_task_data_split(
            None, V, n_support=n_sup, n_query=n_qry, seed=seed + i,
            remap=p, stop_at_halt=True)
        tasks['S6T%d' % i] = {
            'support': [(c, tg[:N_RESP6]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP6]) for (c, tg) in qry],
            'perm': p, 'seen': True}
    for i, p in enumerate(unseen_perms):
        sup, qry = gen_task_data_split(
            None, V, n_support=n_sup, n_query=n_qry, seed=2000 + i,
            remap=p, stop_at_halt=True)
        tasks['S6U%d' % i] = {
            'support': [(c, tg[:N_RESP6]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP6]) for (c, tg) in qry],
            'perm': p, 'seen': False}
    return tasks


def extend_head_n(pln, learner, old_n, new_n):
    old = pln.predictor[2]
    new = torch.nn.Linear(old.in_features, (V + 2) * new_n)
    with torch.no_grad():
        new.weight[:(V + 2) * old_n].copy_(old.weight)
        new.bias[:(V + 2) * old_n].copy_(old.bias)
        new.weight[(V + 2) * old_n:].zero_()
        new.bias[(V + 2) * old_n:].zero_()
    pln.predictor[2] = new
    pln.n_response = new_n
    pln.step_beta = torch.nn.Parameter(
        torch.full((pln._n_params(),),
                   math.log(pln.step_beta.exp().mean().item())))
    from hibs_lnn.swifttd_head import SwiftTDHead
    hd = learner.head
    learner.head = SwiftTDHead(
        hd.d_model, V + 2, new_n, beta_init=hd.beta_init,
        kappa=hd.kappa, eta=hd.eta, eps=hd.eps, beta_min=hd.beta_min,
        input_dim=learner.intent_dim, tau_norm=hd.tau_norm)
    learner.head.reset_state()
    learner.n_resp = new_n
    return pln, learner


def struct_mean6(accs, tasks):
    by = {'c3': [], 'c4': [], 'c5': [], 'c6': [], 'dbl': []}
    for n, a in accs.items():
        k = struct_key6(cycle_type_n(tasks[n]['perm']))
        if k in by:
            by[k].append(a)
    return {k: (sum(v) / len(v) if v else None) for k, v in by.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.22 — S6 递推: 群外推一般性 (S5-value → S6)")
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

    # 1) S5-value 训练 (5 寄存器世界, 直接实现以拿到 learner)
    print("\n== S5-value 训练 ==")
    random.seed(42); torch.manual_seed(42)
    tasks = make_s5_tasks(seed=42)
    seen5 = [n for n in tasks if tasks[n]['seen']]
    unseen5 = [n for n in tasks if not tasks[n]['seen']]
    rln, pln = build_model_resp(V, N_RESP5, s1)
    learner = V35_Learner_A(
        rln, pln, V, K=20, seed=42, sleep_iters=2, n_resp=N_RESP5,
        use_intent=True, use_compose=False, use_atoms=False)
    proposer = S5ValueProposer([tasks[n]['perm'] for n in seen5])
    t0 = time.time()
    for it in range(args.meta_iters):
        learner.meta_iters = it
        extra = []
        ps = proposer.propose(n=2)
        for i, p in enumerate(ps):
            key = '__v%d' % i
            sup, qry = gen_task_data_split(
                None, V, n_support=8, n_query=8, seed=10000 + it * 10 + i,
                remap=p, stop_at_halt=True)
            tasks[key] = {
                'support': [(c, tg[:N_RESP5]) for (c, tg) in sup],
                'query': [(c, tg[:N_RESP5]) for (c, tg) in qry],
                'perm': p, 'seen': False, 'gen_sig': None}
            extra.append(key)
        oml_step(learner, tasks, seen5, it, extra=extra)
        for i in range(2):
            tasks.pop('__v%d' % i, None)
    print("    S5-value 训练 %.0fs" % (time.time() - t0), flush=True)

    # 2) patch S6 世界
    print("== patch S6 (6 寄存器) ==")
    cw.N_REGISTERS = 6
    mrw.N_REGISTERS = 6
    mrw.MetaRuleWorld.READ_WINDOW = 7

    # 3) head 扩维 5→6 + S6 零样本
    print("== head 扩维 5→6 + S6 零样本 ==")
    extend_head_n(pln, learner, N_RESP5, N_RESP6)
    s6_tasks = make_s6_tasks(seed=42)
    s6_unseen = [n for n in s6_tasks if not s6_tasks[n]['seen']]
    acc0 = s5_eval(learner, s6_tasks, s6_unseen)
    res['s5v_s6_zero'] = {
        'unseen': round(sum(acc0.values()) / len(acc0), 4),
        'by_struct': struct_mean6(acc0, s6_tasks)}
    print("    S5-value→S6 zero unseen=%.4f %s" % (
        res['s5v_s6_zero']['unseen'], res['s5v_s6_zero']['by_struct']))

    # 4) S6 scratch 对照 (同进程, 6 寄存器世界)
    print("== S6-scratch 对照 ==")
    random.seed(42); torch.manual_seed(42)
    rln6, pln6 = build_model_resp(V, N_RESP6, s1)
    l6 = V35_Learner_A(
        rln6, pln6, V, K=20, seed=42, sleep_iters=2, n_resp=N_RESP6,
        use_intent=True, use_compose=False, use_atoms=False)
    s6_seen = [n for n in s6_tasks if s6_tasks[n]['seen']]
    t0 = time.time()
    for it in range(args.meta_iters):
        l6.meta_iters = it
        oml_step(l6, s6_tasks, s6_seen, it)
    print("    S6-scratch 训练 %.0fs" % (time.time() - t0), flush=True)
    acc1 = s5_eval(l6, s6_tasks, s6_unseen)
    res['s6_scratch'] = {
        'unseen': round(sum(acc1.values()) / len(acc1), 4),
        'by_struct': struct_mean6(acc1, s6_tasks)}
    print("    S6-scratch unseen=%.4f %s" % (
        res['s6_scratch']['unseen'], res['s6_scratch']['by_struct']))

    print("\n" + "=" * 70)
    print("S6 递推汇总:")
    for k in ['s5v_s6_zero', 's6_scratch']:
        r = res[k]
        print("  %-14s unseen=%.4f %s" % (k, r['unseen'], r['by_struct']))
    out = ROOT / 'results' / 'v35_22_s6.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("JSON:", out)


if __name__ == '__main__':
    main()
