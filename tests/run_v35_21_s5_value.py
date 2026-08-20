"""V35.21 — S5 内价值函数直接训练 (120 候选池) + 多 seed 确认 (2026-08-16)

Part A: 多 seed 确认 V1 迁移优势 → 调 run_v35_20_value_transfer.py
  --configs V1 --seed 7/123

Part B: S5 世界内直接用价值函数 (简约+自洽+覆盖) 训练
  候选池 = 120 perm - 13 seen = 107 → "选择"真正有意义
  对比: S5-scratch (base oml_step) vs S5-value (价值提议注入), 同进程

关键: 价值函数是纯代数 (与模型无关), S5 版用 cycle_type5 + 10 对换。
训练绕过 V33.meta_step (probe view(4,4) 写死), 用 oml_step 协议。
"""
import os, sys, time, json, math, random
import itertools
from collections import deque, Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from run_v31_meta_learning import batch_pairs, head_forward
from run_v32_1_matrix import build_model_resp
from run_v35_16a_ablation import V35_Learner_A
from run_v35_17_s5_transfer import (
    make_s5_tasks, s5_eval, struct_mean5, TRANS5, PERMS5, cycle_type5,
    N_RESP5, V as _V)
from run_v35_20_value_transfer import train_s4_cfg

V = 200

# ============================================================
# S5 字长 (BFS 最短对换分解, 预计算)
# ============================================================
IDENT5 = (0, 1, 2, 3, 4)
_S5_WORDLEN = {}


def _apply5(p, g):
    q = list(p)
    a, b = g
    q[a], q[b] = q[b], q[a]
    return tuple(q)


def s5_wordlen(p):
    if p in _S5_WORDLEN:
        return _S5_WORDLEN[p]
    q = deque([(IDENT5, 0)])
    seen = {IDENT5}
    while q:
        cur, d = q.popleft()
        if cur == p:
            _S5_WORDLEN[p] = d
            return d
        for g in TRANS5:
            nxt = _apply5(cur, g)
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return 99


def struct_sim5(p, q):
    cp, cq = cycle_type5(p), cycle_type5(q)
    gp, gq = s5_wordlen(p), s5_wordlen(q)
    s = 0.6 if cp == cq else 0.0
    s += 0.4 * max(0.0, 1 - abs(gp - gq) / 5.0)
    return s


def _compose5(p, q):
    return tuple(p[q[x]] for x in range(5))


# ============================================================
# S5 价值函数提议器 (纯代数)
# ============================================================
class S5ValueProposer:
    def __init__(self, learned_perms, lambda_sim=1.0, lambda_con=1.0,
                 lambda_cov=1.0, tau=1.0, k=2):
        self.learned = list(learned_perms)
        self.learned_types = {cycle_type5(p) for p in self.learned}
        self.freq = Counter()
        self.ls, self.lc, self.lcov = lambda_sim, lambda_con, lambda_cov
        self.tau, self.k = tau, k
        self.cands = [p for p in PERMS5 if p not in self.learned]

    def sim(self, p):
        return 1.0 - max(struct_sim5(p, q) for q in self.learned)

    def con(self, p):
        n_bad = n = 0
        for tau in self.learned:
            ct = cycle_type5(_compose5(p, tau))
            n += 1
            if ct not in self.learned_types:
                n_bad += 1
        return 1.0 - n_bad / max(n, 1)

    def value(self, p):
        return (self.ls * self.sim(p) + self.lc * self.con(p)
                + self.lcov / (1 + self.freq.get(p, 0)))

    def propose(self, n=None):
        n = n or self.k
        scores = torch.tensor([self.value(p) for p in self.cands])
        probs = (scores / self.tau).softmax(-1)
        idxs = torch.multinomial(probs, min(n, len(self.cands)),
                                 replacement=False)
        out = []
        for i in idxs.tolist():
            p = self.cands[i]
            self.freq[p] += 1
            out.append(p)
        return out


# ============================================================
# Part B: S5 内训练 (oml_step + 提议注入)
# ============================================================
def oml_step(learner, tasks, names, it, n_tasks=4, extra=None):
    rng = random.Random(learner.seed + it)
    chosen = rng.sample(names, min(n_tasks, len(names)))
    if extra:
        chosen = chosen + extra
    loss = 0.0
    for rname in chosen:
        t = tasks[rname]
        m = learner._define(t, detach=False)
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


def make_s5_value_task(perm, seed, n_sup=8, n_qry=8):
    from hibs_lnn.meta_rule_world import gen_task_data_split
    sup, qry = gen_task_data_split(
        None, V, n_support=n_sup, n_query=n_qry, seed=seed,
        remap=perm, stop_at_halt=True)
    return {'support': [(c, tg[:N_RESP5]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP5]) for (c, tg) in qry],
            'perm': perm, 'seen': False, 'gen_sig': None}


def train_s5(mode, s1_state, meta_iters=40, seed=42, n_prop=2):
    """S5 训练: mode='scratch' (纯 oml_step) 或 'value' (提议注入)."""
    tasks = make_s5_tasks(seed=42)
    seen = [n for n in tasks if tasks[n]['seen']]
    unseen = [n for n in tasks if not tasks[n]['seen']]
    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(V, N_RESP5, s1_state)
    learner = V35_Learner_A(
        rln, pln, V, K=20, seed=seed, sleep_iters=2, n_resp=N_RESP5,
        use_intent=True, use_compose=False, use_atoms=False)
    proposer = (S5ValueProposer([tasks[n]['perm'] for n in seen])
                if mode == 'value' else None)
    t0 = time.time()
    for it in range(meta_iters):
        learner.meta_iters = it
        extra = None
        if proposer is not None:
            ps = proposer.propose(n=n_prop)
            extra = []
            for i, p in enumerate(ps):
                key = '__v%d' % i
                tasks[key] = make_s5_value_task(p, 10000 + it * 10 + i)
                extra.append(key)
        oml_step(learner, tasks, seen, it, extra=extra)
        if proposer is not None:
            for i in range(n_prop):
                tasks.pop('__v%d' % i, None)
    print("    [S5-%s seed%d] 训练 %.0fs" % (mode, seed, time.time() - t0),
          flush=True)
    accs = s5_eval(learner, tasks, unseen)
    return accs, tasks, proposer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='B', choices=['A', 'B'])
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--seeds', default='7,123')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.21 — S5 内价值函数直接训练 + 多 seed 确认")
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

    if args.part == 'A':
        # 多 seed 确认 V1 迁移 (V35.20 协议)
        for sd in [int(x) for x in args.seeds.split(',')]:
            print("\n== Part A: V1 seed %d ==" % sd)
            cfg = dict(world='new', lm='compose', alpha=1.0,
                       hopfield=True, adaptive_k=True, use_think=True,
                       n_think=6, lambda_icl=0.5, lambda_combo=0.5,
                       active_query=True, aq_n=6, aq_p3=0.7,
                       lambda_probe=1.0, use_perm=True, lambda_perm=1.0,
                       n_prop=2, prop_tau=1.0, lambda_sim=1.0,
                       lambda_con=1.0, lambda_cov=1.0,
                       label='V1-value+cov')
            from run_v35_19_value_proposer import V35_Learner_V
            from run_v35_17_s5_transfer import (
                make_s5_tasks as m5, extend_head)
            rln, pln, learner = train_s4_cfg(cfg, V35_Learner_V, s1,
                                             meta_iters=args.meta_iters,
                                             seed=sd)
            extend_head(pln, learner)
            t5 = m5(seed=42)
            u5 = [n for n in t5 if not t5[n]['seen']]
            acc0 = s5_eval(learner, t5, u5)
            res['V1_s%d_zero' % sd] = {
                'unseen': round(sum(acc0.values()) / len(acc0), 4),
                'by_struct': struct_mean5(acc0, t5)}
            print("    zero unseen=%.4f %s" % (
                res['V1_s%d_zero' % sd]['unseen'],
                res['V1_s%d_zero' % sd]['by_struct']))
        out = ROOT / 'results' / 'v35_21_multiseed.json'
    else:
        # Part B: S5 内 scratch vs value (同进程)
        print("\n== Part B: S5-scratch (seed %d) ==" % args.seed)
        acc_s, tasks_s, _ = train_s5('scratch', s1, args.meta_iters,
                                     seed=args.seed)
        res['s5_scratch'] = {
            'unseen': round(sum(acc_s.values()) / len(acc_s), 4),
            'by_struct': struct_mean5(acc_s, tasks_s)}
        print("    scratch unseen=%.4f %s" % (
            res['s5_scratch']['unseen'], res['s5_scratch']['by_struct']))
        print("\n== Part B: S5-value (价值函数提议, seed %d) ==" % args.seed)
        acc_v, tasks_v, prop = train_s5('value', s1, args.meta_iters,
                                        seed=args.seed)
        res['s5_value'] = {
            'unseen': round(sum(acc_v.values()) / len(acc_v), 4),
            'by_struct': struct_mean5(acc_v, tasks_v)}
        res['value_stats'] = {
            'prop_n': sum(prop.freq.values()),
            'freq_top': prop.freq.most_common(5),
            'cands': len(prop.cands)}
        print("    value unseen=%.4f %s" % (
            res['s5_value']['unseen'], res['s5_value']['by_struct']))
        print("    proposer: %s" % res['value_stats'])
        out = ROOT / 'results' / ('v35_21_s5_value_s%d.json' % args.seed)

    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("\nJSON:", out)


if __name__ == '__main__':
    main()
