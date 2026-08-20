"""V35.19 — 价值函数提议器 (简约性 + 自洽性) + redefine — 2026-08-15 用户理论

用户定义:
  简约性: 简约点 = 全局扫描后不可归约的原理 (真正的新原理)
  自洽性: 彼此不矛盾 (兼容已学原理)
  两个都是价值函数的选择方向 → V(p) = λ_sim·sim(p) + λ_con·con(p)
  redefine: 矛盾候选不丢弃, 而是注入学习 (表示被迫重新定义边界)
  loop 调整, 目的达成为止

实现:
  sim(p) = 1 - max_{q∈learned} struct_sim(p,q)   (cycle type + 字长相似)
  con(p) = 1 - conflict(p)                        (p∘τ 产生已学类型外结构的比例)
  redefine: con < τ 的候选仍注入 (redefine 通道, 统计计数)

Cell:
  V0 = P1 复现 (温度提议参照)
  V1 = 价值函数提议 (top-k 按 V 评分)
  V2 = V1 + redefine (矛盾候选注入)
"""
import os, sys, time, json, math, random
import itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import generator_signature, cycle_type, TRANS, IDENT, _apply
from hibs_lnn.meta_rule_world import gen_task_data_split
from run_v35_16a_ablation import PERMS24
from run_v35_18_proposer import V35_Learner_P


def _compose(p, q):
    """p∘q = p(q(x))"""
    return tuple(p[q[x]] for x in range(len(p)))


def struct_sim(p, q):
    """结构相似度: cycle type 相同 0.6 + 字长接近 0.4."""
    cp, cq = cycle_type(p), cycle_type(q)
    gp = generator_signature(p)[1]
    gq = generator_signature(q)[1]
    s = 0.6 if cp == cq else 0.0
    s += 0.4 * max(0.0, 1 - abs(gp - gq) / 3.0)
    return s


class V35_Learner_V(V35_Learner_P):
    """+ 价值函数提议: V(p) = λ_sim·sim(p) + λ_con·con(p) + redefine."""

    def __init__(self, rln, pln, V, use_compose=False, alpha=1.0,
                 use_atoms=True, use_fractal=False, use_hopfield=False,
                 adaptive_k=False, active_query=False, **kw):
        self._lambda_sim = kw.pop('lambda_sim', 1.0)
        self._lambda_con = kw.pop('lambda_con', 1.0)
        self._lambda_cov = kw.pop('lambda_cov', 0.0)
        self._redefine_thresh = kw.pop('redefine_thresh', 0.5)
        self._use_redefine = kw.pop('use_redefine', False)
        self._prop_k = kw.pop('prop_k', 2)          # top-k 提议数
        self._prop_tau = kw.pop('prop_tau', 1.0)    # 评分 softmax 温度
        super().__init__(rln, pln, V, use_compose=use_compose, alpha=alpha,
                         use_atoms=use_atoms, use_fractal=use_fractal,
                         use_hopfield=use_hopfield, adaptive_k=adaptive_k,
                         active_query=active_query, **kw)
        self.learned_perms = [_apply(IDENT, g) for g in TRANS]  # 初始: 6 对换
        self.learned_types = {cycle_type(p) for p in self.learned_perms}
        self.value_scores = []
        self.redefine_count = 0
        from collections import Counter
        self.prop_freq = Counter()

    def _sync_learned(self, tasks, train_names):
        """已学原理 = 训练流 seen 任务的 perm (identity + 6 对换 + 5 c3)."""
        perms = [tasks[n]['perm'] for n in train_names]
        self.learned_perms = list(set(perms))
        self.learned_types = {cycle_type(p) for p in self.learned_perms}

    # ── 价值函数 ──────────────────────────────────────────
    def _sim(self, p):
        """简约性: 1 - 与已学结构的最大相似 (不可归约度)."""
        if not self.learned_perms:
            return 1.0
        return 1.0 - max(struct_sim(p, q) for q in self.learned_perms)

    def _con(self, p):
        """自洽性: 1 - 矛盾度 (p∘τ 产生已学类型外结构的比例)."""
        if not self.learned_perms:
            return 1.0
        n_bad = 0
        n = 0
        for tau in self.learned_perms:
            ct = cycle_type(_compose(p, tau))
            n += 1
            if ct not in self.learned_types:
                n_bad += 1
        if n == 0:
            return 1.0
        return 1.0 - n_bad / n

    def _value(self, p):
        return self._lambda_sim * self._sim(p) + self._lambda_con * self._con(p)

    # ── 提议: 候选池评分 → top-k (探索: softmax 温度) ──────
    def _propose(self, tasks, train_names):
        if self.perm_head is None:
            return []
        cands = [p for p in PERMS24 if p not in self.learned_perms]
        if not cands:
            return []
        scores = torch.tensor(
            [self._value(p) + self._lambda_cov / (1 + self.prop_freq.get(p, 0))
             for p in cands])
        self.value_scores.append(scores.mean().item())
        probs = (scores / self._prop_tau).softmax(-1)
        idxs = torch.multinomial(probs, min(self._prop_k, len(cands)),
                                 replacement=False)
        out = []
        rng = random.Random(self.seed + self._meta_iters)
        for i in idxs.tolist():
            p = cands[i]
            self.prop_history.append(p)          # 记录 (允许重复)
            self.prop_freq[p] += 1
            self.prop_struct.append(cycle_type(p))
            # redefine 检测: 矛盾候选 (con 低) 仍注入
            if self._use_redefine and self._con(p) < self._redefine_thresh:
                self.redefine_count += 1
            for _try in range(5):
                seed = rng.randint(0, 10 ** 6)
                sup, qry = gen_task_data_split(
                    None, self.V, n_support=4, n_query=8, seed=seed,
                    remap=p, stop_at_halt=True)
                if sup and qry:
                    break
            if not sup or not qry:
                continue   # 空数据保护 (stop_at_halt 短 episode)
            sig = generator_signature(p)
            out.append({'perm': p, 'support': sup, 'query': qry,
                        'gen_sig': sig[0] if sig else None})
        return out


def run_cell_v19(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
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
    rln, pln = build_model_resp(200, 4, s1_state)
    use_compose = (lm == 'compose')
    learner = V35_Learner_V(
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
        lambda_icl=cfg.get('lambda_icl', 0.0),
        alpha=cfg.get('alpha', 1.0),
        beta_train=cfg.get('beta_train', 3.0),
        beta_test=cfg.get('beta_test', 10.0),
        D=cfg.get('D', 1.5),
        aq_n=cfg.get('aq_n', 4),
        aq_max_combo=cfg.get('aq_max_combo', 3),
        k_lam=cfg.get('k_lam', 0.005),
        k_patience=cfg.get('k_patience', 3),
        use_anchor=cfg.get('use_anchor', False),
        lambda_anchor=cfg.get('lambda_anchor', 0.5),
        use_gencomp=cfg.get('use_gencomp', False),
        lambda_gen=cfg.get('lambda_gen', 0.5),
        p_target=cfg.get('p_target', 0.0),
        n_prop=cfg.get('n_prop', 0),
        prop_tau=cfg.get('prop_tau', 2.0),
        lambda_sim=cfg.get('lambda_sim', 1.0),
        lambda_con=cfg.get('lambda_con', 1.0),
        lambda_cov=cfg.get('lambda_cov', 0.0),
        redefine_thresh=cfg.get('redefine_thresh', 0.5),
        use_redefine=cfg.get('use_redefine', False),
        prop_k=cfg.get('prop_k', 2))
    learner._sync_learned(tasks, train_names)
    t0 = time.time()
    n_iters = cfg.get('meta_iters', meta_iters)
    for it in range(n_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_names)
        if verbose and (it + 1) % 20 == 0:
            print("    [%s] it %d/%d loss=%.4f" % (key, it + 1, n_iters, loss),
                  flush=True)
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
    if getattr(learner, 'perm_head', None) is not None:
        pa = learner.perm_acc
        aux['perm_acc'] = {
            'seen': (round(sum(pa['seen']) / len(pa['seen']), 3)
                     if pa['seen'] else None),
            'unseen': (round(sum(pa['unseen']) / len(pa['unseen']), 3)
                       if pa['unseen'] else None)}
    if getattr(learner, 'prop_history', None):
        from collections import Counter
        ph = learner.prop_history
        aux['prop_n'] = len(ph)
        aux['prop_unique'] = len(set(ph))
        aux['prop_struct'] = dict(Counter(learner.prop_struct))
        freq = Counter(ph)
        aux['prop_freq_top'] = [(p, c) for p, c in freq.most_common(5)]
        aux['prop_dup_rate'] = round(1 - len(set(ph)) / max(len(ph), 1), 3)
        # 价值-频率: top3 高价值结构的平均提议频率 vs 全体平均
        by_v = sorted(PERMS24, key=learner._value, reverse=True)
        top3 = by_v[:3]
        aux['prop_top3_value_avg_freq'] = round(
            sum(freq.get(p, 0) for p in top3) / 3, 2)
        aux['prop_top3_value_perms'] = [str(p) for p in top3]
        aux['prop_freq_mean'] = round(sum(freq.values()) / len(freq), 2)
    if getattr(learner, 'value_scores', None):
        vs = learner.value_scores
        aux['value_score_mean'] = round(sum(vs) / len(vs), 3)
    if getattr(learner, 'redefine_count', 0):
        aux['redefine_count'] = learner.redefine_count
    n_params = (sum(p.numel() for p in rln.parameters())
                + sum(p.numel() for p in pln.parameters()))
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'factors': {'lambda_sim': cfg.get('lambda_sim', 1.0),
                       'lambda_con': cfg.get('lambda_con', 1.0),
                       'lambda_cov': cfg.get('lambda_cov', 0.0),
                       'use_redefine': cfg.get('use_redefine', False),
                       'prop_k': cfg.get('prop_k', 2),
                       'n_prop': cfg.get('n_prop', 0)},
           'time_s': round(time.time() - t0, 1),
           'n_params': n_params,
           'seen_acc': {n: st['acc_after'][n] for n in train_names},
           'unseen_acc': {n: st['acc_after'][n] for n in unseen_names},
           'unseen_by_struct': struct_mean,
           'aux': aux}
    u = rec['unseen_acc']; ua = sum(u.values()) / len(u)
    print((u"  [%s] unseen=%.4f struct=%s" % (key, ua, struct_mean)),
          flush=True)
    return rec


BASE = dict(world='new', lm='compose', alpha=1.0,
            hopfield=True, adaptive_k=True, use_think=True, n_think=6,
            lambda_icl=0.5, lambda_combo=0.5, active_query=True,
            aq_n=6, aq_p3=0.7, lambda_probe=1.0)

CELLS19 = {
    'V0': dict(BASE, use_perm=True, lambda_perm=1.0, n_prop=2,
               prop_tau=1.0, lambda_sim=0.0, lambda_con=0.0,
               label='V0-random-ref'),
    'V1': dict(BASE, use_perm=True, lambda_perm=1.0, n_prop=2,
               prop_tau=1.0, lambda_sim=1.0, lambda_con=1.0,
               label='V1-value-proposer'),
    'V2': dict(BASE, use_perm=True, lambda_perm=1.0, n_prop=2,
               prop_tau=1.0, lambda_sim=1.0, lambda_con=1.0,
               use_redefine=True, label='V2-value+redefine'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='V0,V1,V2')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    ap.add_argument('--out', default='v35_19_value_proposer.json')
    ap.add_argument('--lambda-sim', type=float, default=None)
    ap.add_argument('--lambda-con', type=float, default=None)
    ap.add_argument('--lambda-cov', type=float, default=None)
    ap.add_argument('--use-redefine', action='store_true')
    ap.add_argument('--prop-k', type=int, default=None)
    args = ap.parse_args()
    print("=" * 70)
    print("V35.19 价值函数提议器 (简约性+自洽性) + redefine")
    print("=" * 70)
    tasks = build_v32_tasks(200, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
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
    res = []
    for key in [c.strip() for c in args.cells.split(',')]:
        cfg = dict(CELLS19[key])
        if args.lambda_sim is not None:
            cfg['lambda_sim'] = args.lambda_sim
        if args.lambda_con is not None:
            cfg['lambda_con'] = args.lambda_con
        if args.lambda_cov is not None:
            cfg['lambda_cov'] = args.lambda_cov
        if args.use_redefine:
            cfg['use_redefine'] = True
        if args.prop_k is not None:
            cfg['prop_k'] = args.prop_k
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell_v19(key, cfg, s1, tasks, U_STRUCT,
                             meta_iters=args.meta_iters)
        except Exception:
            import traceback; traceback.print_exc()
            print("!!! %s 失败" % key)
            continue
        res.append(r)
    out = ROOT / 'results' / args.out
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s value=%.3f redefine=%s" % (
            r['cell'], ua, r['unseen_by_struct'],
            r['aux'].get('value_score_mean', -1),
            r['aux'].get('redefine_count', 0)))


if __name__ == '__main__':
    main()
