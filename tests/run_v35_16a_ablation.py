"""V35.14 四方向消融 — c4 锁死带 (2026-08-15, 用户发起)

以 S0 (V35.12 最强, c4 0.0964 首次超基线) 为参照, 消融四个方向:

  A1 符号锚定   : CycleTypeAnchor — m→5 类 cycle type 粗粒度监督
                  (给 c4 类身份, 补"无不动点锚"缺失)
  B1 结构化复合 : GenCompose — 回读→逐步复合闭环
                  (perm_head soft 回读 + one-hot 生成元 → 共享两跳 MLP
                  学生成元右乘代数, 监督 = 组合 perm; 梯度回流表示层)
  C2 定向采样   : AQ4C — 目标结构采样 (50% 概率定向 c4, BFS 最短对换分解)
  D  评估视角   : perm_acc_by_struct — probe 回读按 cycle type 分桶 (公共)

纪律: 同进程共享 Stage1 / 随机基线参照 A0 / 结构化报告 JSON+MD。
"""
import os, sys, time, json, math, random
import itertools
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import (
    generator_signature, cycle_type, TRANS, IDENT, _apply, CYCLET)
from run_v34_3_cpg import N_GEN
from hibs_lnn.meta_rule_world import gen_task_data_split
from run_v35_fractal import (
    V35_Learner, ActiveQueryGenerator, compose_swaps)

# ============================================================
# 预计算: 24 perm / 6 个 c4 的最短对换分解 (BFS, 与 compose_swaps 同语义)
# ============================================================
PERMS24 = list(itertools.permutations(range(4)))
PERM_IDX = {p: i for i, p in enumerate(PERMS24)}
C4_PERMS = [p for p in PERMS24 if cycle_type(p) == 3]   # 6 个 4-cycle


def _shortest_decomp(target):
    q = deque([(IDENT, [])])
    seen = {IDENT}
    while q:
        p, path = q.popleft()
        if p == target:
            return path
        for gi in range(6):
            np = _apply(p, TRANS[gi])
            if np not in seen:
                seen.add(np)
                q.append((np, path + [gi]))
    return None


C4_DECOMP = {p: _shortest_decomp(p) for p in C4_PERMS}


def _onehot6(gi, device):
    v = torch.zeros(6, device=device)
    v[gi] = 1.0
    return v


# ============================================================
# C2: AQ4C — 目标结构采样 (c4 定向)
# ============================================================
class AQ4C(ActiveQueryGenerator):
    """p_target 概率从 6 个 c4 排列轮流选目标 (BFS 最短对换分解),
    其余走父类采样. gen_history 存 4 元组 (perm, sig, ln, gens)."""

    def __init__(self, V, seed=42, n_support=8, n_query=8, max_combo=3,
                 p_combo3=0.6, p_target=0.5):
        super().__init__(V, seed=seed, n_support=n_support, n_query=n_query,
                         max_combo=max_combo, p_combo3=p_combo3)
        self.p_target = p_target
        self._tgt_cycle = 0

    def _pick_gens_target(self, bank):
        for _ in range(len(C4_PERMS)):
            target = C4_PERMS[self._tgt_cycle % len(C4_PERMS)]
            self._tgt_cycle += 1
            gens = C4_DECOMP[target]
            if all(bank.atoms[gi] is not None for gi in gens):
                return list(gens)
        return None

    def _pick_gens(self, bank):
        if self.p_target > 0 and self.rng.random() < self.p_target:
            g = self._pick_gens_target(bank)
            if g is not None:
                return g
        return super()._pick_gens(bank)

    def generate(self, bank, n=4):
        out = []
        seen_sigs = set(h[1] for h in self.gen_history)
        for _ in range(n * 3):
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
                None, self.V, n_support=self.n_support,
                n_query=self.n_query, seed=seed, remap=perm,
                stop_at_halt=True)
            out.append({'perm': perm, 'gen_sig': sig, 'gen_len': ln,
                        'support': sup, 'query': qry, 'gens': tuple(gens)})
            self.gen_history.append((perm, sig_t, ln, tuple(gens)))
        return out


# ============================================================
# V35_Learner_A: + A1 符号锚定 / B1 结构化复合
# ============================================================
class V35_Learner_A(V35_Learner):
    def __init__(self, rln, pln, V, use_compose=False, alpha=1.0,
                 use_atoms=True, use_fractal=False, use_hopfield=False,
                 adaptive_k=False, active_query=False, **kw):
        # 新参数必须先 pop, 不泄漏给上层 (V35 坑)
        self._use_anchor = kw.pop('use_anchor', False)
        self._lambda_anchor = kw.pop('lambda_anchor', 0.5)
        self._use_gencomp = kw.pop('use_gencomp', False)
        self._lambda_gen = kw.pop('lambda_gen', 0.5)
        self._p_target = kw.pop('p_target', 0.0)
        super().__init__(rln, pln, V, use_compose=use_compose, alpha=alpha,
                         use_atoms=use_atoms, use_fractal=use_fractal,
                         use_hopfield=use_hopfield, adaptive_k=adaptive_k,
                         active_query=active_query, **kw)
        # C2: 替换 AQ 为定向采样版 (保留父类构造的槽位/尺寸)
        if self._active_query and self._aq is not None and self._p_target > 0:
            aq = self._aq
            self._aq = AQ4C(V, seed=42, n_support=aq.n_support,
                            n_query=aq.n_query, max_combo=aq.max_combo,
                            p_combo3=getattr(aq, 'p_combo3', 0.6),
                            p_target=self._p_target)
        d = rln.d_model
        # A1: cycle type 锚 (5 类, 粗粒度结构监督)
        if self._use_anchor:
            self.anchor_head = nn.Sequential(
                nn.Linear(d, d), nn.Tanh(), nn.Linear(d, CYCLET))
            base = list(self.outer_opt.param_groups[0]['params'])
            self.outer_opt = torch.optim.Adam(
                base + list(self.anchor_head.parameters()), lr=self.outer_lr)
        # B1: 生成元右乘复合算子 (one-hot 24+6 -> 64 -> 24)
        if self._use_gencomp:
            self.gen_comp = nn.Sequential(
                nn.Linear(30, 64), nn.Tanh(), nn.Linear(64, 24))
            base = list(self.outer_opt.param_groups[0]['params'])
            self.outer_opt = torch.optim.Adam(
                base + list(self.gen_comp.parameters()), lr=self.outer_lr)
            self.gencomp_acc = []

    # A1: 锚损失
    def _anchor_loss(self, m, t):
        cyc = cycle_type(t['perm'])
        logits = self.anchor_head(m)
        return F.cross_entropy(
            logits.unsqueeze(0), torch.tensor([cyc], device=m.device))

    # B1: 回读->逐步复合->目标 闭环损失 (梯度经 gen_comp+perm_head 回流 m)
    def _gencomp_loss(self, m, t):
        gens = t.get('gens')
        if not gens:
            return None
        target = PERM_IDX[t['perm']]
        p_soft = self.perm_head(m)          # (24,) soft 回读
        for gi in gens:
            x = torch.cat([p_soft, _onehot6(gi, m.device)])
            p_soft = self.gen_comp(x)       # (24,) 复合后分布
        loss = F.cross_entropy(
            p_soft.unsqueeze(0), torch.tensor([target], device=m.device))
        with torch.no_grad():
            self.gencomp_acc.append(
                1.0 if p_soft.argmax().item() == target else 0.0)
        return loss

    def _post_define_aux_loss(self, m, t):
        extra = super()._post_define_aux_loss(m, t)
        if m is None:
            return extra
        if self._use_anchor and t.get('perm') is not None:
            al = self._anchor_loss(m, t) * self._lambda_anchor
            extra = al if extra is None else extra + al
        if self._use_gencomp and t.get('gens'):
            gl = self._gencomp_loss(m, t)
            if gl is not None:
                gl = gl * self._lambda_gen
                extra = gl if extra is None else extra + gl
        return extra

    # 复制 V35_Learner.meta_step + 给 AQ 任务补 gens 序列
    def meta_step(self, tasks, train_rules, K=None):
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        if self._active_query and self._aq is not None:
            n_aq = self._aq_n
            qs = self._aq.generate(self.bank, n=n_aq)
            for i, q in enumerate(qs):
                t = {'support': [(c, tg[:4]) for (c, tg) in q['support']],
                     'query': [(c, tg[:4]) for (c, tg) in q['query']],
                     'perm': q['perm'], 'gen_sig': q['gen_sig'],
                     'seen': False, 'name': 'AQ%d' % len(self._aq.gen_history),
                     'struct': 'aq', 'gens': q.get('gens')}
                tasks['__aq__%d' % i] = t
                chosen.append('__aq__%d' % i)
        loss = super().meta_step(tasks, chosen, K=K)
        for k in list(tasks):
            if k.startswith('__aq__'):
                del tasks[k]
        return loss


# ============================================================
# run_cell_v14: 复制 run_cell + 新参数 + D 公共评估
# ============================================================
def run_cell_v14(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
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
    learner = V35_Learner_A(
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
        p_target=cfg.get('p_target', 0.0))
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
    # D 公共评估: probe 回读 perm 按 cycle type 分桶
    pa_struct = {'c3': [], 'c4': [], 'dbl': []}
    if getattr(learner, 'probe_head', None) is not None:
        with torch.no_grad():
            for n in unseen_names:
                task = tasks[n]
                m = learner._define(task, detach=True)
                pred = learner.probe_head(m).view(4, 4).argmax(dim=-1)
                ok = 1.0 if tuple(pred.tolist()) == task['perm'] else 0.0
                s = U_STRUCT[n]
                if s in pa_struct:
                    pa_struct[s].append(ok)
        aux['perm_acc_by_struct'] = {
            k: (round(sum(v) / len(v), 3) if v else None)
            for k, v in pa_struct.items()}
    # AQ 评估 (aq_eval)
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
        aux['compose_params'] = sum(p.numel()
                                    for p in learner.compose.parameters())
    if learner._adaptive_k:
        kus = learner._k_used
        aux['k_used'] = {'mean': round(sum(kus) / len(kus), 2),
                         'min': min(kus), 'max': max(kus)}
    if learner._aq is not None:
        aux['aq_gen'] = len(learner._aq.gen_history)
        aux['aq_combo_lens'] = [h[2] for h in learner._aq.gen_history[-10:]]
    if getattr(learner, 'think_net', None) is not None:
        tr = learner.think_rounds
        aux['think_rounds'] = {'mean': round(sum(tr) / len(tr), 2),
                               'min': min(tr), 'max': max(tr)} if tr else None
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
    if getattr(learner, 'gen_comp', None) is not None:
        ga = learner.gencomp_acc
        aux['gencomp_acc'] = (round(sum(ga) / len(ga), 3) if ga else None)
        aux['gencomp_n'] = len(ga)
    if getattr(learner, 'anchor_head', None) is not None:
        aux['anchor_params'] = sum(p.numel()
                                   for p in learner.anchor_head.parameters())
    try:
        hs = learner.head.step_size_stats()
        aux['head_beta'] = {k: (round(v, 6) if isinstance(v, float)
                                else v) for k, v in hs.items()}
    except Exception:
        pass
    n_params = (sum(p.numel() for p in rln.parameters())
                + sum(p.numel() for p in pln.parameters()))
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'world': cfg.get('world', 'new'),
           'factors': {'LM': lm, 'alpha': cfg.get('alpha', 1.0),
                       'use_compose': use_compose,
                       'use_hopfield': cfg.get('hopfield', False),
                       'adaptive_k': cfg.get('adaptive_k', False),
                       'active_query': cfg.get('active_query', False),
                       'use_anchor': cfg.get('use_anchor', False),
                       'use_gencomp': cfg.get('use_gencomp', False),
                       'p_target': cfg.get('p_target', 0.0),
                       'aq_n': cfg.get('aq_n', 4),
                       'lambda_probe': cfg.get('lambda_probe', 0.3)},
           'time_s': round(time.time() - t0, 1),
           'n_params': n_params,
           'params_kb': round(n_params * 4 / 1024, 1),
           'seen_acc': {n: st['acc_after'][n] for n in train_names},
           'unseen_acc': {n: st['acc_after'][n] for n in unseen_names},
           'unseen_by_struct': struct_mean,
           'aux': aux}
    u = rec['unseen_acc']; ua = sum(u.values()) / len(u)
    print((u"  [%s] unseen=%.4f struct=%s n_params=%d" %
           (key, ua, struct_mean, n_params)), flush=True)
    return rec


BASE = dict(world='new', lm='compose', alpha=1.0,
            hopfield=True, adaptive_k=True, use_think=True, n_think=6,
            lambda_icl=0.5, lambda_combo=0.5, active_query=True,
            aq_n=6, aq_p3=0.7, lambda_probe=1.0)

CELLS14 = {
    'A0': dict(BASE, label='A0-S0-ref'),
    'A1': dict(BASE, use_anchor=True, lambda_anchor=0.5,
               label='A1-anchor'),
    'B1': dict(BASE, use_perm=True, lambda_perm=1.0, use_gencomp=True,
               lambda_gen=0.5, label='B1-gencomp'),
    'C2': dict(BASE, p_target=0.5, label='C2-c4-targeted'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='A0,A1,B1,C2')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.14 四方向消融 (c4 锁死带): A 锚 / B 复合 / C2 定向 / D 评估")
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
        cfg = CELLS14[key]
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell_v14(key, cfg, s1, tasks, U_STRUCT,
                             meta_iters=args.meta_iters)
        except Exception:
            import traceback
            traceback.print_exc()
            print("!!! %s 失败" % key)
            continue
        res.append(r)
    out = ROOT / 'results' / 'v35_14_ablation_report.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']
        ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s" %
              (r['cell'], ua, r['unseen_by_struct']))
        if 'perm_acc_by_struct' in r['aux']:
            print("   D: perm_acc_by_struct =", r['aux']['perm_acc_by_struct'])
        if 'gencomp_acc' in r['aux']:
            print("   B: gencomp_acc =", r['aux']['gencomp_acc'])


if __name__ == '__main__':
    main()
