"""V35.18 L3: 模型自主结构提议 (E-V 闭环内部化) — 2026-08-15 用户发起

用户洞察: 迁移成立 ⟹ 表示层已有规则抽象 (潜伏), 但未变成架构级能力
(L2 证据: c5 新结构零样本 +5.5pp)。L3 = 把外部课程器 (AQ 随机/定向采样)
内部化为模型自主提议:

  E (提议) : perm_head(m) 温度采样 → 模型输出"下一个结构假设" p̂
  V (验证) : 世界执行器 gen_task_data_split(remap=p̂) 生成真值任务
  吸收     : 任务注入 meta_step (带 perm 标签, 同 AQ 通道)

与 C2 (外部定向) 对照: C2 是外部代码选结构, L3 是模型表示选结构。
测: 模型自主提议的结构能否带来 ≥ 外部定向的学习效果。

Cell:
  P0 = BASE (A0 同配置, 同进程参照)
  P1 = BASE + perm_head(λ_perm 1.0) + n_prop=2 (模型提议)
  P2 = BASE + perm_head + n_prop=2 + p_target 0.5 (提议+定向叠加)
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
from run_v35_fractal import V35_Learner
from run_v35_16a_ablation import (
    V35_Learner_A, PERMS24, C4_PERMS, AQ4C, CELLS14)


class V35_Learner_P(V35_Learner_A):
    """+ L3 提议器: perm_head(m) 温度采样提议新结构 → 世界验证 → 注入训练流."""

    def __init__(self, rln, pln, V, use_compose=False, alpha=1.0,
                 use_atoms=True, use_fractal=False, use_hopfield=False,
                 adaptive_k=False, active_query=False, **kw):
        self._n_prop = kw.pop('n_prop', 0)
        self._prop_tau = kw.pop('prop_tau', 2.0)
        super().__init__(rln, pln, V, use_compose=use_compose, alpha=alpha,
                         use_atoms=use_atoms, use_fractal=use_fractal,
                         use_hopfield=use_hopfield, adaptive_k=adaptive_k,
                         active_query=active_query, **kw)
        self.prop_history = []
        self.prop_struct = []

    def _propose(self, tasks, train_names):
        """E: perm_head(m) 温度采样 → 候选 perm (去重) → V: 世界真值任务."""
        if self._n_prop <= 0 or self.perm_head is None:
            return []
        rng = random.Random(self.seed + self._meta_iters)
        src = rng.sample(train_names, min(2, len(train_names)))
        out = []
        for n in src:
            t = tasks[n]
            with torch.no_grad():
                m = self._define(t, detach=True)
                logits = self.perm_head(m) / self._prop_tau
                idx = torch.multinomial(logits.softmax(-1), 1).item()
            p = PERMS24[idx]
            if p in self.prop_history:
                continue
            self.prop_history.append(p)
            self.prop_struct.append(cycle_type(p))
            seed = rng.randint(0, 10 ** 6)
            sup, qry = gen_task_data_split(
                None, self.V, n_support=4, n_query=8, seed=seed,
                remap=p, stop_at_halt=True)
            sig = generator_signature(p)
            out.append({'perm': p, 'support': sup, 'query': qry,
                        'gen_sig': sig[0] if sig else None})
        return out

    def meta_step(self, tasks, train_rules, K=None):
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        # AQ 注入 (含 C2 定向)
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
        # L3 提议注入 (模型自主)
        if self._n_prop > 0:
            props = self._propose(tasks, train_rules)
            for i, q in enumerate(props):
                t = {'support': [(c, tg[:4]) for (c, tg) in q['support']],
                     'query': [(c, tg[:4]) for (c, tg) in q['query']],
                     'perm': q['perm'], 'gen_sig': q['gen_sig'],
                     'seen': False, 'name': 'PROP%d' % i, 'struct': 'prop'}
                tasks['__prop__%d' % i] = t
                chosen.append('__prop__%d' % i)
        loss = super().meta_step(tasks, chosen, K=K)
        for k in list(tasks):
            if k.startswith('__aq__') or k.startswith('__prop__'):
                del tasks[k]
        return loss


def run_cell_p(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
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
    learner = V35_Learner_P(
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
        prop_tau=cfg.get('prop_tau', 2.0))
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
    if getattr(learner, 'perm_head', None) is not None:
        pa = learner.perm_acc
        aux['perm_acc'] = {
            'seen': (round(sum(pa['seen']) / len(pa['seen']), 3)
                     if pa['seen'] else None),
            'unseen': (round(sum(pa['unseen']) / len(pa['unseen']), 3)
                       if pa['unseen'] else None),
            'n_seen': len(pa['seen']), 'n_unseen': len(pa['unseen'])}
    if learner._aq is not None:
        aux['aq_gen'] = len(learner._aq.gen_history)
    if getattr(learner, 'prop_history', None):
        ph = learner.prop_history
        aux['prop_n'] = len(ph)
        aux['prop_unique'] = len(set(ph))
        from collections import Counter
        aux['prop_struct'] = dict(Counter(learner.prop_struct))
        # 提议去重率
        aux['prop_dup_rate'] = round(
            (sum(len(ph) - len(set(ph)) for _ in [0]) / max(len(ph), 1)), 3)
    try:
        hs = learner.head.step_size_stats()
        aux['head_beta'] = {k: (round(v, 6) if isinstance(v, float)
                                else v) for k, v in hs.items()}
    except Exception:
        pass
    n_params = (sum(p.numel() for p in rln.parameters())
                + sum(p.numel() for p in pln.parameters()))
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'factors': {'use_perm': cfg.get('use_perm', False),
                       'n_prop': cfg.get('n_prop', 0),
                       'p_target': cfg.get('p_target', 0.0),
                       'aq_n': cfg.get('aq_n', 4)},
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

CELLS18 = {
    'P0': dict(BASE, label='P0-ref'),
    'P1': dict(BASE, use_perm=True, lambda_perm=1.0, n_prop=2,
               label='P1-proposer'),
    'P2': dict(BASE, use_perm=True, lambda_perm=1.0, n_prop=2,
               p_target=0.5, label='P2-proposer+targeted'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='P0,P1,P2')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.18 L3: 模型自主结构提议 (E-V 闭环内部化)")
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
        cfg = CELLS18[key]
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell_p(key, cfg, s1, tasks, U_STRUCT,
                           meta_iters=args.meta_iters)
        except Exception:
            import traceback; traceback.print_exc()
            print("!!! %s 失败" % key)
            continue
        res.append(r)
    out = ROOT / 'results' / 'v35_18_proposer.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s" % (r['cell'], ua,
                                            r['unseen_by_struct']))
        if 'prop_n' in r['aux']:
            print("   proposer: n=%d unique=%d struct=%s dup=%.3f" % (
                r['aux']['prop_n'], r['aux']['prop_unique'],
                r['aux']['prop_struct'], r['aux']['prop_dup_rate']))


if __name__ == '__main__':
    main()
