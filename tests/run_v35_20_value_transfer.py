"""V35.20 — S5 迁移测试: 价值函数训练的表示迁移质量 (2026-08-16)

问题: S4 内不同训练配置的表示, 迁移到 S5 的质量是否不同?
  B (base 无提议) vs C2 (外部定向采样) vs V1cov1 (价值函数自主提议)

协议 (同进程, 复用 V35.17 L2 迁移链路):
  1. S4 训练 (每配置 40 iters)
  2. head 扩维 n_resp 4→5 (前块复制新块零)
  3. S5 零样本迁移评估 (纯表示路径, 按 cycle type 分桶)
  4. S5 微调 20 iters → 再评估

关键问题: 价值函数 (自主提议) 训练的表示迁移是否 ≥ C2 (外部定向)?
即"自主选择学习内容"是否产出更可迁移的表示。
"""
import os, sys, time, json, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import generator_signature
from run_v35_16a_ablation import CELLS14
from run_v35_17_s5_transfer import (
    make_s5_tasks, extend_head, s5_eval, struct_mean5, N_RESP, N_RESP5)
from run_v35_19_value_proposer import V35_Learner_V, CELLS19

V = 200


def train_s4_cfg(cfg, learner_cls, s1_state, meta_iters=40, extra=None,
                 seed=42):
    """S4 训练指定配置, 返回 (rln, pln, learner)."""
    tasks = build_v32_tasks(V, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
    tasks = make_regs_tasks(tasks)
    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    T_STRUCT = build_t_struct(42)
    for n in tasks:
        tasks[n]['name'] = n
        tasks[n]['struct'] = (U_STRUCT.get(n) if not tasks[n]['seen']
                              else T_STRUCT.get(n, '?'))
        sig = generator_signature(tasks[n]['perm'])
        tasks[n]['gen_sig'] = sig[0] if sig else None
        tasks[n]['gen_len'] = sig[1] if sig else 0
    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(V, N_RESP, s1_state)
    lm = cfg.get('lm', 'compose')
    use_compose = (lm == 'compose')
    learner = learner_cls(
        rln, pln, V, K=20, seed=seed, sleep_iters=2, n_resp=N_RESP,
        use_intent=True, use_compose=use_compose,
        use_atoms=(lm != 'base'),
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
    if hasattr(learner, '_sync_learned'):
        learner._sync_learned(tasks, train_names)
    t0 = time.time()
    for it in range(meta_iters):
        learner.meta_iters = it
        learner.meta_step(tasks, train_names)
    print("    [S4 %s] 训练 %.0fs" % (cfg.get('label', '?'),
                                      time.time() - t0), flush=True)
    return rln, pln, learner


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--s4-iters', type=int, default=40)
    ap.add_argument('--finetune-iters', type=int, default=20)
    ap.add_argument('--no-stage1', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--configs', default='B,C2,V1')
    args = ap.parse_args()
    print("=" * 70)
    print("V35.20 — S5 迁移: 价值函数表示 vs C2 vs base")
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
    s5_tasks = make_s5_tasks(seed=42)
    s5_unseen = [n for n in s5_tasks if not s5_tasks[n]['seen']]

    configs = {
        # B: base 无机制 (V34.7 结构)
        'B': (dict(world='new', lm='base', label='B-base'), V35_Learner_V),
        # C2: 外部定向采样
        'C2': (dict(CELLS14['C2'], label='C2-targeted'), V35_Learner_V),
        # V1cov1: 价值函数自主提议 (简约+自洽+覆盖)
        'V1': (dict(CELLS19['V1'], lambda_cov=1.0, label='V1-value+cov'),
               V35_Learner_V),
    }
    keys = [k.strip() for k in args.configs.split(',')]
    for key in keys:
        cfg, cls = configs[key]
        print("\n== S4 训练: %s (seed %d) ==" % (key, args.seed))
        rln, pln, learner = train_s4_cfg(cfg, cls, s1,
                                         meta_iters=args.s4_iters,
                                         seed=args.seed)
        # head 扩维 → S5 零样本
        extend_head(pln, learner)
        acc0 = s5_eval(learner, s5_tasks, s5_unseen)
        res[key + '_zero'] = {
            'unseen': round(sum(acc0.values()) / len(acc0), 4),
            'by_struct': struct_mean5(acc0, s5_tasks)}
        print("    %s zero-shot unseen=%.4f %s" % (
            key, res[key + '_zero']['unseen'],
            res[key + '_zero']['by_struct']))
        # S5 微调 (纯 oml_step 协议, 同 V35.17)
        from run_v35_17_s5_transfer import oml_step as _oml
        s5_seen = [n for n in s5_tasks if s5_tasks[n]['seen']]
        t0 = time.time()
        for it in range(args.finetune_iters):
            learner.meta_iters = 1000 + it
            _oml(learner, s5_tasks, s5_seen, 1000 + it)
        print("    微调 %.0fs" % (time.time() - t0), flush=True)
        acc1 = s5_eval(learner, s5_tasks, s5_unseen)
        res[key + '_ft'] = {
            'unseen': round(sum(acc1.values()) / len(acc1), 4),
            'by_struct': struct_mean5(acc1, s5_tasks)}
        print("    %s finetuned unseen=%.4f %s" % (
            key, res[key + '_ft']['unseen'], res[key + '_ft']['by_struct']))

    print("\n" + "=" * 70)
    print("S5 迁移汇总 (同进程):")
    for k in ['%s_zero' % kk for kk in keys] + ['%s_ft' % kk for kk in keys]:
        r = res[k]
        print("  %-10s unseen=%.4f %s" % (k, r['unseen'], r['by_struct']))
    out = ROOT / 'results' / ('v35_20_value_transfer_s%d.json' % args.seed)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print("JSON:", out)


if __name__ == '__main__':
    main()
