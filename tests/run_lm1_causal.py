#!/usr/bin/env python3
"""LM1 · 因果持续学习对照实验 (2026-09-06)

假设 Hcausal：用因果技术栈改进自主数据提议器，能让模型在**真实反馈驱动**下
更高效地覆盖"原因原子"，从而用更少训练轮次达到更好/更稳的 unseen 泛化。

对照：
  · value   —— 原 S4ValueProposer（启发式: 简约/自洽/覆盖）
  · causal  —— CausalProposer（do-效应/反事实/发现 + 每轮真实 acc 反馈）

关键机制（因果持续学习）：
  每轮 evaluate 后，把 LM1 对 12 个 unseen 排列的**真实 acc** 回传给因果提议器
  （`observe`），使提议器能从"模型实际盲区"中因果归因该补哪些生成元，
  而不是只用代数先验。这是 RL 式 reward 反馈 + 因果 credit assignment 的结合。

用法:
  python3 tests/run_lm1_causal.py --rounds 3 --iters 6 --d-model 64
  （小模型快速验证; 可用 d-model 192 + iters 40 做更充分对比）
"""
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch   # noqa: E402
from run_lm1_production import (   # noqa: E402
    LM1System, s4_eval, V, N_RESP)


class CausalFeedbackLM1(LM1System):
    """LM1 + 因果提议器，每轮用真实 unseen acc 反馈驱动因果持续学习。"""

    def __init__(self, beta_f=0.6, **kw):
        self.beta_f = beta_f
        kw['proposer'] = 'causal'
        super().__init__(**kw)
        # 重建因果提议器：用较大的真实反馈权重（因果持续学习主导决策）
        from hibs_lnn.causal import CausalProposer
        learned = [self.tasks[n]['perm'] for n in self.train_names]
        self.proposer = CausalProposer(learned, k=self.n_prop,
                                       seed=self.seed, beta_f=self.beta_f)
        # 候选 unseen（固定 12 个结构未见排列）
        self.cand_unseen = [n for n in self.tasks
                            if not self.tasks[n]['seen']]

    def _after_evaluate(self, rec):
        # 用真实 acc 反馈给因果提议器（仅当用了因果提议器）
        if not hasattr(self.proposer, 'observe'):
            return rec
        with torch.no_grad():
            accs = s4_eval(self.learner, self.tasks, self.cand_unseen,
                           k_adapt=20)
        for n in self.cand_unseen:
            perm = self.tasks[n]['perm']
            self.proposer.observe(perm, accs.get(n, 0.0))
        rec['causal_covered_atoms'] = len(
            getattr(self.proposer, 'learned_atoms', set()))
        rec['causal_n_obs'] = len(getattr(self.proposer, 'real_obs', {}))
        return rec


def run_one(proposer, rounds, iters, d_model, d_state, n_layers, n_prop,
            beta_f=0.6):
    if proposer == 'causal':
        sys = CausalFeedbackLM1(d_model=d_model, d_state=d_state,
                                n_layers=n_layers, iters_per_round=iters,
                                n_prop=n_prop, seed=42, beta_f=beta_f)
    else:
        sys = LM1System(d_model=d_model, d_state=d_state, n_layers=n_layers,
                        iters_per_round=iters, n_prop=n_prop, seed=42,
                        proposer='value')
    t0 = time.time()
    sys.run(rounds, with_migrate_every=999)   # 关闭 S5 迁移评估以提速
    wall = time.time() - t0
    return sys, wall


def report(name, sys_obj, wall):
    h = sys_obj.history
    unseen = [r.get('unseen_mean', 0) for r in h]
    forget = [r.get('forget', 0) for r in h]
    final = h[-1] if h else {}
    c4 = (final.get('unseen_by_struct') or {}).get('c4')
    print("\n=== %s ===" % name)
    print("  轮数: %d  耗时: %.1fs  (%.1fs/轮)" % (len(h), wall, wall / max(1, len(h))))
    print("  unseen 曲线: %s" % [round(u, 4) for u in unseen])
    print("  final unseen=%.4f  c4=%s  遗忘=%s" % (
        unseen[-1] if unseen else 0,
        '%.4f' % c4 if c4 is not None else 'N/A',
        round(forget[-1], 4) if forget else 'N/A'))
    if 'causal_n_obs' in final:
        print("  (causal) 已覆盖原子=%d/%d  真实反馈样本=%d"
              % (final.get('causal_covered_atoms', 0), 6,
                 final.get('causal_n_obs', 0)))
    return unseen


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--d-state', type=int, default=8)
    ap.add_argument('--n-layers', type=int, default=1)
    ap.add_argument('--n-prop', type=int, default=2)
    ap.add_argument('--proposers', type=str, default='value,causal')
    ap.add_argument('--beta-f', type=float, default=3.0,
                    help='因果提议器的真实反馈权重(越大越以真实acc驱动)')
    args = ap.parse_args()

    print("=" * 70)
    print("LM1 因果持续学习对照实验")
    print("  模型: d=%d/ds=%d/l%d  每轮 iters=%d, n_prop=%d, rounds=%d"
          % (args.d_model, args.d_state, args.n_layers,
             args.iters, args.n_prop, args.rounds))
    print("=" * 70)

    results = {}
    for pname in [p.strip() for p in args.proposers.split(',')]:
        print("\n>>> 训练 [%s] ..." % pname)
        sys_obj, wall = run_one(pname, args.rounds, args.iters,
                                args.d_model, args.d_state, args.n_layers,
                                args.n_prop, beta_f=args.beta_f)
        results[pname] = report(pname.upper(), sys_obj, wall)
        # 释放显存/内存
        del sys_obj
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 汇总对比
    print("\n" + "=" * 70)
    print("汇总: 各配置每轮 unseen")
    for k, v in results.items():
        print("  %-8s %s" % (k, [round(u, 4) for u in v]))
    # 结论（若都跑满则给直观对比）
    vals = {k: v[-1] if v else 0 for k, v in results.items()}
    if 'value' in vals and 'causal' in vals:
        d = vals['causal'] - vals['value']
        arrow = "因果提议器胜出 (+%.4f)" % d if d > 0 else "本轮持平/未胜出 (%.4f)" % d
        print("结论: value final=%.4f / causal final=%.4f → %s"
              % (vals['value'], vals['causal'], arrow))
    print("=" * 70)


if __name__ == "__main__":
    main()