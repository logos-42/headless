#!/usr/bin/env python3
"""阶段五的核心问题: **多个 GVF 到底能不能产生互补证据?**

规格: `docs/stage5_evidence_bank_spec.md` §5。

做法: 在 KeyDoor 上跑 regime 链(3 个 regime, 各若干回合), 打开 `rr_evidence=True`,
从证据库取**逐 step 的 z 序列**, 用每个 cumulant 的二值化当标签各算一次:

    AUC_i(单条)  AUC_union(至少两条同时超阈)  complementarity = union − 最好单条
    disagreement(至少一条投 bad 且至少一条投 healthy 的步占比)
    corr_err(冗余度)

★ 四个标签都算 —— 只挑一个会偏袒与它同源的那条 GVF。
★ **允许结论是"冗余"**; 那就照实报冗余, 不写成收益。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hibs_lnn.skill_mdp import KeyDoorMDP                    # noqa: E402
from hibs_lnn.rr_agent import RRSkillAgent                   # noqa: E402
from hibs_lnn.evidence_bank import complementarity           # noqa: E402

REGIMES = [(2, 5, 7), (1, 4, 7), (3, 6, 7)]
N_PER = 30
SEEDS = (0, 1, 2)
LABELS = ("prediction", "reward", "transition", "regime")


def run(seed):
    mdp = KeyDoorMDP(n_pos=8, key_pos=2, door_pos=5, goal_pos=7, horizon=30)
    mdp.set_regime(*REGIMES[0])
    ag = RRSkillAgent(mdp, mode="rr", seed=seed, opt_prob=1.0, rr_evidence=True)
    for reg in REGIMES:
        mdp.set_regime(*reg)
        for _ in range(N_PER):
            ag.run_episode()
    return ag


print("=" * 100)
print("阶段五: GVF 证据的互补性度量")
print("=" * 100)

per_label = {L: [] for L in LABELS}
bank_stats = []
for seed in SEEDS:
    ag = run(seed)
    assert ag.rr_bank is not None, "rr_evidence=True 必须建出 bank"
    h = ag.rr_bank.z_history()
    cum = h["cumulant"]
    z = h["z"]
    n = len(next(iter(z.values())))
    bank_stats.append({"seed": seed, "n_steps": n,
                       "n_bad_ever": int(ag.rr_ev_bad_n),
                       "evidence": {k: {"n": v["n"], "z": round(v["z"], 3)}
                                    for k, v in
                                    ag.rr_bank.evidence(step=ag.rr_ev_step,
                                                        regime=ag.rr_regime_idx).items()}})
    print(f"\n--- seed={seed}  步数={n}  bad票累计={ag.rr_ev_bad_n} ---")
    print("  每条证据: " + "  ".join(
        f"{k}(n={v['n']},z={v['z']:.2f})" for k, v in
        ag.rr_bank.evidence(step=ag.rr_ev_step, regime=ag.rr_regime_idx).items()))
    for L in LABELS:
        lab = (cum[L] > 0).astype(int)
        if lab.min() == lab.max():
            print(f"  标签={L:11s} 全为同一值({lab.mean():.3f}), 跳过")
            continue
        r = complementarity(z, lab, z_bad=ag.rr_ev_z_bad)
        r["label"] = L
        r["seed"] = seed
        r["pos_rate"] = float(lab.mean())
        per_label[L].append(r)
        print(f"  标签={L:11s} 正例={lab.mean():.3f} | "
              f"AUC_i=" + ",".join(f"{k}:{v:.3f}" for k, v in r["auc_i"].items()) +
              f" | any={r['auc_any']:.3f} cons={r['auc_consensus']:.3f} best={r['best_single']:.3f} "
              f"comp={r['complementarity']:+.4f} dis={r['disagreement']:.3f}")

print()
print("=" * 100)
print("汇总(3 seed)")
print("=" * 100)
print(f"{'标签':<12}{'单条 AUC(均值)':>44}{'any':>8}{'cons':>8}{'best':>8}{'comp':>9}{'disagree':>10}")
for L in LABELS:
    rs = per_label[L]
    if not rs:
        print(f"{L:<12}{'(全部 seed 标签退化)':>44}")
        continue
    names = list(rs[0]["auc_i"])
    am = {k: float(np.nanmean([r["auc_i"][k] for r in rs])) for k in names}
    u = float(np.nanmean([r["auc_any"] for r in rs]))
    uc = float(np.nanmean([r["auc_consensus"] for r in rs]))
    b = float(np.nanmean([r["best_single"] for r in rs]))
    c = float(np.nanmean([r["complementarity"] for r in rs]))
    d = float(np.nanmean([r["disagreement"] for r in rs]))
    print(f"{L:<12}" + "  ".join(f"{k[:5]}={am[k]:.3f}" for k in names).ljust(44)
          + f"{u:>8.3f}{uc:>8.3f}{b:>8.3f}{c:>+9.4f}{d:>10.3f}")

print()
print("=" * 100)
print("冗余度 (corr_err, 3 seed 平均)")
print("=" * 100)
keys = list(per_label[LABELS[0]][0]["corr_err"]) if per_label[LABELS[0]] else []
for k in keys:
    v = [r["corr_err"][k] for L in LABELS for r in per_label[L] if k in r["corr_err"]]
    print(f"  {k:<22} corr = {np.mean(v):+.3f}")

print()
print("=" * 100)
print("裁定")
print("=" * 100)
allc = [r["complementarity"] for L in LABELS for r in per_label[L]]
allb = [r["best_single"] for L in LABELS for r in per_label[L]]
alla = [v for L in LABELS for r in per_label[L] for v in r["auc_i"].values()]
print(f"  complementarity 范围: {min(allc):+.4f} … {max(allc):+.4f}  (均值 {np.mean(allc):+.4f})")
print(f"  best_single 均值: {np.nanmean(allb):.3f}   union 均值: "
      f"{np.nanmean([r['auc_any'] for L in LABELS for r in per_label[L]]):.3f}")
print(f"  单条 AUC 范围: {np.nanmin(alla):.3f} … {np.nanmax(alla):.3f}")
if np.nanmax(alla) <= 0.5 + 1e-9:
    print("  ⇒ 单条全部 ≤ 0.5: 这组 GVF **连检测都做不到** ⇒ 证据机制不成立")
elif np.mean(allc) <= 0.0:
    print("  ⇒ complementarity ≤ 0: **冗余** —— 当前这组 GVF 不提供互补证据")
else:
    print("  ⇒ complementarity > 0: 存在互补证据的迹象(仍需多 seed 置信区间)")
