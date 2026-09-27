#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_rr_sentinel.py — **剂量响应 + 内部哨兵**(配方 §2 / Pitfall 34)。

## 为什么必须有这一个

一个"加了机制反而更差"的负结果有两种可能读法, 在真实任务上**无法区分**:
  (a) 机制的**执行**有代价(它真的有害), 或
  (b) 它被发现时所依赖的结构是**陈旧**的, 或 harness 自己坏了。

区分方法: 扫"机制被允许动作的程度" `opt_prob ∈ {0, 0.25, 0.5, 1.0}`, 看**伤害
是否随剂量增长**:

    伤害随剂量增长  -> 定位到机制的**执行**
    伤害对剂量平坦  -> 定位到别处(陈旧结构 / harness)

## 内部哨兵(先读这个, 再读任何剂量点)

`opt_prob = 0` 必须**逐位复现**对照臂(`SkillAgent(mode="primitive")`)。

    哨兵不一致 -> 诊断自身的实现错了, 其余剂量点**全部作废**(不解读)。

一个坏掉的诊断看起来和"机制有害"一模一样 —— 所以哨兵不是一个可选检查。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hibs_lnn.skill_mdp import KeyDoorMDP                       # noqa: E402
from hibs_lnn.skill_agent import DEFAULT_REGIMES, SkillAgent    # noqa: E402
from hibs_lnn.rr_agent import RRSkillAgent                      # noqa: E402

EPISODES = 300
SEEDS = 3
DOSES = [0.0, 0.25, 0.5, 1.0]


def run_once(agent, chain, episodes_per, thresh=0.8, window=20):
    """跑一条链, 返回每段的 (T_adapt, 成功率, 成功平均步数, 执行步数占比)。"""
    out = []
    for reg in chain:
        agent.mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        agent._rr_ensure() if isinstance(agent, RRSkillAgent) else None
        succ, steps_ok, t_adapt = [], [], None
        for ep in range(episodes_per):
            ok, nstep = agent.run_episode()
            succ.append(1.0 if ok else 0.0)
            if ok:
                steps_ok.append(int(nstep))
            if t_adapt is None and len(succ) >= window and \
                    float(np.mean(succ[-window:])) >= thresh:
                t_adapt = ep + 1
        out.append({"t_adapt": int(t_adapt if t_adapt is not None else episodes_per),
                    "succ": float(np.mean(succ[-window:])),
                    "steps": float(np.mean(steps_ok)) if steps_ok else np.nan})
    return out


def main():
    chain = [dict(r) for r in DEFAULT_REGIMES] * 2
    print("=" * 88)
    print("剂量响应 + 内部哨兵   |   %d seed   %d 回合/段   链 A B C A B C"
          % (SEEDS, EPISODES))
    print("=" * 88)

    # ── 对照臂(哨兵要对的基准) ─────────────────────────────────────
    ctrl = []
    for si in range(SEEDS):
        mdp = KeyDoorMDP(n_pos=8, horizon=30, **chain[0])
        ctrl.append(run_once(SkillAgent(mdp, mode="primitive", seed=42 + si),
                             chain, EPISODES))
    ctrl_T = np.array([[r["t_adapt"] for r in rows] for rows in ctrl], float)
    ctrl_S = np.array([[r["steps"] for r in rows] for rows in ctrl], float)
    print("\n对照 primitive  T=%s" % " ".join("%.0f±%.0f" % (m, s) for m, s
                                              in zip(ctrl_T.mean(0), ctrl_T.std(0))))
    print("               S=%s" % " ".join("%.1f±%.1f" % (m, s) for m, s
                                          in zip(np.nanmean(ctrl_S, 0),
                                                 np.nanstd(ctrl_S, 0))))

    # ── 剂量扫描 ────────────────────────────────────────────────────
    table = {}
    for dose in DOSES:
        rows_all = []
        for si in range(SEEDS):
            mdp = KeyDoorMDP(n_pos=8, horizon=30, **chain[0])
            ag = RRSkillAgent(mdp, mode="rr", seed=42 + si, opt_prob=dose)
            rows_all.append(run_once(ag, chain, EPISODES))
        T = np.array([[r["t_adapt"] for r in rows] for rows in rows_all], float)
        S = np.array([[r["steps"] for r in rows] for rows in rows_all], float)
        table[dose] = {"T": T, "S": S,
                       "bk": None}
        print("\n剂量 opt_prob=%.2f" % dose)
        print("  T=%s" % " ".join("%.0f±%.0f" % (m, s) for m, s
                                   in zip(T.mean(0), T.std(0))))
        print("  S=%s" % " ".join("%.1f±%.1f" % (m, s) for m, s
                                   in zip(np.nanmean(S, 0), np.nanstd(S, 0))))

    # ── ★ 哨兵检: dose=0 必须复现对照 ───────────────────────────────
    print()
    print("=" * 88)
    z = table[0.0]
    sent_T = bool(np.array_equal(z["T"], ctrl_T))
    # S 可能因 NaN 无法逐位比 -> 退化为 allclose(nan-equal)
    sent_S = bool(np.allclose(z["S"], ctrl_S, equal_nan=True))
    print("★ 哨兵检(dose=0 vs 对照 primitive): T %s   S %s"
          % ("逐位相同 ✓" if sent_T else "**不同 ✗**",
             "逐位相同 ✓" if sent_S else "**不同 ✗**"))
    if not (sent_T and sent_S):
        print("  ⇒ 哨兵不一致: **诊断自身的实现有误**, 其余剂量点全部作废。")
        print("     不要在此状态下解读任何剂量结论。")
        return 1
    print("  ⇒ 哨兵通过, 剂量点可解读。")

    # ── 伤害是否随剂量增长? ─────────────────────────────────────────
    print()
    print("=" * 88)
    print("★ 剂量趋势(相对对照): ΔT 正=更慢, ΔS 正=步数更多(更差)")
    print("=" * 88)
    for dose in DOSES:
        dT = (table[dose]["T"].mean(0) - ctrl_T.mean(0))
        dS = (np.nanmean(table[dose]["S"], 0) - np.nanmean(ctrl_S, 0))
        print("  opt_prob=%.2f  ΔT: %s (|均|%5.1f)   ΔS: %s (|均|%4.2f)"
              % (dose, " ".join("%+6.0f" % x for x in dT), np.abs(dT).mean(),
                 " ".join("%+6.2f" % x for x in dS), np.abs(dS).mean()))
    print()
    print("  读法: 伤害随 opt_prob 单调增长 -> 定位到机制的**执行**;")
    print("        对剂量平坦           -> 定位到**陈旧结构 / harness**, 不在这里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
