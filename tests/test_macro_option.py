"""test_macro_option.py — 验证 macro 模式: option 当宏动作执行 + 成败回填。

判据(事前写死):
  1. macro 模式必须让 `total_visits > 0` —— 否则仍是"从不去结算"的死账
  2. rediscover 模式(逐步顾问)应当 `total_visits == 0` —— 证明这个坑确实存在过
  3. macro 不应崩, 且三模式的最终成功率都应为正(学习仍在工作)
"""
import sys
sys.path.insert(0, "/Users/apple/Downloads/headless")

import numpy as np

from hibs_lnn.skill_mdp import KeyDoorMDP
from hibs_lnn.skill_agent import SkillAgent, DEFAULT_REGIMES

CHAIN = DEFAULT_REGIMES * 2
EPS_PER = 120
WINDOW = 20


def run_mode(mode, seed=42):
    mdp = KeyDoorMDP(n_pos=8, horizon=30, **CHAIN[0])
    ag = SkillAgent(mdp, mode=mode, seed=seed)
    out = []
    for ci, reg in enumerate(CHAIN):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        if mode != "primitive":
            ag.maybe_discover(force=True)
        succ, t_adapt = [], None
        for ep in range(EPS_PER):
            ok, _ = ag.run_episode()
            succ.append(1.0 if ok else 0.0)
            if t_adapt is None and len(succ) >= WINDOW:
                if float(np.mean(succ[-WINDOW:])) >= 0.8:
                    t_adapt = ep + 1
        out.append((ci, t_adapt if t_adapt is not None else EPS_PER,
                    round(float(np.mean(succ[-WINDOW:])), 2)))
    return out, ag.option_bookkeeping()


print("%-12s %-5s %-9s %-11s %-14s %s"
      % ("mode", "seg", "T_adapt", "final_succ", "succ/visits", "n_options"))
print("-" * 78)
results = {}
for mode in ("primitive", "rediscover", "macro"):
    rows, bk = run_mode(mode)
    results[mode] = (rows, bk)
    for ci, ta, fs in rows:
        print("%-12s %-5d %-9d %-11.2f %-14s %d"
              % (mode, ci, ta, fs,
                 "%d/%d" % (bk["total_success"], bk["total_visits"]),
                 bk["n_options"]))
    print()

print("=" * 78)
print("判据核对")
print("=" * 78)
b_r = results["rediscover"][1]
b_m = results["macro"][1]
ok1 = b_m["total_visits"] > 0
ok2 = b_r["total_visits"] == 0
print("  ① macro 有结算记录 (visits>0)      : %s  (visits=%d)"
      % ("✓" if ok1 else "✗", b_m["total_visits"]))
print("  ② rediscover 无结算记录 (visits=0) : %s  (visits=%d)"
      % ("✓" if ok2 else "✗", b_r["total_visits"]))
allpos = all(rr[0][-1][2] > 0 for rr in results.values())
print("  ③ 三模式都还在学习 (末段成功率>0)  : %s" % ("✓" if allpos else "✗"))

print()
print("=" * 78)
print("★ 3 段 T_adapt 对比 (前3段=第1轮, 后3段=第2轮同 regime)")
print("=" * 78)
for mode in ("primitive", "rediscover", "macro"):
    arr = np.array([r[1] for r in results[mode][0]], float)
    first, second = arr[:3].mean(), arr[3:].mean()
    print("  %-12s 第1轮 %6.1f  第2轮 %6.1f  Δ=%+.1f  (%.2fx)"
          % (mode, first, second, second - first, first / max(second, 1e-9)))
