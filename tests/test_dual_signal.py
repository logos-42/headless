#!/usr/bin/env python3
"""验证 δ^pred 与 δ^RL 真的分开了, 且覆盖度门控能挡住规划幻觉。

① **信号隔离** (可判定形式):
   只调 update_prediction()  -> θ_v / alpha / h 必须**逐位不变**
   只调 update_value()       -> W_p 必须**逐位不变**
② **覆盖度门控** (Q8): 未访问 g=1.0 / 访问 10 次 g≈0.30; 未访问候选 trustworthy=False
③ **规划幻觉对策**: 复现实测 case (连训同一候选), 门控把该反事实标为不可信
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.dual_proposer import DualSignalProposer  # noqa: E402

rs = np.random.RandomState(0)
desc = rs.randn(40, 3)
DIM_X = 6


def fresh(seed=1):
    return DualSignalProposer(desc, dim_x=DIM_X, k=3, seed=seed)


print("=" * 92)
print("① 信号隔离 —— 判据: 另一条通路的参数必须逐位不变")
print("=" * 92)

# --- 只跑预测通路 ---
p = fresh()
th0, al0, h0 = p.theta.copy(), p.alpha.copy(), p.h.copy()
W0 = p.W_p.copy()
x_target = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.0])
for _ in range(25):
    idx = p.act()
    p.update_prediction(idx, x_target)
print("  只调 update_prediction() x25:")
print("    W_p  最大变化  = %.6f   %s" % (np.abs(p.W_p - W0).max(),
      "已更新 ✓" if np.abs(p.W_p - W0).max() > 1e-9 else "**没更新 ✗**"))
print("    θ_v  最大变化  = %.12f   %s" % (np.abs(p.theta - th0).max(),
      "逐位不变 ✓" if np.abs(p.theta - th0).max() == 0 else "**被污染 ✗**"))
print("    α    最大变化  = %.12f   %s" % (np.abs(p.alpha - al0).max(),
      "逐位不变 ✓" if np.abs(p.alpha - al0).max() == 0 else "**被污染 ✗**"))
print("    h    最大变化  = %.12f   %s" % (np.abs(p.h - h0).max(),
      "逐位不变 ✓" if np.abs(p.h - h0).max() == 0 else "**被污染 ✗**"))

# --- 只跑价值通路 ---
q = fresh()
Wb = q.W_p.copy()
thb = q.theta.copy()
for t in range(25):
    idx = q.act()
    q.update_value(idx, r=0.01)
print("  只调 update_value() x25:")
print("    θ_v  最大变化  = %.6f   %s" % (np.abs(q.theta - thb).max(),
      "已更新 ✓" if np.abs(q.theta - thb).max() > 1e-9 else "**没更新 ✗**"))
print("    W_p  最大变化  = %.12f   %s" % (np.abs(q.W_p - Wb).max(),
      "逐位不变 ✓" if np.abs(q.W_p - Wb).max() == 0 else "**被污染 ✗**"))

print()
print("=" * 92)
print("② 覆盖度门控 (Q8)")
print("=" * 92)
r = fresh()
print("  访问 0 次   gate = %.3f" % r.gate(0))
r.visits[0] = 10
print("  访问 10 次  gate = %.3f  (预测贡献被压到 %.0f%%)" % (r.gate(0), 100 * r.gate(0)))
print("  counterfactual(已访问):     trustworthy=%s" % r.counterfactual(0)[1])
print("  counterfactual(从未访问):   trustworthy=%s" % r.counterfactual(1)[1])
print("  -> %s" % ("未访问候选被正确标为不可信 ✓"
                   if not r.counterfactual(1)[1] else "**未挡住 ✗**"))

print()
print("=" * 92)
print("③ 规划幻觉对策 (复现实测 case: 连训同一候选, 其余零覆盖)")
print("=" * 92)
t = fresh()
TARGET = 2
# 模拟真实回路: 每轮 act() 选候选 -> 喂预测真值 -> 喂价值奖励
for step in range(30):
    # 前 10 步只让 TARGET 有覆盖 (复制实测 [2,2,...] 的零覆盖情形)
    idx = [TARGET] if step < 30 else t.act()
    for i in idx:
        t.visits[i] += 1
    x_next = np.full(DIM_X, 0.5 + 0.01 * min(step, 10))
    t.update_prediction(idx, x_next)
    t.update_value(idx, r=0.01)
print("  连训候选 %d 30 次后, 各候选的反事实可信度:" % TARGET)
for i in range(6):
    xn, trust, g = t.counterfactual(i, min_visits=3)
    print("    候选 %d: 访问=%2d  gate=%.3f  trustworthy=%-5s  预测均值=%.4f"
          % (i, int(t.visits[i]), g, trust, xn.mean()))
n_trust = sum(1 for i in range(6) if t.counterfactual(i, min_visits=3)[1])
print("  -> 只有 %d/6 个候选被标为可信" % n_trust)
print("     实测对照: 转移模型对 [2,2,2,2,2] 预测 0.9582, 真实执行 0.4357。")
print("     规划器若只看预测值就会选它; 门控会把零覆盖的候选标为不可信。")
print()
print("  stats:", t.stats())
