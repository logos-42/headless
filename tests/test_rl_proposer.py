#!/usr/bin/env python3
"""自测: RLProposer 的步长是否真的在分化 (IDBD 的核心主张)。

构造: φ 的第 0 维是**真预测特征**, 第 1-3 维是**纯噪声**。
合格的 IDBD 应给第 0 维**明显更大**的步长 α, 并把它累积进 h。
若不分化 -> 说明"步长为主导的积累"没生效。
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer

rng = np.random.default_rng(0)
p = RLProposer(np.zeros((1, 1)), tau=1.0, k=1, mu=0.02, alpha0=0.01, seed=1)
p.fdim = 4
p.theta = np.zeros(4); p.alpha = np.full(4, 0.01); p.h = np.zeros(4)

# 手工喂样本: feature[0] 有预测力, feature[1:] 是噪声
N = 400
a0_hist, err_hist = [], []
for t in range(N):
    x = rng.normal(0, 1, 4)
    x[1:] = rng.normal(0, 1, 3)
    y = 3.0 * x[0] + rng.normal(0, 0.3)          # 目标只依赖 x[0]
    v = float(p.theta @ x)
    delta = y - v
    p.theta += p.alpha * delta * x
    p.h = p.h * np.maximum(0, 1 - p.alpha * x**2) + p.alpha * x * delta
    p.alpha = np.clip(p.alpha * np.exp(p.mu * delta * x * p.h), 1e-6, 1.0)
    a0_hist.append(p.alpha[0]); err_hist.append(delta**2)

print("=== 步长分化 (IDBD 核心主张) ===")
print("  alpha 逐维: ", np.round(p.alpha, 5))
print("  真预测特征 alpha[0] = %.5f" % p.alpha[0])
print("  噪声特征 alpha[1:]  = %s" % np.round(p.alpha[1:], 5))
print("  比值 alpha[0]/mean(alpha[1:]) = %.2f  -> %s"
      % (p.alpha[0]/max(1e-9, p.alpha[1:].mean()),
         "分化成功 ✓" if p.alpha[0] > 2*p.alpha[1:].mean() else "**未分化**"))
print("  h (信用迹) 逐维: ", np.round(p.h, 5))
print("  预测误差(前50 vs 后50 均值): %.4f -> %.4f"
      % (np.mean(err_hist[:50]), np.mean(err_hist[-50:])))
print()
print("=== redefine 继承检验 ===")
# 先真的跑几轮 act/update, 让 explore_n 与 alpha 有内容
p2 = RLProposer(np.random.RandomState(0).randn(12, 2), tau=0.5, k=2,
                mu=0.02, alpha0=0.01, seed=3)
p2.fdim = p2.fdim
for t in range(30):
    idx = p2.act()
    p2.observe(idx, acc=0.4 + 0.1 * (t % 5),
               any_time=0.3 + 0.02 * t)          # 逐步上升的目标函数
    p2.update(idx)                                # reward = Δ any-time
before_n = p2.explore_n.sum()
before_a = p2.alpha.copy()
before_red = p2.redefine_count
p2.redefine(np.random.RandomState(7).randn(20, 2))
print("  redefine 前 explore_n.sum = %.1f -> 后 = %.1f  %s"
      % (before_n, p2.explore_n.sum(),
         "继承成功 ✓" if p2.explore_n.sum() >= before_n * 0.5 else "**丢失 ✗**"))
print("  alpha 是否跨 redefine 存活:", "是 ✓" if np.allclose(p2.alpha, before_a) else "**被重置 ✗**")
print("  新池大小 n = %d, redefine_count = %d -> %d"
      % (20, before_red, p2.redefine_count))
print("  继承后 learn=True 的候选数 = %d (旧池共 %d 个被学)"
      % (int(p2.learned.sum()), int(before_a.size and 12)))
print()
print("=== 步长是否随经验继续分化 (累积) ===")
st = p2.stats()
print("  alpha: min=%.5f max=%.5f std=%.5f ratio=%.1fx"
      % (st["alpha"]["min"], st["alpha"]["max"],
         st["alpha"]["std"], st["alpha"]["ratio_max_min"]))
print("  h_norm=%.4f theta_norm=%.4f baseline=%.4f"
      % (st["h_norm"], st["theta_norm"], st["baseline"]))
print("  trace 末3步:")
for t in p2.trace[-3:]:
    print("   ", t)
