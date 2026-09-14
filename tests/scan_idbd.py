#!/usr/bin/env python3
"""扫描: 归一化后 IDBD 的步长分化是否在**真实提议回路**里出现。

标准: alpha_std > 0 且 max/min 比值 > 1.5 => 步长真的在学"谁快谁慢"。
若 alpha_std == 0 => 步长仍是超参, "步长为主导的积累"没成立。
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

rs = np.random.RandomState(0)
desc = rs.randn(20, 2)

print("  mu     alpha0   alpha_std   max/min   h_norm   theta_norm  any_time")
print("  " + "-" * 72)
best = None
for mu in (0.01, 0.05, 0.2, 1.0, 5.0):
    for a0 in (0.01, 0.05, 0.2):
        p = RLProposer(desc, tau=0.5, k=2, mu=mu, alpha0=a0, seed=3)
        for t in range(120):
            idx = p.act()
            # 环境: 部分候选"好"(返回高), 部分差 —— 制造真实的 reward 差异
            acc = 0.5 + 0.4 * np.tanh(desc[idx, 0]).mean()
            p.observe(idx, acc=acc, any_time=0.3 + 0.0015 * t)
            p.update(idx)
        st = p.stats()
        print("  %-6s %-8s %-11.2e %-9.2f %-8.4f %-11.4f %.3f"
              % (mu, a0, st["alpha"]["std"], st["alpha"]["ratio_max_min"],
                 st["h_norm"], st["theta_norm"], st["any_time"]))
        if best is None or st["alpha"]["std"] > best[0]:
            best = (st["alpha"]["std"], mu, a0)
print()
print("  最佳分化: alpha_std=%.2e  (mu=%s, alpha0=%s)" % best)
print("  判据: alpha_std > 0 且 ratio > 1.5 -> %s"
      % ("步长分化成立 ✓" if best[0] > 0 else "**步长仍是常数 ✗**"))
