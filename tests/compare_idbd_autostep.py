#!/usr/bin/env python3
"""IDBD vs Autostep 的**免调参性**对比 (用论文自己的 weight-flipping 基准)。

判据 (Degris 2024 §5 + Mahmood 2012):
  IDBD     最优 mu 跨问题漂移数个数量级, 好区间窄且紧邻发散点
           -> 表现: 只在很窄一角有分化, 别处步长死亡
  Autostep 归一化后指数项 unitless 且 |·|<=1, 有效步长 <= 1 (过冲不可能)
           -> 表现: **大范围**内都分化 = 免调参

weight-flipping (Sutton 1992): 20 维输入, 前 15 维目标恒 0, 后 5 维每 20 步翻转。
最优策略 = 给 15 个常权重低步长、5 个翻转权重高步长。合格的算法必须自发区分。
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")

D, N_CONST = 20, 15


def weight_flipping(algo="idbd", mu=0.01, alpha0=0.1, steps=40000,
                    flip_every=20, scale=1.0, seed=0):
    rng = np.random.RandomState(seed)
    w_star = np.zeros(D)
    w_star[N_CONST:] = scale * rng.choice([-1.0, 1.0], size=D - N_CONST)

    w = np.zeros(D)
    alpha = np.full(D, alpha0)
    h = np.zeros(D)
    v = np.zeros(D)
    tau_a, errs, a_hist = 1e4, [], []

    for t in range(steps):
        if t > 0 and t % flip_every == 0:
            j = rng.randint(N_CONST, D)
            w_star[j] = -w_star[j]
        x = rng.randn(D)
        y = float(w_star @ x)
        # IDBD 已知会发散 (论文 §5): 用 clip 保住数值, 但不掩盖发散本身
        w = np.clip(w, -1e4, 1e4)
        delta = float(np.clip(y - float(w @ x), -1e6, 1e6))
        errs.append(min(delta ** 2, 1e12))

        if algo == "autostep":
            g = np.abs(delta * x * h)
            v = np.maximum(g, v + (1.0 / tau_a) * alpha * x ** 2 * (g - v))
            nz = v > 0
            alpha[nz] *= np.exp(np.clip(mu * delta * x[nz] * h[nz] / v[nz], -2, 2))
            alpha = alpha / max(float(np.sum(alpha * x ** 2)), 1.0)
            w = w + alpha * delta * x
            h = h * np.maximum(0.0, 1.0 - alpha * x ** 2) + alpha * delta * x
        else:
            alpha = np.clip(alpha * np.exp(np.clip(mu * delta * x * h, -2, 2)),
                            1e-12, 10.0)
            w = w + alpha * delta * x
            h = h * np.maximum(0.0, 1.0 - alpha * x ** 2) + alpha * delta * x

        if t % 2000 == 0:
            a_hist.append(alpha.copy())

    a_end = np.mean(a_hist[-4:], axis=0)
    ratio = float(a_end[N_CONST:].mean() / max(1e-12, a_end[:N_CONST].mean()))
    return float(np.mean(errs[steps // 2:])), ratio


print("=" * 88)
print("ratio >> 1 = 正确把步长分给翻转权重;  ratio ~ 1 = 步长死亡(无分化)")
print("=" * 88)
print("  %-9s %-18s %-16s %s" % ("algo", "(mu, alpha0)", "MSE(后半程)", "分化比"))
print("  " + "-" * 80)
grid = [(0.001, 0.1), (0.01, 0.1), (0.01, 1e-3), (0.01, 0.01),
        (0.1, 0.1), (0.001, 1e-3), (0.05, 0.01), (0.5, 0.1)]
for algo in ("idbd", "autostep"):
    ok = 0
    for mu, a0 in grid:
        mse, ratio = weight_flipping(algo=algo, mu=mu, alpha0=a0)
        good = ratio > 3
        ok += good
        print("  %-9s %-18s %-16.4f %.2f  %s"
              % (algo, "(%.3g,%.0e)" % (mu, a0), mse, ratio,
                 "✓" if good else "✗ 死亡"))
    print("  -> %s: %d/%d 组分化成功" % (algo, ok, len(grid)))
    print()
