#!/usr/bin/env python3
"""「怎么让步长分化」—— 扫 mu × 更新次数, 找出 Autostep 的可分化区间。

根因 (tests/diag_stepsize.py 实测):
  Autostep 的指数被拦在 mu  =>  n 步后 alpha 最多变 exp(mu*n)
      mu=0.01, n=36 -> exp(0.36) = 1.43x 上限
  IDBD 的指数无界 (clip ±2) =>  单步就能 exp(2) = 7.4x
  => **Autostep 的「不发散保证」本身封住了分化的上限。**

本脚本回答: 给定 n 次更新, 需要多大的 mu 才能让 alpha max/min 达到 3x?
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

ND, D = 60, 3
rs = np.random.RandomState(0)
desc = rs.randn(ND, D)


def run(algo, mu, alpha0, steps, seed=1, scope="global"):
    p = RLProposer(desc, tau=0.5, k=3, mu=mu, alpha0=alpha0, algo=algo,
                   seed=seed, auto_scope=scope)
    for t in range(steps):
        idx = p.act()
        r = 0.5 + 0.3 * np.tanh(desc[idx, 0].mean())
        delta = r - float(np.mean([p.value(i) for i in idx]))
        for i in idx:
            phi = p.phi(i)
            if algo == "autostep":
                g = np.abs(delta * phi * p.h)
                if scope == "global":
                    gm = float(g.max()) if g.size else 0.0
                    p.v_g = max(gm, p.v_g + (1.0 / p.tau_a)
                                * float(np.sum(p.alpha * phi ** 2)) * (gm - p.v_g))
                    den = p.v_g if p.v_g > 0 else 1.0
                    p.alpha *= np.exp(np.clip(p.mu * delta * phi * p.h / den, -2, 2))
                else:
                    p.v = np.maximum(g, p.v + (1.0 / p.tau_a) * p.alpha
                                     * phi ** 2 * (g - p.v))
                    nz = p.v > 0
                    p.alpha[nz] *= np.exp(np.clip(
                        p.mu * delta * phi[nz] * p.h[nz] / p.v[nz], -2, 2))
                p.alpha = p.alpha / max(float(np.sum(p.alpha * phi ** 2)), 1.0)
            else:
                p.alpha = np.clip(p.alpha * np.exp(
                    np.clip(p.mu * delta * phi * p.h, -2, 2)), 1e-6, 1.0)
            p.theta += p.alpha * delta * phi
            p.h = p.h * np.maximum(0.0, 1.0 - p.alpha * phi ** 2) + p.alpha * phi * delta
        p.any_time = r
    a = p.alpha
    return a.std(), a.max() / max(1e-12, a.min()), np.abs(p.h).max() / max(1e-12, np.abs(p.h).min())


print("=" * 96)
print("「怎么让步长分化」—— mu × 更新次数 扫描  (判据: alpha max/min > 3 才算分化)")
print("=" * 96)
for algo, scope in [("autostep", "global"), ("idbd", "-")]:
    print()
    print("--- %s ---" % algo)
    print("  %-8s %s" % ("mu \\ n", "".join("%12d" % n for n in (36, 100, 300, 1000))))
    print("  " + "-" * 62)
    for mu in (0.01, 0.05, 0.2, 0.5, 1.0):
        row = []
        for n in (36, 100, 300, 1000):
            sd, ratio, hrat = run(algo, mu, 0.2, n, scope=scope)
            row.append("%8.2fx%s" % (ratio, "✓" if ratio > 3 else " "))
        print("  %-8.3g %s" % (mu, "".join("%12s" % c for c in row)))
print()
print("注: 只有描述子第 0 维有预测力, 其余 8 维是噪声 —— 分化 = 正确把步长分给第 0 维。")
print("    idbd 不适用 'scope' (无归一化)。")
