#!/usr/bin/env python3
"""步长为什么不分化 —— 逐分量诊断。

用户问题: 「步长为什么不分化, 要开始找方法分化, 增加丰富的内部知识」

假设 (待证):
  IDBD       指数 = μ·δ·x_i·h_i          量级 ∝ |h_i|
  Autostep   指数 = μ·δ·x_i·h_i / v_i    v_i = 逐分量 running-max|δ·x_i·h_i|
             => |指数| <= μ (所有分量同一个上界), 只剩**符号**差异
             => h_i 携带的量级信息被自己的归一化抹掉 => ratio -> 1

诊断要看四件事:
  A. |h_i| 各分量是否有分化?        (若有而 alpha 不分化 -> 归一化是元凶)
  B. 指数的**量级**分布 (IDBD vs Autostep)
  C. 每个 alpha_i 的轨迹 (是否在动)
  D. 更新次数是否足够 (Autostep 只剩符号 -> 需要更多步才能累积)

若 A 有分化但 C 没有 -> 确认是归一化抹平, 而不是「特征本身不可分」。
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

ND, D = 60, 3                      # 60 个候选, 描述子 3 维 (同 lm4 的物理描述子)
rs = np.random.RandomState(0)
desc = rs.randn(ND, D)

# 造一个**有真实结构**的环境: 只有「描述子第 0 维」决定收益, 其余是噪声。
# 这样才能检验算法能否把步长分给真正有预测力的方向。
def env_reward(idx):
    return 0.5 + 0.3 * np.tanh(desc[idx, 0].mean())


def run(algo, mu, alpha0, steps=36, seed=1, scope="global"):
    p = RLProposer(desc, tau=0.5, k=3, mu=mu, alpha0=alpha0, algo=algo,
                   seed=seed, auto_scope=scope)
    exps = []                       # 每步的指数向量
    hs, als = [], []
    for t in range(steps):
        idx = p.act()
        r = env_reward(idx)
        # 复制 update() 的核心, 但额外记录指数
        delta = r - float(np.mean([p.value(i) for i in idx]))
        for i in idx:
            phi = p.phi(i)
            g = np.abs(delta * phi * p.h)
            if algo == "autostep":
                e = np.zeros(p.fdim)
                if p.auto_scope == "global":
                    gmax = float(g.max()) if g.size else 0.0
                    p.v_g = max(gmax, p.v_g + (1.0 / p.tau_a)
                                * float(np.sum(p.alpha * phi ** 2)) * (gmax - p.v_g))
                    den = p.v_g if p.v_g > 0 else 1.0
                    e = p.mu * delta * phi * p.h / den
                    p.alpha *= np.exp(np.clip(e, -2, 2))
                else:
                    p.v = np.maximum(g, p.v + (1.0 / p.tau_a) * p.alpha
                                     * phi ** 2 * (g - p.v))
                    nz = p.v > 0
                    e[nz] = p.mu * delta * phi[nz] * p.h[nz] / p.v[nz]
                    p.alpha[nz] *= np.exp(np.clip(e[nz], -2, 2))
                M = max(float(np.sum(p.alpha * phi ** 2)), 1.0)
                p.alpha = p.alpha / M
            else:
                e = np.clip(p.mu * delta * phi * p.h, -2, 2)
                p.alpha = np.clip(p.alpha * np.exp(e), 1e-6, 1.0)
            p.theta += p.alpha * delta * phi
            p.h = p.h * np.maximum(0.0, 1.0 - p.alpha * phi ** 2) + p.alpha * phi * delta
            p.explore[i] += r
            p.explore_n[i] += 1
            exps.append(np.abs(e))
        p.any_time = r
        hs.append(p.h.copy())
        als.append(p.alpha.copy())
    return p, np.array(exps), np.array(hs), np.array(als)


print("=" * 92)
print("步长分化诊断 (60 候选 / fdim=%d / 36 次更新, 只有描述子第0维有预测力)"
      % RLProposer(desc).fdim)
print("=" * 92)

CASES = [("idbd", 0.05, 0.2, "-"),
         ("autostep", 0.01, 0.1, "per_component"),
         ("autostep", 0.01, 0.1, "global"),
         ("autostep", 0.05, 0.2, "per_component"),
         ("autostep", 0.05, 0.2, "global")]
for algo, mu, a0, sc in CASES:
    p, exps, hs, als = run(algo, mu, a0, scope=sc)
    h_final = np.abs(hs[-1])
    a_final = als[-1]
    print()
    print("--- %s  (mu=%.3g, alpha0=%.2g, scope=%s) ---" % (algo, mu, a0, sc))
    print("  A. |h_i| 逐分量 (最终):  %s" % np.round(h_final, 5))
    print("     |h| 分化度: max=%.5f  min=%.5f  std=%.5f  max/min=%s"
          % (h_final.max(), h_final.min(), h_final.std(),
             "%.2f" % (h_final.max() / max(1e-12, h_final.min()))))
    print("  B. 指数|e| 的量级:  mean=%.3e  max=%.3e  上界受限=%s"
          % (exps.mean(), exps.max(), "是 (<=mu)" if exps.max() <= mu * 1.5 else "否"))
    print("  C. alpha 轨迹: 首 %s" % np.round(als[0], 5))
    print("             末 %s" % np.round(a_final, 5))
    print("     alpha 变化倍数: %s" % np.round(a_final / als[0], 4))
    print("     alpha std=%.5f  max/min=%.2f" % (a_final.std(),
          a_final.max() / max(1e-12, a_final.min())))
    # D: 关键判据
    h_spread = h_final.std() / max(1e-12, h_final.mean())
    a_spread = a_final.std() / max(1e-12, a_final.mean())
    print("  D. **h 相对离散度=%.3f  vs  alpha 相对离散度=%.3f**" % (h_spread, a_spread))
    if h_spread > 0.3 and a_spread < 0.1:
        print("     => h 有分化但 alpha 没有 -> **归一化把量级信息抹平了** ✓ 假设成立")
    elif h_spread < 0.3:
        print("     => h 本身没分化 -> 特征/环境不提供区分信号 (需更丰富的内部知识)")
    else:
        print("     => alpha 也分化了")
