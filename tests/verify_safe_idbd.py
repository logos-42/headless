#!/usr/bin/env python3
"""safe IDBD 验证 (用模块本体, 不是副本)。

对 IDBD 的三个已知问题逐一验证修复是否有效, **且是否保住累积性**:

  ① 发散 (raw IDBD 在 weight-flipping 有 3/8 档 MSE 1.6e9)
     -> 判据: 训练误差有界, 不出现 inf/1e9
  ② alpha 掉地板且回不来 (实测多个分量死在 1e-5)
     -> 判据: 训练后 dead 分量占比, 且它们能否靠 sign 推进复活
  ③ mu 与任务量纲耦合 (论文单位分析: 指数 delta*x*h 量纲 y^2)
     -> 判据: **把 reward 缩 100 倍, 最优 mu 是否基本不变**
        (raw IDBD 上应为 ~100 倍差异)

并且必须确认**累积性没被杀掉**: ratio 应随 n 增长 (Autostep 是 n 无关)。
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

ND, D = 60, 3
rs = np.random.RandomState(0)
desc = rs.randn(ND, D)


def run(algo, mu, alpha0, steps, scale=1.0, seed=1, n_know=0):
    """在 synthetic 环境上跑 n 步; 只有描述子第 0 维有预测力。"""
    p = RLProposer(desc, tau=0.5, k=3, mu=mu, alpha0=alpha0, algo=algo,
                   seed=seed, n_know=n_know)
    errs = []
    for t in range(steps):
        idx = p.act()
        r = scale * (0.5 + 0.3 * np.tanh(desc[idx, 0].mean()))
        # 走模块自己的 update 通路
        for i in idx:
            p.acc[i] = r
        p.any_time = r
        p.update(idx, r=(r - p.global_acc if t else r))
        p.global_acc = r
        errs.append(abs(r - float(np.mean([p.value(i) for i in idx]))))
    a = p.alpha
    return (a.std(), a.max() / max(1e-12, a.min()),
            float(np.mean(errs[-max(1, steps // 4):])), int((a <= 1e-5).sum()),
            float(np.sqrt(p.grms)))


print("=" * 96)
print("safe IDBD 验证 (模块本体, 60 候选 / fdim 9 / 仅描述子第 0 维有预测力)")
print("=" * 96)
print()
print("--- ① 是否仍累积 (ratio 应随 n 增长; Autostep 是 n 无关 = 不累积) ---")
print("  %-8s %s" % ("mu \\ n", "".join("%14d" % n for n in (36, 100, 300, 1000))))
for mu in (0.05, 0.2):
    row = []
    for n in (36, 100, 300, 1000):
        sd, ratio, err, dead, grms = run("idbd", mu, 0.2, n)
        row.append("%9.2fx%s" % (ratio, "✓" if ratio > 3 else " "))
    print("  %-8.3g %s" % (mu, "".join("%14s" % c for c in row)))

print()
print("--- ② 是否不发散 (raw IDBD 有 3/8 档 MSE 1.6e9) ---")
print("  %-8s %12s %12s %10s" % ("mu", "末期误差", "grad_rms", "dead分量"))
for mu in (0.05, 0.2, 1.0, 5.0):
    sd, ratio, err, dead, grms = run("idbd", mu, 0.2, 300)
    print("  %-8.4g %12.4f %12.3e %10d" % (mu, err, grms, dead))

print()
print("--- ③ mu 是否与 reward 量纲解耦 (把 reward 缩 100 倍看最优 mu 是否漂移) ---")
print("  %-8s %10s %12s   %s" % ("scale", "mu", "alpha ratio", "说明"))
for sc in (1.0, 0.01):
    for mu in (0.05, 0.2, 1.0):
        sd, ratio, err, dead, grms = run("idbd", mu, 0.2, 300, scale=sc)
        print("  %-8.3g %10.3g %9.2fx" % (sc, mu, ratio))
    print()

print("--- ④ 内部知识是否扩大可分化维度 ---")
for nk in (0, 14):
    sd, ratio, err, dead, grms = run("idbd", 0.2, 0.2, 300, n_know=nk)
    p_tmp = RLProposer(desc, n_know=nk)
    print("  n_know=%-3d fdim=%-3d alpha std=%.5f  ratio=%8.2fx  dead=%d"
          % (nk, p_tmp.fdim, sd, ratio, dead))
