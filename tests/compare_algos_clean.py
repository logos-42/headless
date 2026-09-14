#!/usr/bin/env python3
"""干净对照: 三种算法在**模块本体**上的分化性与累积性。

前一轮我犯了个方法错误: 用**脚本内副本**测 Autostep, 那次默认是 per_component
作用域; 而模块默认已改成 global。两处结论不能混用。这里统一走模块。

判据:
  分化性  alpha max/min (训练结束时)
  累积性  ratio 是否随 n 单调增长  (只看 n=36 -> 1000 的变化倍数)
  安全性  是否有分量掉到地板 / 顶到天花板 / 训练误差发散
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

desc = np.random.RandomState(0).randn(60, 3)


def measure(algo, mu, n, alpha0=0.2, scope="global"):
    p = RLProposer(desc, tau=0.5, k=3, mu=mu, alpha0=alpha0, algo=algo,
                   seed=1, auto_scope=scope)
    for t in range(n):
        idx = p.act()
        r = 0.5 + 0.3 * np.tanh(desc[idx, 0].mean())
        for i in idx:
            p.acc[i] = r
        p.any_time = r
        p.update(idx, r=(r - p.global_acc if t else r))
        p.global_acc = r
    a = p.alpha
    return (a.max() / max(1e-12, a.min()), int((a <= 1e-5).sum()),
            int((a >= 9.99).sum()))


CASES = [
    ("autostep", 0.05, "global"),
    ("autostep", 1.0, "global"),
    ("autostep", 0.05, "per_component"),
    ("idbd", 0.05, "-"),
    ("idbd-acc", 0.05, "-"),
    ("idbd-acc", 0.2, "-"),
]
NS = (36, 100, 300, 1000)

print("=" * 104)
print("三种步长算法在我们任务上的分化性 / 累积性 / 安全性  (模块本体, 60候选, fdim 9)")
print("=" * 104)
print("  %-11s %-6s %-15s %s" % ("algo", "mu", "scope", "".join("%17d" % n for n in NS)))
print("  " + "-" * 98)
for algo, mu, scope in CASES:
    row = []
    for n in NS:
        rt, dead, cap = measure(algo, mu, n, scope=scope if scope != "-" else "global")
        flag = ("d" if dead else " ") + ("c" if cap else " ")
        row.append("%10.2fx%s" % (rt, flag))
    print("  %-11s %-6.3g %-15s %s" % (algo, mu, scope, "".join("%17s" % c for c in row)))

print()
print("  flag: d = 有分量掉到地板 1e-5;  c = 有分量顶到天花板 10")
print("  累积判据: ratio(n=1000) / ratio(n=36) 应 > 1.5")
print()
print("  %-11s %-6s %-15s %10s %10s %8s" % ("algo", "mu", "scope", "r(36)", "r(1000)", "倍数"))
for algo, mu, scope in CASES:
    sc = scope if scope != "-" else "global"
    a, _, _ = measure(algo, mu, 36, scope=sc)
    b, _, _ = measure(algo, mu, 1000, scope=sc)
    print("  %-11s %-6.3g %-15s %9.2fx %9.2fx %7.2f  %s"
          % (algo, mu, scope, a, b, b / max(1e-9, a),
             "累积 ✓" if b / max(1e-9, a) > 1.5 else "不累积"))
