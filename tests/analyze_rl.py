#!/usr/bin/env python3
"""IDBD vs Autostep vs value 的真实回路对照分析。

回答: **步长分化到底带不带来收益?**

臂:
  idbd        IDBD, mu=0.05, alpha0=0.2   (我们之前唯一能找到的活点)
  auto        Autostep, mu=1e-2, alpha0=0.1 (论文推荐默认, **未针对本任务调参**)
  auto2       Autostep, mu=0.05, alpha0=0.2 (IDBD 的活点; 看 Autostep 是否不敏感)
  value       手工价值函数 (固定公式排序器) —— 已证等价于随机

判据:
  1) 步长是否分化: alpha_std > 0 且 max/min > 1.5
  2) 分化是否带来收益: 相对 value 的 any_time 提升, Welch t 检验
  3) Autostep 是否免调参: 两档 (mu=1e-2/alpha0=0.1) 与 (mu=0.05/alpha0=0.2)
     的**表现是否接近** —— 接近 = 免调参成立
  4) 可靠标准: std < mean/2
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/work/liuyuanjie/headless"


def t_sf(x, df):
    """t 分布上尾概率 (stdlib 数值积分, 避免依赖 scipy)。"""
    if x <= 0:
        return 0.5 + (0.5 - t_sf(-x, df))
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    n, hi = 20000, x + 60.0
    h = (hi - x) / n
    s = 0.0
    for i in range(n + 1):
        t = x + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        s += w * c * (1 + t * t / df) ** (-(df + 1) / 2)
    return s * h / 3


rows = {}
for f in glob.glob(os.path.join(ROOT, "results/rl_*/lm4_wave_results.json")):
    tag = os.path.basename(os.path.dirname(f))
    m = re.match(r"rl_(.+?)_s(\d+)$", tag)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    try:
        d = json.load(open(f))
    except Exception as e:
        print("  [坏文件]", tag, e)
        continue
    v = d.get("replay")
    if isinstance(v, dict) and v.get("any_time_acc") is not None:
        rows.setdefault(arm, []).append((seed, v))

if not rows:
    sys.exit("没有可用结果")

print("=" * 100)
print("步长自适应在真实回路的效果 (lm4, replay 臂, stream=revisit, 12 轮)")
print("=" * 100)
print("%-8s %-3s %9s %8s %6s %9s %9s | %-24s %s"
      % ("arm", "n", "any_time", "std", "CV%", "replay", "最差遗忘",
         "alpha std / max-min", "algo"))
print("-" * 100)
for arm, vs in sorted(rows.items()):
    at = np.array([v["any_time_acc"] for _, v in vs], float)
    if len(at) < 2:
        print("%-8s %-3d %9.4f %8s %6s %9s %9s" % (arm, len(at), at.mean(),
              "-", "-", "-", "-") + "  (待更多 seed)")
        continue
    rp = np.array([v["final_mean_acc"] for _, v in vs], float)
    wf = np.array([v["worst_case_forget"] for _, v in vs], float)
    st = vs[0][1].get("proposer_stats", {})
    al = st.get("alpha", {})
    cv = 100 * at.std(ddof=1) / at.mean() if len(at) > 1 else float("nan")
    print("%-8s %-3d %9.4f %8.4f %6.1f %9.4f %9.4f | %-24s %s"
          % (arm, len(at), at.mean(), at.std(ddof=1) if len(at) > 1 else 0, cv,
             rp.mean(), wf.mean(),
             "%.4f / %.2fx" % (al.get("std", 0), al.get("ratio_max_min", 0)),
             st.get("algo", "-")))

print()
print("=== 判据 1: 步长是否分化 (alpha_std>0 且 ratio>1.5) ===")
for arm, vs in sorted(rows.items()):
    al = vs[0][1].get("proposer_stats", {}).get("alpha", {})
    s, r = al.get("std", 0), al.get("ratio_max_min", 0)
    print("  %-8s std=%.4f ratio=%.2f  %s" % (arm, s, r,
          "分化 ✓" if (s > 0 and r > 1.5) else ("未分化" if s == 0 else "弱")))

if len(rows) > 1:
    print()
    print("=== 判据 2: 两两 Welch t 检验 (any_time) ===")
    keys = sorted(rows)
    print("  %-26s %9s %7s %9s  %s" % ("对比", "Δmean", "t", "p", "判定"))
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A = np.array([v["any_time_acc"] for _, v in rows[a]], float)
            B = np.array([v["any_time_acc"] for _, v in rows[b]], float)
            if len(A) < 2 or len(B) < 2:
                print("  %-26s %+9.4f %7s %9s  %s"
                      % ("%s - %s" % (a, b), A.mean() - B.mean(), "-", "-",
                         "n<2, 待更多 seed"))
                continue
            se = math.sqrt(A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B))
            t = (A.mean() - B.mean()) / se
            df = se ** 4 / ((A.var(ddof=1) / len(A)) ** 2 / (len(A) - 1)
                            + (B.var(ddof=1) / len(B)) ** 2 / (len(B) - 1))
            p = 2 * t_sf(abs(t), df)
            print("  %-26s %+9.4f %7.2f %9.4f  %s"
                  % ("%s - %s" % (a, b), A.mean() - B.mean(), t, p,
                     "显著" if p < 0.05 else "不显著"))

print()
print("=== 判据 3: Autostep 免调参性 (两档超参的表现差) ===")
if "auto" in rows and "auto2" in rows:
    a1 = np.mean([v["any_time_acc"] for _, v in rows["auto"]])
    a2 = np.mean([v["any_time_acc"] for _, v in rows["auto2"]])
    print("  auto (论文默认 mu=1e-2,a0=0.1) %.4f  vs  auto2 (mu=0.05,a0=0.2) %.4f"
          % (a1, a2))
    print("  差 %.4f -> %s" % (abs(a1 - a2),
          "免调参成立 (两档表现接近) ✓" if abs(a1 - a2) < 0.03
          else "仍对超参敏感 ✗"))
else:
    print("  (需要 auto 与 auto2 两个臂都完成)")

print()
print("=== 可靠标准 (leo: std < mean/2) ===")
for arm, vs in sorted(rows.items()):
    at = np.array([v["any_time_acc"] for _, v in vs], float)
    s = at.std(ddof=1) if len(at) > 1 else 0
    print("  %-8s std=%.4f  mean/2=%.4f  %s"
          % (arm, s, at.mean() / 2, "✓" if s < at.mean() / 2 else "✗"))
