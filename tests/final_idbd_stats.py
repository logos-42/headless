#!/usr/bin/env python3
"""IDBD vs 手工 value 的最终联合分析 (最差遗忘界 = 主判据)。"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/work/liuyuanjie/headless"


def tsf(x, d):
    if x <= 0:
        return 0.5 + (0.5 - tsf(-x, d))
    c = math.gamma((d + 1) / 2) / (math.sqrt(d * math.pi) * math.gamma(d / 2))
    n, hi = 12000, x + 60.0
    h = (hi - x) / n
    s = 0.0
    for i in range(n + 1):
        tt = x + i * h
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        s += w * c * (1 + tt * tt / d) ** (-(d + 1) / 2)
    return s * h / 3


def welch(m1, s1, n1, m2, s2, n2, lab):
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    t = (m1 - m2) / se
    df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * tsf(abs(t), df)
    print("  %-34s Δ=%+.4f t=%6.2f df=%4.1f p=%.4f  %s"
          % (lab, m1 - m2, t, df, p, "**显著**" if p < 0.05 else "不显著"))
    return p


def get(pats):
    at, wf = [], []
    for pat in pats:
        for f in glob.glob(os.path.join(ROOT, "results", pat, "lm4_wave_results.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            v = d.get("replay")
            if isinstance(v, dict) and v.get("any_time_acc") is not None:
                at.append(v["any_time_acc"])
                wf.append(v["worst_case_forget"])
    return np.array(at), np.array(wf)


S8 = (11, 12, 13, 14, 15, 16)
RLX = (1, 2, 3, 4, 5, 6)
A, WA = get(["s8_idbd_s%d" % s for s in S8])
B, WB = get(["s8_value_s%d" % s for s in S8])
C, WC = get(["rlx_idbd_s%d" % s for s in RLX])
D, WD = get(["rlx_value_s%d" % s for s in RLX])

print("=" * 96)
print("IDBD vs 手工 value —— 主判据: 最差遗忘界 (越小越好)")
print("=" * 96)
print()
print("--- 新一批 s8 (同代码版本, 各 n=%d) ---" % len(A))
print("  idbd-raw  any_time %.4f±%.4f   最差遗忘 %.4f±%.4f"
      % (A.mean(), A.std(ddof=1) if len(A) > 1 else 0,
         WA.mean(), WA.std(ddof=1) if len(WA) > 1 else 0))
print("  value     any_time %.4f±%.4f   最差遗忘 %.4f±%.4f"
      % (B.mean(), B.std(ddof=1) if len(B) > 1 else 0,
         WB.mean(), WB.std(ddof=1) if len(WB) > 1 else 0))
welch(A.mean(), A.std(ddof=1), len(A), B.mean(), B.std(ddof=1), len(B),
   "any_time: idbd-raw vs value")
p1 = welch(WA.mean(), WA.std(ddof=1), len(A), WB.mean(), WB.std(ddof=1), len(B),
        "最差遗忘: idbd-raw vs value")
print()

A2 = np.concatenate([A, C])
WA2 = np.concatenate([WA, WC])
B2 = np.concatenate([B, D])
WB2 = np.concatenate([WB, WD])
print("--- 合并两批 s8 + rlx (各 n=%d) ---" % len(A2))
print("  IDBD 系  any_time %.4f±%.4f   最差遗忘 %.4f±%.4f"
      % (A2.mean(), A2.std(ddof=1), WA2.mean(), WA2.std(ddof=1)))
print("  value 系 any_time %.4f±%.4f   最差遗忘 %.4f±%.4f"
      % (B2.mean(), B2.std(ddof=1), WB2.mean(), WB2.std(ddof=1)))
p_a = welch(A2.mean(), A2.std(ddof=1), len(A2), B2.mean(), B2.std(ddof=1), len(B2),
         "any_time: IDBD系 vs value系")
p2 = welch(WA2.mean(), WA2.std(ddof=1), len(A2), WB2.mean(), WB2.std(ddof=1), len(B2),
        "最差遗忘: IDBD系 vs value系")
print()
print("=== 判读 ===")
print("  最差遗忘界: 单批 p=%.3f, 合并 p=%.3f" % (p1, p2))
print("  IDBD 系 %.4f vs value 系 %.4f  ->  %+.1f%%"
      % (WA2.mean(), WB2.mean(), 100 * (WA2.mean() / WB2.mean() - 1)))
print("  any_time : 合并 p=%.3f" % p_a)
