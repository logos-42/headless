#!/usr/bin/env python3
"""新 benchmark 汇总: 任务流 + 调度臂, 报边缘设备关心的指标。

指标:
  replay            最终平均准确率
  any_time          在线全程平均 (边学边可用)
  worst_forget      最差遗忘界 (任一旧任务的最大跌幅; 安全功能不能崩)
  mean_forget       平均遗忘
"""
import glob
import json
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"
SEEDS = ["42", "1", "7"]


def load(pat):
    out = {}
    for s in SEEDS:
        p = f"{R}/{pat.format(s=s)}/lm4_wave_results.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        r = d.get("replay") or {}
        out[s] = {
            "replay": r.get("final_mean_acc"),
            "any_time": r.get("any_time_acc"),
            "worst": r.get("worst_case_forget"),
            "mean_f": r.get("mean_forget_all"),
            "strap": (r.get("proposer_stats") or {}).get("value_cv"),
        }
    return out


ARMS = [
    ("启发式 fixed 域序", "bm_bins_fixed_s{s}"),
    ("流: perm (随机排列)", "bm_bins_perm_s{s}"),
    ("流: revisit (反复)", "bm_bins_revisit_s{s}"),
    ("流: nonstationary", "bm_bins_nonstationary_s{s}"),
    ("★ 价值函数 value", "bm_value_s{s}"),
    ("  消融 value-nofb", "bm_value-nofb_s{s}"),
    ("  均匀随机 random", "bm_random_s{s}"),
    ("★ 频率对齐 random-matched", "bm_random-matched_s{s}"),
]

print("=" * 100)
print("LM4 新 benchmark · 任务流 + 调度臂 (12 轮 × 3 seed, 6 域, replay)")
print("=" * 100)
print()
print("  %-26s %-22s %-22s %-16s %s" %
      ("臂", "replay mean±std", "any-time mean±std", "最差遗忘界", "n"))
print("  " + "-" * 92)
rows = {}
for name, pat in ARMS:
    d = load(pat)
    if not d:
        print("  %-26s (无结果)" % name)
        continue
    rep = [v["replay"] for v in d.values() if v["replay"] is not None]
    ant = [v["any_time"] for v in d.values() if v["any_time"] is not None]
    wst = [v["worst"] for v in d.values() if v["worst"] is not None]
    cv = 100 * np.std(rep) / max(1e-9, abs(np.mean(rep))) if rep else float("nan")
    rows[name] = (np.mean(rep) if rep else np.nan, np.std(rep) if rep else np.nan, len(rep))
    print("  %-26s %.4f±%.4f (CV%4.1f%%)  %-22s %-16s %d" %
          (name,
           np.mean(rep) if rep else float("nan"),
           np.std(rep) if rep else float("nan"), cv,
           ("%.4f±%.4f" % (np.mean(ant), np.std(ant))) if ant else "-",
           ("%.4f" % np.mean(wst)) if wst else "-",
           len(rep)))

print()
print("  ── 关键对比 ──")
def cmp(a, b):
    if a not in rows or b not in rows:
        return "  (数据不全)"
    ma, sa, na = rows[a]
    mb, sb, nb = rows[b]
    import math
    se = math.sqrt(sa**2 / max(1, na) + sb**2 / max(1, nb))
    t = (ma - mb) / se if se > 0 else float("nan")
    v = "显著" if abs(t) >= 2 else "**不显著**"
    return "  %-26s vs %-26s Δ=%+.4f  SE=%.4f  t=%+.2f  -> %s" % (a, b, ma - mb, se, t, v)

for a, b in [("★ 价值函数 value", "★ 频率对齐 random-matched"),
             ("★ 价值函数 value", "  均匀随机 random"),
             ("★ 价值函数 value", "  消融 value-nofb"),
             ("★ 价值函数 value", "流: perm (随机排列)"),
             ("启发式 fixed 域序", "流: perm (随机排列)")]:
    print(cmp(a, b))
print()
print("  判读要点: value vs random-matched 才是「价值函数是否更聪明」的干净对照;")
print("           value vs random 混淆了「更聪明」与「复习更均匀」。")
