#!/usr/bin/env python3
"""lm4 价值函数调度: fixed / random / value 三臂 × 多 seed 对照。"""
import glob
import json
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"
ARMS = ["fixed", "random", "value"]
data = {a: {} for a in ARMS}
for p in sorted(glob.glob(f"{R}/vp2_*_s*/lm4_wave_results.json")):
    tag = os.path.basename(os.path.dirname(p))
    _, arm, seed = tag.split("_")
    d = json.load(open(p))
    v = d.get("replay") or {}
    data[arm][seed] = {
        "replay": v.get("final_mean_acc"),
        "forget": v.get("mean_forget"),
        "naive": (d.get("naive") or {}).get("final_mean_acc"),
        "naive_forget": (d.get("naive") or {}).get("mean_forget"),
    }

print("=" * 78)
print("LM4 价值函数驱动调度 · 三臂对照 (6 域, replay)")
print("=" * 78)
hdr = "  %-9s %-30s %-24s %s" % ("调度", "replay mean±std (CV)", "naive mean±std (CV)", "n")
print(hdr)
print("  " + "-" * 74)
for a in ARMS:
    rs = [v["replay"] for v in data[a].values() if v["replay"] is not None]
    ns = [v["naive"] for v in data[a].values() if v["naive"] is not None]
    if not rs:
        print("  %-9s (无)" % a)
        continue
    m, s = np.mean(rs), np.std(rs)
    cv = 100 * s / max(1e-9, abs(m))
    tag = "可信" if (m > 0 and s < m / 2) else "不可信"
    if ns:
        mn, sn = np.mean(ns), np.std(ns)
        nstr = "%.4f±%.4f (%4.1f%%)" % (mn, sn, 100 * sn / max(1e-9, abs(mn)))
    else:
        nstr = "-"
    print("  %-9s %.4f±%.4f (%5.1f%% %s)  %-24s %d" % (a, m, s, cv, tag, nstr, len(rs)))

print()
print("  ── 逐 seed (replay) ──")
seeds = sorted({s for a in ARMS for s in data[a]}, key=lambda x: int(x) if x.isdigit() else 0)
print("  %-9s" % "seed" + "".join("%10s" % s for s in seeds))
for a in ARMS:
    row = "".join(("%10.4f" % data[a][s]["replay"]) if s in data[a] and data[a][s]["replay"]
                  else "%10s" % "-" for s in seeds)
    print("  %-9s" % a + row)

print()
print("  ── 参照 ──")
print("  固定顺序基线 (c1_replay_rr2, 单 seed): replay 0.7368")
print("  SSM 原版 (q5_agg, 6 seed):            replay 0.6126±0.0180")
print("  联合训练上限 (6 域):                   joint 0.7948")
if not any(data[a] for a in ARMS):
    print("  (无结果)")
