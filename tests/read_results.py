#!/usr/bin/env python3
"""读实验结果 — 关键指标对比。"""
import json
import os
import sys

os.chdir("/work/liuyuanjie/headless")
tags = sys.argv[1:] or ["si_d3", "si_d6", "si_d12", "q9_oml2_fixed",
                        "q10_isolate", "q11_replay_pln", "q5_agg_s7", "q7_noconsol",
                        "q8_nometa"]

print("=" * 96)
for t in tags:
    p = f"results/{t}/lm4_wave_results.json"
    if not os.path.exists(p):
        print(f"{t:20s} 跑中/缺失")
        continue
    d = json.load(open(p))
    c = d.get("_config", {})
    parts = []
    for k in ("naive", "replay", "oml", "oml2", "joint"):
        v = d.get(k)
        if isinstance(v, dict) and "final_mean_acc" in v:
            parts.append(f"{k}={v['final_mean_acc']:.4f}")
            if k in ("naive", "replay", "oml", "oml2") and "mean_forget" in v:
                parts[-1] += f"(f={v['mean_forget']:.3f})"
    cfg = (f"backbone={c.get('backbone')} head={c.get('head')} cl={c.get('cl_method')} "
           f"pool={c.get('pool')} agg={int(bool(c.get('agg_path')))} "
           f"stat_in={int(bool(c.get('stat_input')))} d={c.get('domains')} "
           f"k={c.get('inner_k')} consol={c.get('consolidate_every')}")
    print(f"{t:20s} {'  '.join(parts)}")
    print(f"{'':20s}   {cfg}")

print("=" * 96)
print("参照 (3/6/12 域 MLP 聚合基线可达上限): 0.9159 / 0.8065 / 0.6638")
print("参照 (SSM 原版 joint):                 0.8510 / 0.7213 / 0.3674")
print("参照 (q5_agg 简单 replay):             0.5952  (6 域)")
