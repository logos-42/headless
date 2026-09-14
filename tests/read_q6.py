#!/usr/bin/env python3
"""读 q6 双循环结果 + stat-input 天花板。"""
import json
import glob
import os
import sys

os.chdir("/work/liuyuanjie/headless")


def show(tag, want=("replay", "oml2", "oml", "naive", "joint")):
    p = f"results/{tag}/lm4_wave_results.json"
    if not os.path.exists(p):
        print(f"  {tag}: 缺失")
        return None
    d = json.load(open(p))
    c = d.get("_config", {})
    print(f"  {tag}:")
    for k in want:
        v = d.get(k)
        if isinstance(v, dict) and "final_mean_acc" in v:
            fg = v.get("mean_forget", float("nan"))
            print(f"     {k:8s} acc={v['final_mean_acc']:.4f}  forget={fg:.4f}")
    print(f"     cfg: cl={c.get('cl_method')} pool={c.get('pool')} "
          f"agg={c.get('agg_path')} stat_in={c.get('stat_input')} "
          f"backbone={c.get('backbone')} rr={c.get('replay_ratio')} "
          f"reptile={c.get('reptile_lr')} k={c.get('inner_k')} "
          f"consol={c.get('consolidate_every')} ep={c.get('epochs_per_domain')}")
    return d


print("=============== q6 双循环 (修复版) ===============")
for t in ["q6_oml2", "q6_oml2_rep", "q6_oml", "q6_oml2_k5"]:
    show(t)

print()
print("=============== stat-input 天花板 ===============")
for D in (3, 6, 12):
    show(f"si_d{D}", want=("joint",))

print()
print("=============== 参照: q5_agg 基线 (简单 replay) ===============")
for t in ["q5_agg_s7", "q5_agg_s1", "q3_agg_rr2"]:
    show(t, want=("replay", "joint"))

print()
print("=============== 探针参照 (MLP 可达上限) ===============")
print("  3 域 0.9159 | 6 域 0.8065 | 12 域 0.6638")
