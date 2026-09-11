#!/usr/bin/env python3
"""对比若干 LM4 结果目录: joint 天花板 / naive / replay / 保留率。"""
import json, sys, os
import numpy as np

dirs = sys.argv[1:] or ["lm4A_mag", "lm4A_wfr"]
rows = []
for t in dirs:
    p = os.path.join("results", t, "lm4_wave_results.json")
    if not os.path.exists(p):
        print(f"!! 缺 {p}"); continue
    d = json.load(open(p))
    j = d.get("joint", {})
    n = d.get("naive", {})
    r = d.get("replay", {})
    rows.append(dict(
        tag=t, joint=j.get("final_mean_acc", float("nan")),
        per=j.get("per_domain", {}),
        naive=n.get("final_mean_acc", float("nan")),
        rep=r.get("final_mean_acc", float("nan")),
        forg=r.get("mean_forget", float("nan")),
    ))

if not rows:
    sys.exit(1)

print(f"{'tag':<14}{'naive':>8}{'replay':>8}{'forget':>8}{'joint':>8}{'保留率':>9}")
print("-" * 55)
for x in rows:
    ret = x["rep"] / x["joint"] * 100 if x["joint"] else float("nan")
    print(f"{x['tag']:<14}{x['naive']:>8.4f}{x['rep']:>8.4f}{x['forg']:>8.4f}"
          f"{x['joint']:>8.4f}{ret:>8.1f}%")

print("\n=== joint 天花板各域 (D0..D5, 随机 0.167) ===")
keys = sorted({k for x in rows for k in x["per"]}, key=lambda s: int(s))
print(f"{'tag':<14}" + "".join(f"{'D'+k:>8}" for k in keys))
for x in rows:
    print(f"{x['tag']:<14}" + "".join(
        f"{x['per'].get(k, float('nan')):>8.3f}" for k in keys))

print("\n=== replay 最终各域 ===")
for x in rows:
    p = os.path.join("results", x["tag"], "lm4_wave_results.json")
    d = json.load(open(p))
    M = d.get("replay", {}).get("acc_matrix", [])
    if M:
        last = M[-1]
        print(f"{x['tag']:<14}" + "".join(f"{v:>8.3f}" for v in last))
