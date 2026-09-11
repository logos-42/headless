#!/usr/bin/env python3
"""汇总 LM4 修正版 (stride=32 不重叠) 全部结果。"""
import json, glob, os, re
import statistics as st

rows = []
for jf in sorted(glob.glob("results/lm4c_*/lm4_wave_results.json")):
    tag = os.path.basename(os.path.dirname(jf))
    # 排除非标准 run: lm4c_joint20k 用 --epochs-per-domain 1, 其 naive/replay 无意义
    if "joint" in tag:
        continue
    try:
        d = json.load(open(jf))
    except Exception as e:
        print(f"!! {tag}: 读取失败 {e}")
        continue
    naive = d.get("naive", {}).get("final_mean_acc", float("nan"))
    r = d.get("replay", {})
    joint = d.get("joint", {}).get("final_mean_acc", float("nan"))
    m = re.match(r"lm4c_rr([\d.]+)_s(\d+)$", tag)
    if m:
        cfg, seed = "ratio=" + m.group(1), int(m.group(2))
    else:
        m2 = re.match(r"lm4c_s(\d+)$", tag)
        cfg, seed = "ratio=1.0", int(m2.group(1)) if m2 else -1
    rows.append(dict(tag=tag, cfg=cfg, seed=seed, naive=naive,
                     acc=r.get("final_mean_acc", float("nan")),
                     forg=r.get("mean_forget", float("nan")), joint=joint))

print(f"{'tag':<22}{'cfg':>11}{'seed':>6}{'naive':>9}{'replay':>9}{'forget':>9}{'joint':>8}")
print("-" * 74)
for x in sorted(rows, key=lambda t: (t["cfg"], t["seed"])):
    print(f"{x['tag']:<22}{x['cfg']:>11}{x['seed']:>6}{x['naive']:>9.4f}"
          f"{x['acc']:>9.4f}{x['forg']:>9.4f}{x['joint']:>8.4f}")

print("\n=== 按配置聚合 (replay 最终 acc) ===")
cfgs = {}
for x in rows:
    cfgs.setdefault(x["cfg"], []).append(x["acc"])
for cfg, v in sorted(cfgs.items()):
    v = [a for a in v if a == a]
    m = st.mean(v); s = st.stdev(v) if len(v) > 1 else 0.0
    ok = "PASS" if s < m / 2 else "FAIL"
    print(f"  {cfg:<12} n={len(v):<3} mean={m:.4f} std={s:.4f} "
          f"(CV {s/m*100 if m else 0:.1f}%)  std<mean/2? {ok}  min={min(v):.4f} max={max(v):.4f}")

print("\n=== naive 基线一致性 ===")
nv = [x["naive"] for x in rows if x["naive"] == x["naive"]]
print(f"  n={len(nv)}  唯一值: {sorted(set(round(v,4) for v in nv))}")

jn = [x["joint"] for x in rows if x["joint"] == x["joint"]]
if jn:
    print(f"\n=== joint 上限 (各 run 内联的 5000 步诊断) ===\n  {jn}")
# 真上限: lm4c_joint20k (20000 步)
try:
    _j = json.load(open("results/lm4c_joint20k/lm4_wave_results.json"))["joint"]
    print(f"\n=== joint 真上限 (20000 步) ===\n  {_j['final_mean_acc']:.4f}  "
          f"per_domain={ {k: round(v,3) for k,v in _j['per_domain'].items()} }")
except Exception as e:
    print(f"\n(joint20k 未找到: {e})")
