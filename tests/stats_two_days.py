#!/usr/bin/env python3
"""两天结果的统计裁决: agg-path vs 基线 的显著性 + Q4 任务定义效应.

用法: /work/liuyuanjie/envs/vllm-cu128/bin/python tests/stats_two_days.py
"""
import json, math, statistics as st
from pathlib import Path

RES = Path("results")


def load(tag):
    f = RES / tag / "lm4_wave_results.json"
    return json.load(open(f)) if f.exists() else None


def acc(d, key):
    v = (d or {}).get(key)
    return v.get("final_mean_acc") if v else None


def forget(d, key):
    v = (d or {}).get(key)
    return v.get("mean_forget") if v else None


def welch(a, b):
    """Welch t 检验, 返回 (t, df, 近似 p 双侧)."""
    na, nb = len(a), len(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, na + nb - 2, 1.0
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 用正态近似估 p (df 通常 >10, 足够)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, df, p


BASE = ["lm4A_wfr", "lm4A_wfr_s1", "lm4A_wfr_s2", "lm4A_wfr_s3", "lm4A_wfr_s7",
        "lm4A_wfr_s11", "lm4A_wfr_s123", "lm4A_wfr_s999", "lm4A_wfr_s2026"]
AGG = ["q3_agg_rr2", "q5_agg_s7", "q5_agg_s2026", "q5_agg_s123",
       "q5_agg_s1", "q5_agg_s2", "q5_agg_s3"]

print("=" * 92)
print("裁决 1 · agg-path + rr2 (n=7) vs 基线 pool=last rr1 (n=9)")
print("=" * 92)

for label, tags in [("基线 pool=last rr1.0", BASE), ("agg-path pool=cat rr2.0", AGG)]:
    rp = [acc(load(t), "replay") for t in tags]
    fg = [forget(load(t), "replay") for t in tags]
    rp = [x for x in rp if x is not None]
    fg = [x for x in fg if x is not None]
    cv = st.stdev(rp) / st.mean(rp) * 100
    print(f"{label:26s} n={len(rp)}  replay {st.mean(rp):.4f}±{st.stdev(rp):.4f} "
          f"(CV {cv:4.1f}%)  遗忘 {st.mean(fg):.4f}±{st.stdev(fg):.4f}")

rp_b = [acc(load(t), "replay") for t in BASE]
rp_a = [acc(load(t), "replay") for t in AGG]
fg_b = [forget(load(t), "replay") for t in BASE]
fg_a = [forget(load(t), "replay") for t in AGG]

t, df, p = welch(rp_a, rp_b)
print(f"\n  replay 差 = {st.mean(rp_a)-st.mean(rp_b):+.4f}   Welch t={t:.2f} (df={df:.1f})  p≈{p:.3f}"
      f"   → {'显著' if p < 0.05 else '不显著'}")
t, df, p = welch(fg_a, fg_b)
print(f"  遗忘   差 = {st.mean(fg_a)-st.mean(fg_b):+.4f}   Welch t={t:.2f} (df={df:.1f})  p≈{p:.3f}"
      f"   → {'显著' if p < 0.05 else '不显著'}")

# 方差比 (F 检验)
vb, va = st.variance(rp_b), st.variance(rp_a)
F = vb / va
print(f"\n  方差比 F = {F:.2f} (基线方差 / agg 方差)  → agg-path 把 seed 间波动压小 {math.sqrt(F):.1f}×")

print()
print("=" * 92)
print("裁决 2 · Q4 任务定义 (域数 / 顺序 / 特征), 均为 seed=42 单点")
print("=" * 92)
rows = [("3 域", "q4_d3"), ("6 域 (基线)", "lm4A_wfr"), ("8 域", "q4_d8"), ("12 域", "q4_d12")]
print(f"{'配置':16s} {'naive':>7s} {'replay':>7s} {'joint':>7s} {'保留率':>7s} {'遗忘':>7s}")
for label, tag in rows:
    d = load(tag)
    nv, rp, jo, fg = acc(d, "naive"), acc(d, "replay"), acc(d, "joint"), forget(d, "replay")
    ret = rp / jo * 100 if (rp and jo) else float("nan")
    print(f"{label:16s} {nv:7.4f} {rp:7.4f} {jo:7.4f} {ret:6.1f}% {fg:7.4f}")

print()
print(f"{'配置':16s} {'replay':>7s} {'joint':>7s} {'保留率':>7s} {'遗忘':>7s}   对照")
for label, tag, note in [("6域 密度递增", "lm4A_wfr", "基线"),
                         ("6域 随机顺序", "q4_d6_shuf", "vs 基线"),
                         ("6域 +L-shell", "q4_d6_ls", "vs 基线"),
                         ("6域 +agg-path", "q4_d6_agg", "vs 基线")]:
    d = load(tag)
    rp, jo, fg = acc(d, "replay"), acc(d, "joint"), forget(d, "replay")
    ret = rp / jo * 100
    print(f"{label:16s} {rp:7.4f} {jo:7.4f} {ret:6.1f}% {fg:7.4f}   {note}")

# 顺序效应
d_shuf, d_base = load("q4_d6_shuf"), load("lm4A_wfr")
print(f"\n  随机顺序代价: replay {acc(d_shuf,'replay')-acc(d_base,'replay'):+.4f}, "
      f"遗忘 {forget(d_shuf,'replay')-forget(d_base,'replay'):+.4f}  → 密度递增课程有价值")
# L-shell 效应
d_ls = load("q4_d6_ls")
print(f"  L-shell 增益: replay {acc(d_ls,'replay')-acc(d_base,'replay'):+.4f}, "
      f"遗忘 {forget(d_ls,'replay')-forget(d_base,'replay'):+.4f}")
# 12 域崩溃
d12 = load("q4_d12")
pd12 = d12["joint"]["per_domain"]
bad = {k: v for k, v in pd12.items() if v < 0.30}
print(f"\n  12 域天花板崩塌: joint {acc(d12,'joint'):.4f} (naive {acc(d12,'naive'):.4f}); "
      f"接近随机的域 = {sorted(bad.items(), key=lambda x: x[1])}")
