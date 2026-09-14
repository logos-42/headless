#!/usr/bin/env python3
"""lm5 多模态: MLP / SSM / 混合骨干 三方对照 (3 seed × 2000 步)。"""
import json
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"
SEEDS = [42, 1, 7]
BB = [("mlp", "lm5_d_mlp_s{}"), ("ssm", "lm5_d_ssm_s{}"), ("hybrid", "lm5_e_hyb_s{}")]


def collect(pat):
    out = {}
    doms = None
    for s in SEEDS:
        p = f"{R}/{pat.format(s)}/lm5_mm_results.json"
        if not os.path.exists(p):
            return None, {}
        d = json.load(open(p))
        doms = d["domains"]
        out[s] = np.array(d["matrix"][-1], dtype=float)
    return doms, out


data = {}
for name, pat in BB:
    d, m = collect(pat)
    if m:
        data[name] = (d, m)
if not data:
    print("无结果")
    raise SystemExit
doms = next(iter(data.values()))[0]

print("=" * 92)
print("LM5 多模态 · MLP / SSM / 混合骨干 三方对照 (3 seed × 2000 步, 学完 5 域后的最终值)")
print("=" * 92)
print()
hdr = "  %-14s" % "域"
for name, _ in BB:
    if name in data:
        hdr += "%-24s" % (name + " mean±std (CV)")
print(hdr)
print("  " + "-" * 86)
for j, k in enumerate(doms):
    line = "  %-14s" % k
    vals = {}
    for name, _ in BB:
        if name not in data:
            continue
        a = np.array([data[name][1][s][j] for s in data[name][1] if s in data[name][1]])
        m, sd = a.mean(), a.std()
        vals[name] = m
        tag = "可信" if (m > 0 and sd < m / 2) else "不可信"
        line += "%.4f±%.4f(%5.1f%% %s)" % (m, sd, 100 * sd / max(1e-9, abs(m)), tag) + " "
    if len(vals) >= 2:
        best = max(vals, key=vals.get)
        line += " <- " + best
    print(line)

print()
print("  ── 各骨干平均 ──")
for name, _ in BB:
    if name not in data:
        continue
    m = data[name][1]
    allcols = np.mean([np.array([m[s][j] for s in m]).mean() for j in range(len(doms))])
    wc = np.mean([np.array([m[s][j] for s in m]).mean() for j in range(3, len(doms))])
    print("  %-8s 全部 8 列 %.4f | 仅 wave+causal 四项 %.4f" % (name, allcols, wc))

print()
print("  ── 判读参照 ──")
print("  随机线: en/zh/code=0.0002  wave=0.167  causal_bal/do_bal=0.2")
print("  多数类: en=0.094  zh=0.033  code=0.122  causal=0.296  causal_do=0.508")
print("  预期(若模态依赖成立): 文本 SSM/混合 赢; wave MLP/混合 赢; causal 三者相近")
