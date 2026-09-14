#!/usr/bin/env python3
"""lm5 多模态: MLP(新骨干) vs SSM(旧骨干), 3 seed × 2000 步 对照。"""
import json
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"
SEEDS = [42, 1, 7]


def collect(bb):
    out = {}
    doms = None
    for s in SEEDS:
        p = f"{R}/lm5_d_{bb}_s{s}/lm5_mm_results.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        doms = d["domains"]
        out[s] = np.array(d["matrix"][-1], dtype=float)
    return doms, out


doms, mlp = collect("mlp")
_, ssm = collect("ssm")

print("=" * 84)
print("LM5 多模态 · MLP(新) vs SSM(旧) —— 3 seed × 2000 步, 学完 5 域后的最终值")
print("=" * 84)
print()
print("  %-14s %-22s %-22s %8s" % ("域", "MLP mean±std (CV)", "SSM mean±std (CV)", "差值"))
print("  " + "-" * 70)
rows = []
for j, k in enumerate(doms):
    a = np.array([mlp[s][j] for s in mlp])
    b = np.array([ssm[s][j] for s in ssm])
    ma, sa = a.mean(), a.std()
    mb, sb = b.mean(), b.std()
    cva = 100 * sa / max(1e-9, abs(ma))
    cvb = 100 * sb / max(1e-9, abs(mb))
    ok_a = "可信" if (ma > 0 and sa < ma / 2) else "不可信"
    ok_b = "可信" if (mb > 0 and sb < mb / 2) else "不可信"
    print("  %-14s %.4f±%.4f (%5.1f%% %s)  %.4f±%.4f (%5.1f%% %s)  %+.4f" %
          (k, ma, sa, cva, ok_a, mb, sb, cvb, ok_b, ma - mb))
    rows.append((k, ma, sa, mb, sb))

print()
print("  ── 逐 seed 原始值 (最终行) ──")
print("  %-14s %-22s %-22s" % ("域", "MLP s42/s1/s7", "SSM s42/s1/s7"))
for j, k in enumerate(doms):
    print("  %-14s %-22s %-22s" %
          (k, "/".join("%.3f" % mlp[s][j] for s in SEEDS),
           "/".join("%.3f" % ssm[s][j] for s in SEEDS)))

print()
print("  ── 判读参照 ──")
print("  %-14s %-10s %-10s %-10s" % ("域", "随机线", "多数类基线", "结论"))
ref = [("en", "0.0002", "0.094"), ("zh", "0.0002", "0.033"),
       ("code", "0.0002", "0.122"), ("wave", "0.167", "-"),
       ("causal", "0.2", "0.296"), ("causal_do", "0.2", "0.508"),
       ("causal_bal", "0.2", "-"), ("causal_do_bal", "0.2", "-")]
for (k, r_, m_), (k2, ma, _, mb, _) in zip(ref, rows):
    best = max(ma, mb)
    if k in ("en", "zh", "code"):
        msg = "低于多数类" if best < float(m_) else "高于多数类"
    elif k == "causal_do":
        msg = "≈多数类(无增益)" if abs(best - float(m_)) < 0.03 else "有增益"
    else:
        msg = "高于随机" if best > float(r_) + 0.02 else "≈随机"
    print("  %-14s %-10s %-10s %s" % (k, r_, m_, msg))

print()
print("  平均(全部 8 列): MLP %.4f  SSM %.4f" %
      (np.mean([r[1] for r in rows]), np.mean([r[3] for r in rows])))
print("  平均(仅 wave+causal 四项): MLP %.4f  SSM %.4f" %
      (np.mean([r[1] for r in rows[3:]]), np.mean([r[3] for r in rows[3:]])))
