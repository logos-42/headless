#!/usr/bin/env python3
"""LM4 + LM5 benchmark 完整汇总 + 两两 Welch t 检验。

输出 (stdout) 供 docs/bm_benchmark_results.md 使用; 另存 JSON 供机读。

关键对照:
  value vs random-matched  <-- 唯一干净对照 (同为 value 臂实测提议频率驱动)
  value vs random          <-- 混入 "复习更均匀" 效应, 只能作参考
  value vs value-nofb      <-- 隔离真实反馈项 (lam_fb)
"""
import json
import math
import os
import sys

import numpy as np

R = "/work/liuyuanjie/headless/results"
SEEDS = ["42", "1", "7"]

LM4_ARMS = [
    ("fixed(bins)", "bm_bins_fixed_s{s}", "stream"),
    ("perm", "bm_bins_perm_s{s}", "stream"),
    ("revisit", "bm_bins_revisit_s{s}", "stream"),
    ("nonstationary", "bm_bins_nonstationary_s{s}", "stream"),
    ("value", "bm_value_s{s}", "proposer"),
    ("value-nofb", "bm_value-nofb_s{s}", "proposer"),
    ("random", "bm_random_s{s}", "proposer"),
    ("random-matched", "bm_random-matched_s{s}", "proposer"),
]

LM5_ARMS = [
    ("fixed", "bm5_fixed_s{s}", "stream"),
    ("perm", "bm5_perm_s{s}", "stream"),
    ("revisit", "bm5_revisit_s{s}", "stream"),
    ("nonstationary", "bm5_nonstationary_s{s}", "stream"),
    ("value", "bm5_value_s{s}", "proposer"),
    ("value-nofb", "bm5_value-nofb_s{s}", "proposer"),
    ("random-matched", "bm5_random-matched_s{s}", "proposer"),
]

KEY = [("value", "random-matched", "★干净对照: 价值函数 vs 频率对齐随机"),
       ("value", "random", "参考(混淆复习均匀度): value vs 均匀随机"),
       ("value", "value-nofb", "隔离真实反馈项 lam_fb"),
       ("value", "perm", "value vs 随机排列流"),
       ("value", "revisit", "value vs 反复流"),
       ("random-matched", "random", "random-matched vs random (验证频率对齐本身的作用)")]


def lm4_load(pat):
    out = {}
    for s in SEEDS:
        p = f"{R}/{pat.format(s=s)}/lm4_wave_results.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p)).get("replay") or {}
        out[s] = {"replay": r.get("final_mean_acc"),
                  "any_time": r.get("any_time_acc"),
                  "worst": r.get("worst_case_forget"),
                  "mean_f": r.get("mean_forget_all")}
    return out


def lm5_load(pat):
    out = {}
    for s in SEEDS:
        p = f"{R}/{pat.format(s=s)}/lm5_mm_results.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        m = d.get("matrix") or []
        out[s] = {"replay": float(np.mean(m[-1])) if m else None,
                  "any_time": d.get("any_time_acc"),
                  "worst": d.get("worst_case_forget"),
                  "mean_f": d.get("mean_forget_all"),
                  "last_row": [round(float(x), 4) for x in m[-1]] if m else None,
                  "domains": d.get("domains")}
    return out


def agg(d, k):
    v = [d[s][k] for s in sorted(d) if d[s].get(k) is not None]
    if not v:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def welch(a, b):
    ma, sa, na = a
    mb, sb, nb = b
    if not na or not nb or any(map(lambda x: x != x, [ma, mb])):
        return (float("nan"), float("nan"), "数据不全")
    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    if se <= 0:
        return (ma - mb, float("nan"), "方差为0")
    t = (ma - mb) / se
    return (ma - mb, t, "显著 |t|>=2" if abs(t) >= 2 else "不显著 |t|<2")


def report(tag, arms, loader):
    print("=" * 104)
    print(f"{tag}  (3 seed, mean±std, ddof=1)")
    print("=" * 104)
    rows = {}
    print("  %-16s %-9s %-22s %-22s %-12s %-12s" %
          ("臂", "类型", "replay mean±std", "any-time mean±std", "最差遗忘界", "平均遗忘"))
    print("  " + "-" * 100)
    for name, pat, kind in arms:
        d = loader(pat)
        if not d:
            print("  %-16s %-9s (无结果)" % (name, kind))
            continue
        rep, ant, wst, mf = agg(d, "replay"), agg(d, "any_time"), agg(d, "worst"), agg(d, "mean_f")
        rows[name] = {"replay": rep, "any_time": ant, "worst": wst, "mean_f": mf,
                      "n_seed": len(d), "per_seed": d}
        print("  %-16s %-9s %.4f±%.4f (n=%d)   %.4f±%.4f          %.4f       %.4f" %
              (name, kind, rep[0], rep[1], rep[2], ant[0], ant[1], wst[0], mf[0]))
    print()
    print("  ── 两两 Welch t (指标 = replay / any-time / 最差遗忘界) ──")
    tests = {"replay": [], "any_time": [], "worst": []}
    for a, b, desc in KEY:
        if a not in rows or b not in rows:
            print("  (%s) 数据不全" % desc)
            continue
        for k in ("replay", "any_time"):
            delta, t, verdict = welch(rows[a][k], rows[b][k])
            tests[k].append({"a": a, "b": b, "desc": desc, "metric": k,
                             "delta": delta, "t": t, "verdict": verdict})
        # 统一约定: Δ = a - b。对 worst_case_forget **越小越好**,
        # 因此 Δ>0 表示 a 的遗忘界更大 = a 更不安全。
        dw, tw, _ = welch(rows[a]["worst"], rows[b]["worst"])
        tests["worst"].append({"a": a, "b": b, "desc": desc, "metric": "worst",
                               "delta": dw, "t": tw,
                               "verdict": "显著 |t|>=2" if abs(tw) >= 2 else "不显著 |t|<2"})
    for k in ("replay", "any_time", "worst"):
        print("  [%s]%s" % (k, " (Δ = a-b; 遗忘界越小越好 => Δ>0 表示 a 更不安全)" if k == "worst" else " (Δ = a-b)"))
        for x in tests[k]:
            print("    %-15s vs %-15s Δ=%+.4f t=%+.2f  %s   <- %s" %
                  (x["a"], x["b"], x["delta"], x["t"], x["verdict"], x["desc"]))
        print()
    return rows, tests


lm4_rows, lm4_tests = report("LM4 新 benchmark", LM4_ARMS, lm4_load)
print()
lm5_rows, lm5_tests = report("LM5 多模态 benchmark", LM5_ARMS, lm5_load)

print("=" * 104)
print("LM5 各臂 最后一行 (末轮每域准确率) 与 domains")
print("=" * 104)
for name, _, _ in LM5_ARMS:
    if name not in lm5_rows:
        continue
    d = lm5_rows[name]["per_seed"]
    if "42" not in d:
        continue
    print("  %-16s domains(%d)=%s" % (name, len(d["42"]["domains"] or []), d["42"]["domains"]))
    print("  %-16s last_row=%s" % ("", d["42"]["last_row"]))
print()

def pack(rows):
    return {k: {"replay": list(v["replay"]), "any_time": list(v["any_time"]),
                "worst": list(v["worst"]), "mean_f": list(v["mean_f"]),
                "n_seed": v["n_seed"], "per_seed": v["per_seed"]}
            for k, v in rows.items()}

out = {"lm4": pack(lm4_rows), "lm5": pack(lm5_rows),
       "lm4_tests": lm4_tests, "lm5_tests": lm5_tests}
json.dump(out, open("/tmp/bm_summary.json", "w"), indent=1, ensure_ascii=False)
print("[json] /tmp/bm_summary.json")
