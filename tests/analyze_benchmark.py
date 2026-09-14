#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LM4 修正版 benchmark 汇总: 任务流 + 调度臂, 报边缘设备关心的指标。

路径前缀同时覆盖两批 (2026-09-14):
  - `bm_bins_*`  任务流臂 (不受价值函数 bug 影响)
  - `bm2_*`      修正版价值函数臂 (con 实现 / sim 自适应 / 池 60)

指标:
  replay            最终平均准确率
  any_time          在线全程平均 (边学边可用)
  worst_forget      最差遗忘界 (任一旧任务的最大跌幅; 安全功能不能崩)
  mean_forget       平均遗忘

统计口径:
  mean±std 用样本标准差 (ddof=1); 两两对比用 Welch t + Welch–Satterthwaite 自由度,
  双侧 p 由 t 分布精确算 (无 scipy 依赖)。判定阈值 |t| >= 2 —— n=3 时只算边缘证据。

已知数据缺口 (见 docs/bm_benchmark_results.md §0):
  lm4 `bm2_random-matched_s*` 的 json 被 int64 序列化 bug 截断 → 回退读 `lm4_wave_report.md`
  (只有 replay / 平均遗忘), any-time 与最差遗忘界缺失。
"""
import glob
import json
import math
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"
SEEDS = ["42", "1", "7"]


# ---------------- 统计 ----------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lb + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def welch(a, b):
    """Welch t 检验; 返回 Δ / SE / t / df / p(双侧), 样本 < 2 则返回 None。"""
    a = [x for x in a if x is not None and x == x]
    b = [x for x in b if x is not None and x == x]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return dict(delta=ma - mb, se=0.0, t=float("nan"), df=float("nan"), p=float("nan"))
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return dict(delta=ma - mb, se=se, t=t, df=df, p=_betai(df / 2.0, 0.5, df / (df + t * t)))


# ---------------- 装载 ----------------
def _clean(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return None if x != x else x


def load(pat):
    """pat 形如 'bm2_value_s{s}'; 返回 {seed: {...}}。json 坏了则回退 report.md。"""
    out = {}
    for s in SEEDS:
        d = os.path.join(R, pat.format(s=s))
        if not os.path.isdir(d):
            continue
        rec = None
        p = os.path.join(d, "lm4_wave_results.json")
        if os.path.exists(p):
            try:
                r = (json.load(open(p)) or {}).get("replay") or {}
                rec = {
                    "replay": _clean(r.get("final_mean_acc")),
                    "any_time": _clean(r.get("any_time_acc")),
                    "worst": _clean(r.get("worst_case_forget")),
                    "mean_f": _clean(r.get("mean_forget_all")),
                    "src": "json",
                }
            except Exception:
                rec = None
        if rec is None:
            rp = os.path.join(d, "lm4_wave_report.md")
            if os.path.exists(rp):
                for line in open(rp):
                    c = [x.strip() for x in line.strip().strip("|").split("|")]
                    if len(c) == 3 and c[0] == "replay":
                        try:
                            rec = {"replay": float(c[1]), "mean_f": float(c[2]),
                                   "any_time": None, "worst": None, "src": "report.md"}
                        except Exception:
                            pass
        if rec:
            out[s] = rec
    return out


ARMS = [
    ("启发式 fixed 域序", "bm_bins_fixed_s{s}"),
    ("流: perm (随机排列)", "bm_bins_perm_s{s}"),
    ("流: revisit (反复)", "bm_bins_revisit_s{s}"),
    ("流: nonstationary", "bm_bins_nonstationary_s{s}"),
    ("★ 价值函数 value", "bm2_value_s{s}"),
    ("  消融 value-nofb", "bm2_value-nofb_s{s}"),
    ("  均匀随机 random", "bm2_random_s{s}"),
    ("★ 频率对齐 random-matched", "bm2_random-matched_s{s}"),
]

# lm4 `random-matched` 原 driver 的 json 全部被 int64 bug 截断, 且 seed 1/7 的结果文件
# 事后被另一路会话用**改过的脚本**重跑覆盖过。主口径取"与 value 臂同修订版"的原日志值
# (seed 42 已用 不带 default= 的反向修复独立复现, 逐位相同), 其余作敏感性对照。
RM_MATCHED_ORIG = {"42": (0.7635, 0.1178), "1": (0.7542, 0.1982), "7": (0.7594, 0.1116)}
USE_ORIG_RM = os.environ.get("BM_RM_SENSITIVITY", "0") != "1"
if USE_ORIG_RM:
    _old = load("bm2_random-matched_s{s}")
    _patched = {}
    for s, (acc, mf) in RM_MATCHED_ORIG.items():
        # any_time / worst 原 json 被截断, 且目录值已被新修订版覆盖 -> 主口径留空,
        # 不拿"混合修订版"的数字充数 (想看被覆盖后的值: BM_RM_SENSITIVITY=1)
        rec = dict(_old.get(s) or {})
        rec.update({"replay": acc, "mean_f": mf, "any_time": None, "worst": None,
                    "src": "bm2.log (同修订版)"})
        _patched[s] = rec
    _load = load

    def load(pat, _p=_patched, _l=_load):  # noqa: F811
        if pat == "bm2_random-matched_s{s}":
            return _p
        return _l(pat)


def fmt(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    if len(v) == 1:
        return "%.4f" % v[0]
    return "%.4f±%.4f" % (np.mean(v), np.std(v, ddof=1))


print("=" * 104)
print("LM4 修正版 benchmark · 任务流 + 调度臂 (12 轮 × 3 seed, 6 域, replay)")
print("  主口径: bm_bins_* (任务流) + bm2_* (修正版价值函数); random-matched = 同修订版日志值")
print("  设 BM_RM_SENSITIVITY=1 可改用被覆盖后的目录值 (敏感性对照)")
print("=" * 104)
print()
print("  %-26s %-20s %-20s %-14s %-12s %s" %
      ("臂", "replay mean±std", "any-time mean±std", "最差遗忘界", "平均遗忘", "n"))
print("  " + "-" * 98)

rows = {}
recs = {}
for name, pat in ARMS:
    d = load(pat)
    recs[name] = d
    if not d:
        print("  %-26s (无结果)" % name)
        continue
    rep = [v["replay"] for v in d.values()]
    ant = [v["any_time"] for v in d.values()]
    wst = [v["worst"] for v in d.values()]
    mf = [v["mean_f"] for v in d.values()]
    rep_v = [x for x in rep if x is not None]
    cv = 100 * np.std(rep_v, ddof=1) / max(1e-9, abs(np.mean(rep_v))) if len(rep_v) > 1 else float("nan")
    rows[name] = (np.mean(rep_v) if rep_v else np.nan,
                  np.std(rep_v, ddof=1) if len(rep_v) > 1 else 0.0, len(rep_v))
    print("  %-26s %-20s %-20s %-14s %-12s %d  (CV %4.1f%%)" %
          (name, fmt(rep), fmt(ant),
           ("%.4f" % np.mean([x for x in wst if x is not None])) if any(x is not None for x in wst) else "-",
           ("%.4f" % np.mean([x for x in mf if x is not None])) if any(x is not None for x in mf) else "-",
           len(rep_v), cv))
    src = sorted(set(v["src"] for v in d.values()))
    if src != ["json"]:
        print("  %-26s   ↳ 数据来源: %s | 逐 seed: %s" %
              ("", ",".join(src), {s: (round(v["replay"], 4) if v["replay"] is not None else None)
                                   for s, v in d.items()}))

print()
print("  ── 关键对比 (Welch t, ddof=1; |t|>=2 判显著) ──")


def cmp(a, b, key="replay"):
    da, db = recs.get(a), recs.get(b)
    if not da or not db:
        return "  %-30s vs %-30s  (数据不全)" % (a, b)
    A = [da[s][key] for s in SEEDS if s in da]
    B = [db[s][key] for s in SEEDS if s in db]
    w = welch(A, B)
    if not w:
        return "  %-30s vs %-30s  (样本不足: %d vs %d)" % (a, b, len(A), len(B))
    return ("  %-30s vs %-30s [%-7s] Δ=%+.4f  SE=%.4f  t=%+.2f  df=%.1f  p=%.4f  -> %s" %
            (a, b, key, w["delta"], w["se"], w["t"], w["df"], w["p"],
             "★显著" if abs(w["t"]) >= 2 else "不显著"))


for a, b in [("★ 价值函数 value", "★ 频率对齐 random-matched"),
             ("★ 价值函数 value", "  均匀随机 random"),
             ("★ 价值函数 value", "  消融 value-nofb"),
             ("★ 价值函数 value", "流: perm (随机排列)"),
             ("启发式 fixed 域序", "流: perm (随机排列)")]:
    for key in ("replay", "any_time"):
        print(cmp(a, b, key))
print()
print("  判读要点: `value` vs `random-matched` 才是「价值函数是否更聪明」的干净对照;")
print("           `value` vs `random` 混淆了「更聪明」与「复习更均匀」。")
print("           注意 `random` 臂三 seed 几乎相同 (CV ~0), 其 t 值会被虚假放大, 不作为结论。")
