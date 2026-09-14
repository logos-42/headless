#!/usr/bin/env python3
"""analyze_oak.py — OaK E9 vs E10 对照分析。

核心问题 (用户架构主张里最关键的一跳):
    Prediction -> Abstraction -> Option -> Planning -> Control
  是否比
    state -> RL policy -> action
  更适合持续变化的磁通动力学。

E9  = --proposer oak --oak-options 0   (同一 base、同一门控, 不用 option)
E10 = --proposer oak --oak-options 1   (+Options / 时间抽象)
→ **唯一变量是时间抽象**。

判据 (不止看均值, 也看用户要求的多项):
  any_time_acc          在线性能
  worst_case_forget     最差遗忘界 (CL 的一等指标)
  final_mean_acc        末轮平均
  option_starts/steps   option 是否**真的被用上** (用不上 = 机制没生效, 不等于无效)
  n_options / actions   发现了什么技能

统计: Welch t + 效应量; 多 seed 可靠性标准 std < mean/2。
"""
import glob
import json
import math
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return (0.0 if a.mean() == b.mean() else float("inf")), float("nan")
    t = (a.mean() - b.mean()) / se
    # Welch–Satterthwaite 自由度
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        # 无 scipy: 正态近似 (n>=5 时够用, 标注清楚)
        p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def collect(pat, arm):
    """收集所有匹配目录的 `arm` 结果。arm in (naive, replay)。"""
    out = {"any_time": [], "forget": [], "final": [], "opts": [], "starts": [],
           "steps": [], "ntrans": [], "nopt": [], "dirs": []}
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "lm4_wave_results.json")
        if not os.path.exists(f):
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if arm not in j:
            continue
        a = j[arm]
        out["dirs"].append(os.path.basename(d))
        out["any_time"].append(a.get("any_time_acc", float("nan")))
        out["forget"].append(a.get("worst_case_forget", float("nan")))
        out["final"].append(a.get("final_mean_acc", float("nan")))
        o = a.get("oak") or {}
        k = a.get("knowledge") or {}
        out["opts"].append(a.get("oak_transitions", o.get("n_trans", float("nan"))))
        out["starts"].append(o.get("option_starts", float("nan")))
        out["steps"].append(o.get("option_steps", float("nan")))
        out["ntrans"].append(o.get("n_trans", float("nan")))
        out["nopt"].append(k.get("n_options", float("nan")))
    return out


def fmt(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return "%.4f (n=1)" % v[0]
    return "%.4f ± %.4f (n=%d, CV %.1f%%)" % (
        v.mean(), v.std(ddof=1), len(v),
        100 * v.std(ddof=1) / abs(v.mean()) if v.mean() else float("nan"))


print("=" * 96)
print("OaK E9 (无 Options) vs E10 (有 Options) —— 唯一变量是时间抽象")
print("=" * 96)

for arm in ("replay", "naive"):
    e9 = collect(os.path.join(RES, "oak_e9_s*"), arm)
    e10 = collect(os.path.join(RES, "oak_e10_s*"), arm)
    if not e9["dirs"] and not e10["dirs"]:
        continue
    print()
    print("── arm = %s (%s) " % (arm, "有意义的那臂" if arm == "replay" else "对照") + "─" * 50)
    print("  E9  最终平均 acc : %s" % fmt(e9["final"]))
    print("  E10 最终平均 acc : %s" % fmt(e10["final"]))
    print("  E9  any-time     : %s" % fmt(e9["any_time"]))
    print("  E10 any-time     : %s" % fmt(e10["any_time"]))
    print("  E9  最差遗忘界   : %s" % fmt(e9["forget"]))
    print("  E10 最差遗忘界   : %s" % fmt(e10["forget"]))
    print("  ── 机制是否真的生效 (E10) ──")
    print("     发现的 options  : %s" % fmt(e10["nopt"]))
    print("     option 启动次数 : %s" % fmt(e10["starts"]))
    print("     option 执行步数 : %s" % fmt(e10["steps"]))
    print("     转移条数        : %s" % fmt(e10["ntrans"]))
    if np.nanmean(e10["steps"]) < 1 if len(e10["steps"]) else True:
        print("     ** 警告: option 几乎没被执行 -> 这不是「Options 无效」,")
        print("        而是「Options 机制没被触发」, 结论不能当负面证据 **")
    print("  ── 显著性 ──")
    for name, key, better in [("any_time  ", "any_time", "high"),
                              ("最差遗忘界", "forget", "low"),
                              ("最终 acc  ", "final", "high")]:
        t, p = welch(e10[key], e9[key])
        if not np.isfinite(t):
            print("     %s 样本不足" % name)
            continue
        d = np.nanmean(e10[key]) - np.nanmean(e9[key])
        rel = 100 * d / abs(np.nanmean(e9[key])) if np.nanmean(e9[key]) else float("nan")
        verdict = ("E10 更好" if ((d > 0) == (better == "high")) else "E9 更好")
        print("     %s Δ=%+.4f (%+.1f%%)  t=%+.2f  p=%.4f  -> %s  %s"
              % (name, d, rel, t, p, verdict,
                 "**显著**" if p < 0.05 else "不显著"))

print()
print("=" * 96)
print("判读口径 (防止过度解读)")
print("=" * 96)
print("""
  1. 如果 option_steps ≈ 0 -> **机制没被触发**, 不能得出「Options 无效」的结论。
     必须先修触发条件 (option 起点匹配半径 / τ_U 过滤强度) 再重跑。
  2. 如果 E10 与 E9 的差异 p > 0.05 -> 如实写成负面结论, 不用「方向一致」搪塞。
  3. 多 seed 可靠性: std > mean/2 的指标不可用于结论 (用户定的标准)。
  4. 本实验是**被动观测 + 离线/模型内反事实**, 没有真实 actuator。
     口径必须是 "predictive/control-policy learning on passive observations",
     不能写成 "完成了磁通控制实验"。
""")
