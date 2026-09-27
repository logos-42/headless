#!/usr/bin/env python3
"""analyze_k1v2.py — K1v2 的分析器。**先过机制闸门, 再谈对照**。

## 设计原则(本轮的教训直接写成代码)

上一轮 K1 / 长测的闭环档**一步都没执行**(coverage 写入未接线, 见
`docs/oak_repro_ground_truth.md`), 但结果文件里 `option_starts > 0`,
看上去像"机制在跑" —— 一个**静默的空转实验**。

所以本分析器第一步不是算均值, 而是**验证机制计数自洽**;
任何一臂不过闸门, 就**拒绝给出结论**, 只打印诊断。

## 闸门(事前写死)

  G1  start_blocked == 0            (修复后不应再出现算不出动作的起 option)
  G2  档位与计数形态匹配:
        e9      : option_steps == 0
        fixed   : option_steps > 0 且 replan_steps == 0
        goal*   : option_steps > 0 且 replan_steps ≈ option_steps (>0)
  G3  option_starts 与 option_steps 的比例合理:
        闭环档每轮重算, 允许 starts << steps, 但 **starts 不能恒为 1**
        (恒为 1 = 一个 option 锁死跑满全程 = 旧版的病灶)
  G4  同一档位的多个 seed 不能给出**逐位相同**的指标
        (逐位相同 => 该档没生效, 不是"效果相同")

## 判据

  O2 > O1  -> 问题是 open-loop 执行, 不是 temporal abstraction
  O3/O4 >= O2 -> 闭环之上还需要自适应终止/抢占
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys


def load(root="results"):
    arms = {}
    for d in sorted(glob.glob(os.path.join(root, "oak_k1v2_*"))):
        f = os.path.join(d, "lm4_wave_results.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        name = os.path.basename(d)[len("oak_k1v2_"):]
        mode = "e9" if name.startswith("e9") else name.rsplit("_s", 1)[0]
        seed = name.rsplit("_s", 1)[-1]
        rec: dict = {"arm": name, "mode": mode, "seed": seed}
        for sec in ("naive", "replay"):
            if isinstance(j.get(sec), dict):
                rec[sec] = {k: v for k, v in j[sec].items() if isinstance(v, (int, float))}
                rec[sec + "_stats"] = j[sec].get("proposer_stats") or {}
        arms[name] = rec
    return arms


def gate(arms):
    """返回 (是否全过, 报告行列表)。"""
    bad = []
    lines = []
    for name, r in sorted(arms.items()):
        s = r.get("replay_stats") or r.get("naive_stats") or {}
        mode = r["mode"]
        np_, ns_ = s.get("option_steps"), s.get("option_starts")
        rp = s.get("replan_steps")
        sb = s.get("start_blocked", 0)
        ov = s.get("override_count", 0)
        tr = s.get("term_reasons") or {}
        tag = []
        if sb and sb > 0:
            tag.append("G1✗start_blocked=%d" % sb)
        if mode == "e9":
            if (np_ or 0) != 0:
                tag.append("G2✗e9 却有 option_steps=%s" % np_)
        elif mode == "fixed":
            if not np_ or np_ <= 0:
                tag.append("G2✗fixed option_steps=%s" % np_)
            if rp:
                tag.append("G2✗fixed replan_steps=%s (应为 0)" % rp)
        else:
            if not np_ or np_ <= 0:
                tag.append("G2✗%s option_steps=%s" % (mode, np_))
            if not rp:
                tag.append("G2✗%s replan_steps=%s (应 ≈ steps)" % (mode, rp))
        if mode != "e9" and ns_ == 1:
            tag.append("G3✗option_starts==1 (一个 option 锁死)")
        lines.append("  %-30s mode=%-18s starts=%-4s steps=%-4s replan=%-4s "
                     "ovr=%-3s term=%s  %s"
                     % (name, mode, ns_, np_, rp, ov, tr if tr else "-",
                        " ".join(tag) if tag else "✓"))
        if tag:
            bad.append(name)
    return (len(bad) == 0), lines, bad


def summarize(arms):
    from collections import defaultdict
    G = defaultdict(list)
    for name, r in arms.items():
        G[r["mode"]].append(r)
    out = []
    for mode in ("e9", "fixed", "goal", "goal_term", "goal_term_override"):
        g = G.get(mode)
        if not g:
            continue
        at = [r["replay"]["any_time_acc"] for r in g if "replay" in r
              and "any_time_acc" in r["replay"]]
        fn = [r["replay"]["final_mean_acc"] for r in g if "replay" in r
              and "final_mean_acc" in r["replay"]]
        if not at:
            continue
        # G4: 逐位相同检测
        uniq = len(set(round(x, 12) for x in at))
        out.append(dict(mode=mode, n=len(at), any_time=st.mean(at),
                        any_time_sd=(st.stdev(at) if len(at) > 1 else 0.0),
                        final=st.mean(fn) if fn else float("nan"),
                        uniq=uniq))
    return out


def welch(a, b):
    """Welch t 检验(不假设等方差)。返回 (t, p, df)。"""
    import math
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (float("nan"), float("nan"), 0)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0, na + nb - 2)
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 双侧 p 的近似(用正态近似, 大 n 下足够; 诚实标注这一点)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return (t, p, df)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    arms = load(root)
    if not arms:
        print("没找到 oak_k1v2_* 的结果(实验还在跑?)")
        return 1

    print("=" * 96)
    print("第 ① 步: 机制生效闸门(不过闸门就不给结论)")
    print("=" * 96)
    ok, lines, bad = gate(arms)
    for l in lines:
        print(l)
    print("\n  闸门结论: %s" % ("✓ 全部通过, 可以对照" if ok else
                                "✗ %d 臂未通过 -> 本轮对照**无效**, 只列诊断" % len(bad)))

    print("\n" + "=" * 96)
    print("第 ② 步: 指标汇总")
    print("=" * 96)
    summ = summarize(arms)
    print("  %-22s %3s %10s %10s %10s %6s" % ("模式", "n", "any-time", "std", "final", "唯一值"))
    for s in summ:
        flag = "" if s["uniq"] > 1 or s["n"] == 1 else "  ← G4✗ 所有 seed 逐位相同!"
        print("  %-22s %3d %10.4f %10.4f %10.4f %6d%s"
              % (s["mode"], s["n"], s["any_time"], s["any_time_sd"], s["final"], s["uniq"], flag))

    if not ok:
        print("\n  ⚠ 有臂未过闸门 —— 不给出 O1/O2/O3/O4 的对照结论。")
        return 1

    print("\n" + "=" * 96)
    print("第 ③ 步: 对照(判据: O2 > O1 -> 问题是 open-loop; O3/O4 >= O2 -> 需要自适应终止)")
    print("=" * 96)
    def col(mode):
        return [r["replay"]["any_time_acc"] for r in arms.values()
                if r["mode"] == mode and "replay" in r and "any_time_acc" in r["replay"]]
    base = col("e9")
    print("  %-22s %10s   %s" % ("对比", "Δ(any-time)", "Welch t / p"))
    for mode in ("fixed", "goal", "goal_term", "goal_term_override"):
        v = col(mode)
        if not v or not base:
            continue
        t, p, df = welch(v, base)
        d = st.mean(v) - st.mean(base) if v and base else float("nan")
        verdict = ("**显著**" if p == p and p < 0.05 else "不显著")
        print("  %-22s %+10.4f   t=%+6.2f  p=%.4f   %s"
              % ("%s vs e9" % mode, d, t, p, verdict))
    o1, o2 = col("fixed"), col("goal")
    if o1 and o2:
        t, p, _ = welch(o2, o1)
        print("\n  ★ 关键判据 O2(闭环) vs O1(开环): Δ=%+.4f  t=%+.2f  p=%.4f -> %s"
              % (st.mean(o2) - st.mean(o1), t, p,
                 "O2 更好 ✓ (问题是开环执行)" if (p < 0.05 and st.mean(o2) > st.mean(o1))
                 else "O2 未显著更好 ✗"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
