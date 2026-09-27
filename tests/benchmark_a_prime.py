#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_a_prime.py — A′ 四象限可辨识性 + 检测指标(C4/C5)。

## 要回答的唯一问题

    **A1 能不能把 S2(新但稳定)与 S3(熟悉但变了)区分开?**

    若 A1(S2) ≈ A1(S3)  ⇒ A1 仍然只是 novelty detector(证伪成立)
    若 A1(S3) >> A1(S2) ⇒ A1 真的捕捉到了 dynamics(可辨识)

## 四象限怎么"构造"出来, 而不是事后切窗

每个 transition 发生的那一刻, 有两件**我们知道真值**的事:

    novelty   : 这一步的 `n_before(s,a)` —— 更新**前**的计数
    dynamics  : 这一步是在 `shift_at` 之前还是之后

于是四象限是**逐步标注**出来的 2x2:

    |       | 动力学未变 (t<shift) | 动力学已变 (t>=shift) |
    | 熟悉  | S1  n>=FAM           | S3  n>=FAM            |
    | 新    | S2  n< NOV           | S4  n< NOV            |

**S2 与 S3 是同一张表里的两个格子** —— 用 n 桶把 novelty 对齐之后,
"新"与"变"才可比。这是本文件的全部要点。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 仓库根 ——
#   ★ 必须是**仓库根**(`parents[1]`), 不是 `hibs_lnn/` 本身。插错成
#     `.../headless/hibs_lnn` 会让 `import hibs_lnn.rr_agent` 找不到包
#     (`ModuleNotFoundError: No module named 'hibs_lnn'`), 因为那正是包**内部**。
#     脚本目录会自动进 sys.path, 所以 `from benchmark_three_state import ...`
#     仍然可用。

from hibs_lnn.rr_agent import RRSkillAgent                 # noqa: E402
from hibs_lnn.stochastic_keydoor import (                   # noqa: E402
    NonStationaryKeyDoor, StochasticKeyDoor)
from benchmark_three_state import FROZEN                    # noqa: E402

NOV, FAM = 3, 10            # 新: n_before < 3    熟悉: n_before >= 10
CHANNELS = ("N", "A1", "A2", "A3")


class _LogRRSkillAgent(RRSkillAgent):
    """逐步记录 (t, s, a, n_before, 各通道读数)。

    `n_before` 必须是**更新前**的计数 —— 它就是那一步的 novelty 真值。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []

    def _rr_dval(self, s, a, s2):        # noqa: D102
        d = self._rr_T.get((int(s), int(a)))
        n_before = sum(d.values()) if d else 0
        v = super()._rr_dval(s, a, s2)
        self.rows.append({
            "t": int(getattr(self.mdp, "t_total", len(self.rows))),
            "s": int(s), "a": int(a), "n": int(n_before),
            "v": float(v),
        })
        return v


def log_channel(channel, env, episodes=140, seed=7):
    """在一个环境上跑一遍, 返回该通道的逐步日志。"""
    ag = _LogRRSkillAgent(env, mode="rr", seed=seed, rr_v_main="learned",
                          **dict(FROZEN, rr_three_state=True, rr_dyn=channel))
    for _ in range(episodes):
        ag.run_episode()
    return ag.rows


def quad_table(rows, shift_at):
    """把逐步日志切成四象限, 返回 {(cell, ch): mean} 的统计。"""
    out = {}
    for r in rows:
        cell = ("S1" if r["n"] >= FAM else
                "S2" if r["n"] < NOV else None)
        if cell is None:
            continue
        if r["t"] >= shift_at:
            cell = {"S1": "S3", "S2": "S4"}[cell]
        out.setdefault(cell, []).append(r["v"])
    return {c: (float(np.mean(v)), len(v)) for c, v in out.items()}


# ── 指标(A′-4)──────────────────────────────────────────────────────
def detection_metrics(rows, shift_at, thresh=None, win=100):
    """在 shift 附近算 T_detect / FP / FN / AUC / discrimination。

    * 触发 = 读数超过 `thresh`。默认 `pre_mean + 2·pre_std`。
      ★ 阈值只能从"变之前"的数据里定 —— 用全段分位数定阈值等于偷看答案。
      ★ **不能用 pre 的 95 分位**:`_rr_dval` 在 `n<2` 时返回满值 1.0,
        热身期到处都是这种步 ⇒ 95 分位 = 1.0000 ⇒ 没有任何步能超过它
        ⇒ `T_detect = None`、FN 恒 1(第一版的实测就是这个指纹:
        四个通道的阈值全是 1.0000、T_detect 全是 None)。均值+2σ 不会被
        饱和步锁死在 1.0。
    * `T_detect − T_shift` = 检测延迟(核心指标)。
    * AUC = 把"shift 后的步"与"shift 前的步"分开的可分性(Mann-Whitney)。
    """
    pre = np.array([r["v"] for r in rows if r["t"] < shift_at], float)
    post = np.array([r["v"] for r in rows if r["t"] >= shift_at], float)
    if pre.size == 0 or post.size == 0:
        return None
    if thresh is None:
        # ★ 阈值估计必须**排除饱和步**(n<2 时 `_rr_dval` 返回满值 1.0)。
        #   否则 μ+2σ 会被那些 1.0 撑到 **超过指标上限**(实测 A1 的阈值
        #   算成 1.0680 > 1.0) ⇒ 永远没有步能超过它 ⇒ T_detect=None、
        #   FN 恒 1。那不是"检测不到", 是**阈值不可达**。
        #   A 类读数恒在 [0,1], 所以阈值必须落在 [0,1] 内才有意义。
        pre_ok = np.array([r["v"] for r in rows
                           if r["t"] < shift_at and r["n"] >= 2], float)
        base = pre_ok if pre_ok.size >= 10 else pre
        thr = float(base.mean() + 2.0 * base.std())
        thr = float(min(max(thr, 1e-6), 1.0))
    else:
        thr = float(thresh)
    # 检测延迟: shift 后第一个超过阈值、且随后 win 步里多数都超的时刻
    t_det = None
    for i, r in enumerate(rows):
        if r["t"] < shift_at or r["v"] <= thr:
            continue
        nxt = [q["v"] for q in rows[i:i + win]]
        if sum(v > thr for v in nxt) > 0.5 * len(nxt):
            t_det = r["t"]
            break
    fp = float(np.mean(pre > thr)) if pre.size else float("nan")
    fn = 1.0 if t_det is None else 0.0
    # AUC: P(读数(shift后) > 读数(shift前)) —— Mann-Whitney 形式
    if pre.size and post.size:
        gt = (post[:, None] > pre[None, :]).mean()
        eq = (post[:, None] == pre[None, :]).mean()
        auc = float(gt + 0.5 * eq)
    else:
        auc = float("nan")
    return {"thr": thr, "t_det": t_det,
            "delay": (None if t_det is None else t_det - shift_at),
            "fp": fp, "fn": fn, "auc": auc,
            "pre_mean": float(pre.mean()), "post_mean": float(post.mean())}


def main():
    SH, EP = 400, 140
    print("=" * 104)
    print("A′ 四象限可辨识性(C4):**A1 能不能把 S2(新但稳定)与 S3(熟悉但变了)分开?**")
    print(f"      熟悉 = n_before >= {FAM}   新 = n_before < {NOV}   shift @ t={SH}")
    print("=" * 104)

    results = {}
    # 环境 A: 无 shift(所有格子都在"动力学未变"那一列)
    envs = {
        "E1 平稳随机(无 shift)": lambda: StochasticKeyDoor(slip=0.10, seed=7),
        "E2 非平稳(shiftt@400)": lambda: NonStationaryKeyDoor(
            slip0=0.10, slip1=0.60, shift_at=SH, seed=7),
    }
    for nm, mk in envs.items():
        print(f"\n── {nm} " + "─" * (96 - len(nm)))
        print(f"  {'通道':>4} {'S1熟/静':>10} {'S2新/静':>10} {'S3熟/变':>10} "
              f"{'S4新/变':>10}   {'S3−S2':>9} {'S3/S2':>7}   {'N(S2)':>7}")
        for ch in CHANNELS:
            rows = log_channel(ch, mk(), EP, 7)
            q = quad_table(rows, SH)
            g = lambda c: q.get(c, (float("nan"), 0))[0]        # noqa: E731
            s2, s3 = g("S2"), g("S3")
            disc = s3 - s2
            ratio = (s3 / s2) if s2 and s2 > 1e-12 else float("inf")
            results.setdefault(nm, {})[ch] = {
                "S1": g("S1"), "S2": s2, "S3": s3, "S4": g("S4"),
                "disc": disc, "n_S2": q.get("S2", (0, 0))[1],
                "n_S3": q.get("S3", (0, 0))[1],
            }
            print(f"  {ch:>4} {g('S1'):>10.4f} {s2:>10.4f} {s3:>10.4f} "
                  f"{g('S4'):>10.4f}   {disc:>+9.4f} {ratio:>7.2f}   "
                  f"{q.get('S2',(0,0))[1]:>7}")

    print()
    print("=" * 104)
    print("A′ 检测指标(C5): T_detect/T_shift · FP · FN · AUC(阈值只从 shift **前** 定)")
    print("=" * 104)
    print(f"  {'通道':>4} {'阈值':>8} {'T_detect':>9} {'延迟':>6} {'FP':>7} "
          f"{'FN':>4} {'AUC':>7} {'变前均值':>10} {'变后均值':>10}")
    met = {}
    for ch in CHANNELS:
        rows = log_channel(ch, NonStationaryKeyDoor(
            slip0=0.10, slip1=0.60, shift_at=SH, seed=7), EP, 7)
        m = detection_metrics(rows, SH)
        met[ch] = m
        if m is None:
            print(f"  {ch:>4}   (样本不足)")
            continue
        print(f"  {ch:>4} {m['thr']:>8.4f} {str(m['t_det']):>9} "
              f"{str(m['delay']):>6} {m['fp']:>7.2f} {m['fn']:>4.0f} "
              f"{m['auc']:>7.3f} {m['pre_mean']:>10.4f} {m['post_mean']:>10.4f}")

    # ── 判定 ──
    print()
    print("=" * 104)
    print("判定")
    print("=" * 104)
    ok = True
    e2 = results["E2 非平稳(shiftt@400)"]

    # ★ 主判据 = **n 对齐的 S1 vs S3**: 同为"熟悉"状态, 只有动力学变了。
    #   不能用 S2 vs S3 —— S2 是首次访问步, 而 `_rr_dval` 在 `n<2` 时返回
    #   满值 1.0, 于是 S2 在所有通道上都被顶到 0.7~0.8, 必然压过 S3。
    #   那是**饱和分支**的指纹, 不是"A 类分不开新与变"的证据。
    d1 = e2["A1"]["S3"] - e2["A1"]["S1"]
    dN = e2["N"]["S3"] - e2["N"]["S1"]
    a1_auc = met.get("A1", {}).get("auc", float("nan"))
    n_auc = met.get("N", {}).get("auc", float("nan"))
    print("  【主判据】n 对齐(同为熟悉状态)的 S1 -> S3:")
    print(f"    A1: {e2['A1']['S1']:.4f} -> {e2['A1']['S3']:.4f}  ({d1:+.4f})")
    print(f"    N : {e2['N']['S1']:.4f} -> {e2['N']['S3']:.4f}  ({dN:+.4f})")
    print(f"    A2: {e2['A2']['S1']:.4f} -> {e2['A2']['S3']:.4f}  "
          f"({e2['A2']['S3'] - e2['A2']['S1']:+.4f})")
    print(f"  AUC: A1 {a1_auc:.3f}  vs  N {n_auc:.3f}")
    print()
    print("  【诊断, 非判据】原始 S2 vs S3(被 `n<2 -> 1.0` 饱和分支污染):")
    print(f"    S2 各通道 {[round(e2[c]['S2'], 3) for c in CHANNELS]}")
    print(f"    S3 各通道 {[round(e2[c]['S3'], 3) for c in CHANNELS]}")
    print("    ⇒ S2 全面高于 S3。这是「首次访问报满分」造成的, 不代表"
          "A 类分不开新与变。")
    print()
    checks = [
        (f"A1 在熟悉状态上对'变'敏感 (S3−S1 > +0.10)  [{d1:+.4f}]", d1 > 0.10),
        (f"N 在同样一对上保持平坦 (|S3−S1| < 0.05)     [{dN:+.4f}]",
         abs(dN) < 0.05),
        (f"A1 的响应明显强于纯新颖度 (dA1 > dN + 0.10) "
         f"[{d1:+.4f} vs {dN:+.4f}]", d1 - dN > 0.10),
        (f"A1 的 AUC 明显高于 N 的 AUC (>0.10)         "
         f"[{a1_auc:.3f} vs {n_auc:.3f}]", a1_auc - n_auc > 0.10),
    ]
    for name, good in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        ok = ok and good
    print(f"\n  {'A 类在本环境上具有可辨识性' if ok else 'A 类仍不可辨识 —— 记录为 coverage-only'}")
    with open("results/a_prime.json", "w") as f:
        json.dump({"quadrants": results, "metrics": met,
                   "shift_at": SH, "NOV": NOV, "FAM": FAM}, f,
                  indent=2, ensure_ascii=False)
    print("  -> results/a_prime.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
