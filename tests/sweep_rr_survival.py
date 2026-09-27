#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_rr_survival.py — **阶段三: 单因素生存边界扫描**。

## 目标(leo 的定义)

**不是找"最优参数"**, 是画出机制边界:

    安全区间 / 危险区间 / 默认值落在哪里

做法: 每次只动一个参数, 其余全部用**冻结契约**(`docs/calib_baseline_freeze.md`),
三场景(normal / inert / harm)各跑一遍, 按 leo 的六类失效归类。

## 六类失效(leo 的表)

    未知误判为坏   inert 被淘汰 / unknown -> retired
    坏 option 存活  harm 检测延迟过长 / 故障段成功率不恢复
    宽恕过度       retired option 被频繁放回但恢复不了
    学不动         regime 切换后 unknown 长期不下降
    过度保守       几乎所有 option 停在 unknown
    过度激进       未知 option 大量探索, 主任务步数恶化
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, "/Users/apple/Downloads/headless")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_three_state import FROZEN, SCENARIOS, run_arm, ARMS  # noqa: E402

# ── 扫描表(单因素; 其余 = 冻结契约)──────────────────────────────────
SWEEP = {
    "rr_lr_v":        [0.05, 0.2, 0.5],
    "rr_starve_steps": [5, 15, 30, 90, 300, 1000],  # 0.17× ~ 33× horizon
    "rr_unk_p":       [0.0, 0.5, 1.0],
    "rr_min_launches": [3, 6, 12],
    "rr_ucb_c":       [0.5, 1.0, 2.0],
    "rr_forgive_p":   [0.0, 0.02, 0.10, 0.30],
    "rr_forget_hl":   [float("inf"), 200, 50, 20],
}
LABEL = {"rr_lr_v": "rr_lr_v", "rr_starve_steps": "缓存阈值(步)",
         "rr_unk_p": "未知探索率", "rr_min_launches": "最小证据",
         "rr_ucb_c": "UCB 系数", "rr_forgive_p": "宽恕率",
         "rr_forget_hl": "遗忘半衰期"}


def one(cfg, scen, episodes=300, seeds=1):
    """跑一个配置 × 一个场景, 返回末段/故障段的汇总指标。

    注意数据结构: `run_arm` 返回的 `segs` 是**段列表**; 多 seed 时
    `per_seed` 是 `list[list[dict]]`(每个 seed 一个段列表)。所有汇总都要
    **先按段索引跨 seed 聚合**, 不能把两层当一层。
    """
    runs = [run_arm(cfg, scen, episodes, sd) for sd in range(seeds)]
    per_seed: list = [r[0] for r in runs]          # list[list[dict]]
    nseg = len(per_seed[0])
    fl = [i for i in range(nseg) if i >= SCENARIOS[scen]["fault_at"]]

    def av(i, key, fn=np.mean):
        return float(fn([per_seed[k][i][key] for k in range(seeds)]))

    def st(i, which):
        return float(np.mean([per_seed[k][i]["states"][which] for k in range(seeds)]))

    ti = nseg - 1
    dets = [per_seed[k][i]["detect"] for k in range(seeds) for i in fl
            if per_seed[k][i]["detect"] is not None]
    return dict(
        detect=float(np.mean(dets)) if dets else None,
        detected=len(dets) > 0,
        false_kill=float(sum(av(i, "false_kill", np.sum) for i in fl)) if fl else 0.0,
        recovered=float(np.mean([av(i, "recovered") for i in fl])) if fl else 0.0,
        fallback=float(np.mean([av(i, "fallback") for i in fl])) if fl else 0.0,
        unknown=st(ti, "unknown"), good=st(ti, "good"), bad=st(ti, "bad"),
        succ_fault=(float(np.mean([av(i, "succ") for i in fl])) if fl else 0.0),
        steps_all=float(np.nanmean([av(i, "steps_go") for i in range(nseg)])),
        succ_all=float(np.mean([av(i, "succ") for i in range(nseg)])),
        ev_last=av(ti, "evidence"), ev_first=av(0, "evidence"),
        starved_lies=int(sum(1 for k in range(seeds) for i in range(nseg)
                             if per_seed[k][i]["starved"]
                             and per_seed[k][i]["states"]["good"] > 0)),
        # ★ 门**实际开火的频率** —— 判据"安全"必须区分两件事:
        #   "参数取这个值没事" vs "这个门根本没被触发过"。
        #   前者是边界, 后者是空转。不报这个数就会把空转读成安全。
        starved_frac=float(np.mean([1.0 if per_seed[k][i]["starved"] else 0.0
                                    for k in range(seeds)
                                    for i in range(nseg)])),
        usage_fault=float(np.mean([av(i, "usage") for i in fl])) if fl else 0.0,
    )


def classify(scen, r):
    """按 leo 的六类失效归类 —— **必须按场景**, 不能一套判据打三个台。

    第一版把 `detected is False` 无条件当成"坏 option 长期存活", 于是
    normal / inert(本来就没有 harm 故障)全被误标。场景不同, "失效"的定义
    也不同, 混用会把边界扫描的结果全部污染。
    """
    tags = []
    if scen == "normal":
        # 无故障: 目标应该被保留为"好"
        if r["good"] == 0 and r["unknown"] > 0:
            tags.append("过度保守(正常 option 停在未知)")
        if r["good"] == 0 and r["unknown"] == 0:
            tags.append("正常 option 被判坏")
        if r["succ_all"] < 0.8:
            tags.append("主任务恶化")
    elif scen == "inert":
        # 惰性故障: 看着差但无害, **不应误杀**
        if r["false_kill"] > 0:
            tags.append("未知误判为坏(inert 被淘汰)")
        if r["succ_fault"] < 0.8:
            tags.append("主任务恶化")
    else:                                   # harm: 真有害, 应检出且能自救
        if not r["detected"]:
            tags.append("坏 option 长期存活(未检测)")
        elif r["detect"] > 150:
            tags.append("坏 option 长期存活(检测过慢)")
        if r["false_kill"] > 0:
            tags.append("误杀健康 option")
        if r["succ_fault"] < 0.5:
            tags.append("未能自救")
        if r["recovered"] >= 2 and r["succ_fault"] < 0.3:
            tags.append("宽恕过度")
        if r["usage_fault"] > 0.5 and r["bad"] > 0 and r["succ_fault"] < 0.5:
            tags.append("过度激进(判坏后仍在用)")
    if r["starved_lies"]:
        tags.append("自欺(奖励流空仍报好)")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=",".join(SWEEP.keys()))
    ap.add_argument("--episodes-per", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params = [p for p in a.params.split(",") if p in SWEEP]

    allres = {}
    for p in params:
        print("\n" + "=" * 104)
        print(f"扫描 {LABEL[p]}   (其余 = 冻结契约; {a.seeds} seed × {a.episodes_per} 回合)")
        print("=" * 104)
        print(f"  {p:>14}{'场景':>8}{'检测':>7}{'误杀':>6}{'恢复':>6}{'fallback':>10}"
              f"{'未/好/坏':>11}{'故障成功':>9}{'步数':>8}{'自欺':>6}   失效类型")
        rows = {}
        for v in SWEEP[p]:
            cfg = dict(FROZEN, rr_three_state=True, **{p: v})
            rs = {}
            for sc in ("normal", "inert", "harm"):
                r = one(cfg, sc, a.episodes_per, a.seeds)
                rs[sc] = r
                tags = classify(sc, r)
                dv = f"{r['detect']:.0f}" if r["detect"] is not None else (
                    "未检测" if sc == "harm" else "—")
                st = f"{r['unknown']:.0f}/{r['good']:.0f}/{r['bad']:.0f}"
                print(f"  {str(v):>14}{sc:>8}{dv:>7}{r['false_kill']:>6.0f}"
                      f"{r['recovered']:>6.1f}{r['fallback']:>10.3f}{st:>11}"
                      f"{r['succ_fault']:>9.3f}{r['steps_all']:>8.2f}"
                      f"{r['starved_lies']:>6}   {'/'.join(tags) if tags else '—'}")
            rows[str(v)] = rs
        allres[p] = rows

    # ── 汇总: 每个参数的"安全 / 危险"区间 ─────────────────────────
    print("\n" + "=" * 104)
    print("边界汇总 (安全 = 三场景都无失效标签且自欺=0)")
    print("=" * 104)
    for p in params:
        safe, bad = [], []
        for v, rs in allres[p].items():
            ts = set()
            for sc in ("normal", "inert", "harm"):
                ts |= set(classify(sc, rs[sc]))
                if rs[sc]["starved_lies"]:
                    ts.add("自欺")
            if ts:
                bad.append((v, "/".join(sorted(ts))))
            else:
                safe.append(v)
        print(f"  {LABEL[p]:>14}  安全={safe if safe else '（无）'}")
        for v, t in bad:
            print(f"  {'':>14}  危险 {v}: {t}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(allres, indent=2, default=str,
                                          ensure_ascii=False))
        print(f"\n原始数据: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
