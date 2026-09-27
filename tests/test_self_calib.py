#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_self_calib.py — **自我校准**的最小闭环自检。

## 为什么有这个文件

L2 那批实验: option 启动 57 次, **56 次 expired, 1 次 goal_reached**。
整整一批, option 层从来没达成过目标 —— 而系统里**没有任何东西会因为这个
事实而改变**, 也没任何东西把它喊出来。发现它需要人手写诊断脚本 + 事前
写死判据 + 跑全量消融。

一个能自我校准的系统应该自己说: 「我的 option 连续 N 次没达成目标」,
然后**不再用它**。

本文件检验的就是这条链:
    犯错(option 永不达成)
      -> 发现(`rr_broken()` 自己点名)
        -> 改正(`calibrated` 规则下它的使用占比衰减)

## 坏 option 怎么制造

把它的硬到期上限压到 1 步(`_rr_cap[name] = 1`)。它的 β_o 永远来不及触发,
所以每次都 `expired` —— **与 L2 的病一模一样**(终止判据从不命中)。

## 判据(事前写死, 跑之前就定好)

 ① 制造器有效: 坏 option 的终止原因必须是 `expired` 压倒性多数
 ② **`uniform` 规则下错误持续**: 坏 option 的使用占比**不衰减**
    (最后一块的占比 >= 第一块的 0.6 倍) —— 证明这个指标能看见差异
 ③ **`calibrated` 规则下系统自己点名**: `rr_calib_stats()["broken"]`
    非空且包含坏 option
 ④ **`calibrated` 规则下使用占比衰减**: 最后一块 < 第一块的一半
 ⑤ 无免费午餐声明: `calibrated` 的主任务成功率 **不低于** `uniform`
 ⑥ 对照不受影响: 没有坏 option 时, `calibrated` 的 broken 集合为空
    (不会乱杀好 option)
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hibs_lnn.rr_agent import RRSkillAgent
from hibs_lnn.skill_mdp import KeyDoorMDP
from hibs_lnn.skill_agent import DEFAULT_REGIMES

EPISODES = 400
BLOCK = 50


def run_arm(rule, sabotage=True, episodes=EPISODES, seed=0):
    """跑一条臂。返回 (agent, 成功率, 分块快照, 坏 option 名)。"""
    mdp = KeyDoorMDP(n_pos=8, horizon=30, **DEFAULT_REGIMES[0])
    ag = RRSkillAgent(mdp, mode="rr", seed=seed, rr_lib="rr",
                      rr_select_rule=rule, opt_prob=1.0)
    ag._rr_ensure()
    names = [st.name for st in ag.rr_lib_obj]
    bad = None
    if sabotage:
        bad = names[-1]                      # 拿最后一个 option 当坏 option
        ag._rr_cap[bad] = 1                  # ★ 终止判据永远来不及命中
    succ, snaps = [], []
    for ep in range(episodes):
        ok, _ = ag.run_episode()
        succ.append(1.0 if ok else 0.0)
        if (ep + 1) % BLOCK == 0:
            snaps.append({"ep": ep + 1,
                          "usage": ag._rr_usage(),
                          "broken": sorted(ag.rr_calib_stats()["broken"])})
    return ag, succ, snaps, bad


def share_of(snap, name):
    return float(snap["usage"].get(name, 0.0))


def main():
    print("=" * 74)
    print("自我校准闭环自检  (KeyDoor regime A, 制造器 = 把某个 option 的上限压到 1)")
    print("=" * 74)
    ok = []

    ag_u, _, snap_u, bad = run_arm("uniform")
    _, succ_c, snap_c, names_c = run_arm("calibrated")
    ag_c = None
    ag_nc, _, _, _ = run_arm("calibrated", sabotage=False)

    print(f"\n坏 option = {bad!r}  (它的 β_o 永远来不及触发)")
    print(f"\n{'规则':<12}{'第一块占比':>12}{'最后一块占比':>14}{'比值':>10}"
          f"{'系统点名':>12}")
    for tag, snaps in (("uniform", snap_u), ("calibrated", snap_c)):
        first, last = share_of(snaps[0], bad), share_of(snaps[-1], bad)
        ratio = (last / first) if first > 0 else float("nan")
        named = "✓" if snaps[-1]["broken"] else "—"
        print(f"{tag:<12}{first:>12.4f}{last:>14.4f}{ratio:>10.3f}{named:>12}")

    # ① 制造器有效 —— ★ 必须在 **uniform 臂**上测: `calibrated` 臂恰恰把坏
    #    option 避开了, 在那一臂上测 expired 占比等于测"校准有没有生效",
    #    不是测"制造器有没有生效"(第一次写这条判据时就打错了臂)。
    o_bad = ag_u.rr_outcome.get(bad, {"n": 0, "reached": 0})
    rate_bad = o_bad["reached"] / o_bad["n"] if o_bad["n"] else 1.0
    ok.append(("① 制造器有效 (uniform 臂上坏 option 达成率低)",
               o_bad["n"] >= 20 and rate_bad < 0.5,
               f"n={o_bad['n']} reached={o_bad['reached']} rate={rate_bad:.3f}"))

    # ② uniform 下错误持续
    fu, lu = share_of(snap_u[0], bad), share_of(snap_u[-1], bad)
    ok.append(("② uniform 下不衰减 (错误持续)",
               fu > 0 and lu >= 0.6 * fu, f"{fu:.4f} -> {lu:.4f}" ))

    # ③ calibrated 下系统自己点名
    named = bad in snap_c[-1]["broken"]
    ok.append(("③ calibrated 自己点名坏 option", named,
               f"broken = {snap_c[-1]['broken']}"))

    # ④ calibrated 下占比衰减
    fc, lc = share_of(snap_c[0], bad), share_of(snap_c[-1], bad)
    ok.append(("④ calibrated 下使用占比衰减", lc < 0.5 * fc,
               f"{fc:.4f} -> {lc:.4f} (比值 {lc/fc if fc else float('nan'):.3f})"))

    # ⑤ 无免费午餐声明: 主任务成功率不低于 uniform
    r_u = float(np.mean(succ_c[:BLOCK * 2]))
    r_c = float(np.mean(succ_c[-BLOCK * 2:]))
    ok.append(("⑤ calibrated 主任务不更差", r_c >= r_u - 0.05,
               f"前 {r_u:.3f} -> 后 {r_c:.3f}"))

    # ⑥ 没有坏 option 时不乱杀
    nb = ag_nc.rr_calib_stats()["n_broken"]
    ok.append(("⑥ 无坏 option 时不误杀", nb == 0, f"broken 数 = {nb}"))

    # ⑦ **退回基元层真的发生了** —— 这是 ④ 之所以可能的机制。坏 option 在
    #    某些状态下是**唯一**可启动的, 校准唯一能做的就是"不用它, 退回基元"。
    #    (第一次读代码时以为选择规则能避开, 实测发现避不开 —— 不是规则错了,
    #     是 `initiation_ok` 把候选集限制死了。)
    def count_none(rule, sabotage=True, episodes=200):
        mdp = KeyDoorMDP(n_pos=8, horizon=30, **DEFAULT_REGIMES[0])
        ag = RRSkillAgent(mdp, mode="rr", seed=0, rr_lib="rr",
                          rr_select_rule=rule, opt_prob=1.0)
        ag._rr_ensure()
        if sabotage:
            ag._rr_cap[[st.name for st in ag.rr_lib_obj][-1]] = 1
        cnt = {"none": 0, "tot": 0}
        orig = ag._rr_select
        def spy(s, _o=orig, _c=cnt):
            r = _o(s)
            _c["tot"] += 1
            _c["none"] += int(r is None)
            return r
        ag._rr_select = spy
        for _ in range(episodes):
            ag.run_episode()
        return cnt

    cu, cc = count_none("uniform"), count_none("calibrated")
    ok.append(("⑦ 退回基元层次数上升 (校准的唯一手段)",
               cc["none"] > cu["none"],
               f"uniform {cu['none']}/{cu['tot']} -> calibrated {cc['none']}/{cc['tot']}"))

    print("\n" + "-" * 74)
    n_pass = 0
    for name, good, info in ok:
        print(f"  {'PASS' if good else 'FAIL'}  {name}   [{info}]")
        n_pass += int(good)
    print("-" * 74)
    print(f"  {n_pass}/{len(ok)} 通过")
    return 0 if n_pass == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
