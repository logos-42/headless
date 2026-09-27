#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_rr_calib_chain.py — **持续自我校准**在 regime 链上的消融。

## 为什么需要这个文件

`tests/test_self_calib.py` 证明了单 regime 内的闭环(检测 -> 淘汰 -> 退回基元层)。
但「**持续**自我校准」的"持续"还没被测过:

  ① **持久化**: 台账跨 regime 存活吗? 还是每换一个世界就清零?
  ② **遗忘**: 持久化如果配**不遗忘**, n 会累积到几百 -> 老的成功把新的失败
     稀释掉, UCB 探索项也被压死 -> **陈旧的判断被永久锁住**。
     "持续校准"要跟踪的是**当前胜任度**, 不是历史平均。
  ③ **校准到哪个信号上**: option 自己的达成率, 还是主任务的推进量?

## 两类故障 —— 这是能分辨的关键

如果链上所有 option 都是健康的, 校准就**无事可做**。所以注入一个
**随 regime 变化**的故障:

    段 1-3 (A B C):  全部健康        -> agent 对 `rr[door]` 建立**正面**先验
    段 4-6 (A B C):  `rr[door]` 坏掉 -> 看谁**多快**发现并停用它

  **inert(惰性)** —— 硬到期上限压到 1 步。它"达不成自己的子目标"
      (达成率 ~0.2), 但**它执行的那一步仍然是 π_o 的最优步**, 对主任务有益。
      ★ 这是**判据来源的判别器**: 按内部判据淘汰它 -> 白损失好步 -> 变差。
  **harm(有害)** —— 反转 π_o(LEFT<->RIGHT), 主动朝反方向走。
      内部判据与主任务判据**同时**判它坏。淘汰它应当**有益**。

一个能自我校准的系统必须能分开这两者 —— 否则它是在校准到错误的信号上。
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hibs_lnn.rr_agent import RRSkillAgent
from hibs_lnn.skill_mdp import KeyDoorMDP, LEFT, RIGHT
from hibs_lnn.skill_agent import DEFAULT_REGIMES

CHAIN = DEFAULT_REGIMES * 2          # A B C A B C
FAULT_AT = 3                         # 从第 4 段(0 基)起 rr[door] 坏掉
FAULT_OPT = "rr[door]"

ARMS = {
    # 名字                        选择规则           遗忘   淘汰判据      分数
    "uniform":               dict(rr_select_rule="uniform"),
    "internal-hl20":         dict(rr_select_rule="calibrated", rr_forget_hl=20),
    "internal-noforget":     dict(rr_select_rule="calibrated",
                                  rr_forget_hl=float("inf")),
    "internal-nopersist":    dict(rr_select_rule="calibrated", rr_persist=False),
    "adv-hl20":              dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="adv", rr_score="adv"),
    # ★ 势函数质量对照: `rr_pot="exact"` 用模型 VI 的 V_main(干净信号),
    #   用来分离"信号质量差"与"机制结构错"两个因素。
    #   learned 势函数在早期是乐观常数(≈1.0) -> adv≈噪声, 全判负 -> 全杀。
    "advEX-hl20":            dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="adv", rr_score="adv",
                                  rr_pot="exact"),
    "bothEX-hl20":           dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="both", rr_score="adv",
                                  rr_pot="exact"),
    "either-hl20":           dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="either", rr_score="adv"),
    # ══ ★ 对"在线可用势函数"问题的解: option 的**学习价值**(SMDP)══════
    #   不需要任何势函数 —— advantage = Q_opt(s,o) − V_base(s),
    #   两边都从 agent 自己的 Q 表出来, 同一尺度, 同一个初值。
    "advq-hl20":             dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="advq", rr_score="advq"),
    "advq-noforget":         dict(rr_select_rule="calibrated",
                                  rr_forget_hl=float("inf"),
                                  rr_retire_rule="advq", rr_score="advq"),
    "advq-nopersist":        dict(rr_select_rule="calibrated", rr_persist=False,
                                  rr_retire_rule="advq", rr_score="advq"),
    # 对照: 保留旧势函数差分, 但**只换信号源**(advq 判据 + rate 选择)
    "advqselect-hl20":       dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="internal", rr_score="advq"),
    # ══ 诊断: Q_adv 这个信号本身有没有信息量? (只报数, 不淘汰)══════
    #   若健康段 Q_adv > 0 而故障段 < 0, 信号是好的, 问题只在机制;
    #   若两边都是同一符号, 信号本身不可用 —— 这是完全不同的结论。
    "advq-none-hl20":        dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="none", rr_score="advq"),
    # ══ ★ 原谅: 退休可逆 ══════════════════════════════════════════
    "advq-forgive":          dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="advq", rr_score="advq",
                                  rr_forgive_p=0.10),
    # ══ ★★ 在线可用势函数(GVF)════════════════════════════════════════
    #   `rr_pot="gvf"`: 中性初始化(全 0)的状态价值表, TD 从**每个转移**学,
    #   不论那一步是谁出的动作 -> 无 oracle, option 绕过 Q 也不受影响。
    #   实测 Φ ∈ [0, 0.999] 15 个不同值, advq = +0.0495/+0.0356 —— 符号与
    #   `exact`(oracle 对照)一致, 而 `learned` 是 −0.275 的初值 artifact。
    "advqGVF-hl20":          dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="advq", rr_score="advq",
                                  rr_pot="gvf"),
    "advqGVF-noforget":      dict(rr_select_rule="calibrated",
                                  rr_forget_hl=float("inf"),
                                  rr_retire_rule="advq", rr_score="advq",
                                  rr_pot="gvf"),
    "advqGVF-nopersist":     dict(rr_select_rule="calibrated", rr_persist=False,
                                  rr_retire_rule="advq", rr_score="advq",
                                  rr_pot="gvf"),
    "advqGVF-forgive":       dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="advq", rr_score="advq",
                                  rr_pot="gvf", rr_forgive_p=0.10),
    # 对照: 只换势函数来源, 判据仍用内部达成率(分离"势函数"与"判据"两个因素)
    "advqGVFint-hl20":       dict(rr_select_rule="calibrated", rr_forget_hl=20,
                                  rr_retire_rule="internal", rr_score="advq",
                                  rr_pot="gvf"),
}


def apply_fault(ag, kind):
    """把 `FAULT_OPT` 弄坏。返回是否成功。"""
    st = next((x for x in (ag.rr_lib_obj or []) if x.name == FAULT_OPT), None)
    if st is None:
        return False
    if kind == "inert":                  # 达不成子目标, 但每步仍是好步
        ag._rr_cap[FAULT_OPT] = 1
    elif kind == "harm":                 # 主动朝反方向走
        p = np.asarray(st.policy).copy()
        p2 = p.copy()
        p2[p == LEFT] = RIGHT
        p2[p == RIGHT] = LEFT
        st.policy = p2
    else:
        raise ValueError(f"unknown fault: {kind}")
    return True


def run_arm(kw, fault="inert", episodes_per=400, seed=0, thresh=0.8, window=20):
    mdp = KeyDoorMDP(n_pos=8, horizon=30, **CHAIN[0])
    ag = RRSkillAgent(mdp, mode="rr", seed=seed, rr_lib="rr", opt_prob=1.0, **kw)
    segs, n_none, n_sel = [], 0, 0
    for ci, reg in enumerate(CHAIN):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        ag._rr_ensure()
        faulted = ci >= FAULT_AT and apply_fault(ag, fault)
        n0 = ag.rr_outcome.get(FAULT_OPT, {}).get("n", 0)
        broken_at = None
        orig = ag._rr_select

        def spy(s, _o=orig):
            nonlocal n_none, n_sel
            r = _o(s)
            n_sel += 1
            n_none += int(r is None)
            return r

        ag._rr_select = spy
        succ, steps_ok = [], []
        for _ in range(episodes_per):
            ok, st = ag.run_episode()
            succ.append(1.0 if ok else 0.0)
            if ok:
                steps_ok.append(st)
            if faulted and broken_at is None and FAULT_OPT in ag.rr_broken():
                broken_at = ag.rr_outcome[FAULT_OPT]["n"] - n0
        ag._rr_select = orig
        s = np.array(succ)
        t_adapt = episodes_per
        for e in range(window - 1, episodes_per):
            if float(s[e - window + 1:e + 1].mean()) >= thresh:
                t_adapt = e + 1
                break
        o = ag.rr_outcome.get(FAULT_OPT, {})
        segs.append({
            "seg": ci, "tc": ci % 3, "faulted": bool(faulted),
            "t_adapt": int(t_adapt),
            "steps_go": float(np.mean(steps_ok)) if steps_ok else float("nan"),
            "succ": float(s[-window:].mean()),
            "usage": (o.get("n", 0) / max(1, sum(x["n"] for x in ag.rr_outcome.values()))),
            "bad_rate": float(ag._rr_rate(o)) if o else float("nan"),
            "bad_rate_life": (o["reached"] / o["n"]) if o and o["n"] else float("nan"),
            "bad_adv": float(ag._rr_adv(o)) if o else float("nan"),
            "bad_advq": RRSkillAgent._rr_advq_of(o) if o else float("nan"),
            "detect_launches": broken_at,
        })
    return segs, {"none": n_none, "sel": n_sel}, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fault", default="inert", choices=["inert", "harm"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = {k: ARMS[k] for k in a.arms.split(",") if k in ARMS}
    out_path = a.out or f"results/rr_calib_chain_{a.fault}.json"

    chain_s = "/".join(f"{r['key_pos']}{r['door_pos']}{r['goal_pos']}" for r in CHAIN)
    print("=" * 100)
    print(f"持续自我校准 · regime 链消融    故障 = {a.fault}    链(k/d/g) = {chain_s}")
    print(f"故障自第 {FAULT_AT + 1} 段起作用于 `{FAULT_OPT}`  "
          f"(inert=上限压到1步但每步仍是好步 | harm=反转 π_o 朝反方向走)")
    print("=" * 100)

    out = {}
    for name, kw in arms.items():
        runs = [run_arm(kw, a.fault, a.episodes_per, sd) for sd in range(a.seeds)]
        out[name] = runs
        fb = np.mean([r[1]["none"] / max(1, r[1]["sel"]) for r in runs])
        print(f"\n── {name}  (fallback 占比 {fb:.3f}) " + "─" * max(0, 58 - len(name)))
        print(f"   {'段':>3}{'tc':>4}{'故障':>6}{'T_adapt':>9}{'步数':>8}"
              f"{'坏占比':>9}{'当前率':>9}{'终身率':>9}{'主任务adv':>11}"
              f"{'Q_adv':>10}{'检测延迟':>10}")
        for si in range(len(CHAIN)):
            g = [r[0][si] for r in runs]
            dl = [x["detect_launches"] for x in g if x["detect_launches"] is not None]
            dls = f"{np.mean(dl):.0f}" if dl else ("未检测" if g[0]["faulted"] else "—")
            print(f"   {si:>3}{g[0]['tc']:>4}{'是' if g[0]['faulted'] else '否':>6}"
                  f"{np.mean([x['t_adapt'] for x in g]):>9.1f}"
                  f"{np.nanmean([x['steps_go'] for x in g]):>8.2f}"
                  f"{np.mean([x['usage'] for x in g]):>9.4f}"
                  f"{np.nanmean([x['bad_rate'] for x in g]):>9.3f}"
                  f"{np.nanmean([x['bad_rate_life'] for x in g]):>9.3f}"
                  f"{np.nanmean([x['bad_adv'] for x in g]):>11.3f}"
                  f"{np.nanmean([x['bad_advq'] for x in g]):>10.4f}{dls:>10}")

    # ── 事前判据 ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"事前判据  (故障 = {a.fault})")
    print("=" * 100)
    faults = [i for i in range(len(CHAIN)) if i >= FAULT_AT]
    ok = []

    def tail_steps(nm):
        return float(np.nanmean([np.nanmean([r[0][i]["steps_go"] for r in out[nm]])
                                 for i in faults]))

    def detect(nm):
        v = [r[0][FAULT_AT]["detect_launches"] for r in out[nm]]
        return float(np.mean([x if x is not None else a.episodes_per for x in v]))

    def tail_usage(nm):
        return float(np.mean([np.mean([r[0][i]["usage"] for r in out[nm]]) for i in faults]))

    def tail_succ(nm):
        return float(np.mean([np.mean([r[0][i]["succ"] for r in out[nm]]) for i in faults]))

    def fallback(nm):
        return float(np.mean([r[1]["none"] / max(1, r[1]["sel"]) for r in out[nm]]))

    if a.fault == "harm":
        # ★ 用成功率而不是步数: 有害故障下 `uniform` 可能**一个回合都成功不了**
        #   (steps_go = nan), 用步数比较会直接崩在 nan 上。成功率总是有限的。
        ok.append(("① 有害故障: uniform 全面失败而校准臂仍能成功",
                   tail_succ("internal-hl20") > tail_succ("uniform") + 0.05,
                   f"成功率 uniform {tail_succ('uniform'):.3f} -> "
                   f"internal {tail_succ('internal-hl20'):.3f}"))
        ok.append(("② 有害故障被检测到 (internal 坏占比 < 0.6×uniform)",
                   tail_usage("internal-hl20") < 0.6 * tail_usage("uniform"),
                   f"{tail_usage('uniform'):.4f} -> {tail_usage('internal-hl20'):.4f}"))
        if "advEX-hl20" in out:
            ok.append(("②b 干净势函数下主任务判据**也会**开火 (harm 使 adv 转负)",
                       np.nanmean([r[0][FAULT_AT]["bad_adv"] for r in out["advEX-hl20"]]) < 0.0,
                       f"advEX 故障段 adv = "
                       f"{np.nanmean([r[0][FAULT_AT]['bad_adv'] for r in out['advEX-hl20']]):.4f}"))
    else:
        ok.append(("① 惰性故障: 按**内部判据**淘汰反而更差 (internal 步数 > uniform)",
                   tail_steps("internal-hl20") > tail_steps("uniform") + 0.05,
                   f"uniform {tail_steps('uniform'):.2f} -> "
                   f"internal {tail_steps('internal-hl20'):.2f}"))
        if "advEX-hl20" in out:
            # ★ 判别器的核心: 惰性 option 的**内部达成率崩了**(~0.15), 但它的
            #   主任务 advance 仍然**为正** —— 因为它执行的那一步还是好步。
            ok.append(("② 惰性故障: 主任务判据**不开火**, 于是步数与 uniform 一致",
                       abs(tail_steps("advEX-hl20") - tail_steps("uniform")) <= 0.05,
                       f"uniform {tail_steps('uniform'):.2f} vs "
                       f"advEX {tail_steps('advEX-hl20'):.2f}"))
            ok.append(("②b 两条判据在惰性故障上**分离** (内部率崩而 adv 仍正)",
                       (np.nanmean([r[0][FAULT_AT]["bad_rate"] for r in out["advEX-hl20"]])
                        < 0.5 * np.nanmean([r[0][FAULT_AT]["bad_rate"]
                                            for r in out["internal-hl20"]]))
                       and (np.nanmean([r[0][FAULT_AT]["bad_adv"] for r in out["advEX-hl20"]]) > 0),
                       f"advEX: 内部率 "
                       f"{np.nanmean([r[0][FAULT_AT]['bad_rate'] for r in out['advEX-hl20']]):.3f}"
                       f" / adv "
                       f"{np.nanmean([r[0][FAULT_AT]['bad_adv'] for r in out['advEX-hl20']]):.4f}"))
    if "internal-noforget" in out:
        ok.append(("③ 不遗忘的持久化检测更慢 (noforget 延迟 >= hl20)",
                   detect("internal-noforget") >= detect("internal-hl20"),
                   f"noforget {detect('internal-noforget'):.0f} vs "
                   f"hl20 {detect('internal-hl20'):.0f}"))
    if "internal-nopersist" in out:
        ok.append(("④ 不持久化反而检测最快 (nopersist <= hl20 + 1)",
                   detect("internal-nopersist") <= detect("internal-hl20") + 1,
                   f"nopersist {detect('internal-nopersist'):.0f} vs "
                   f"hl20 {detect('internal-hl20'):.0f}"))
    ok.append(("⑤ 校准臂退回基元层更多", fallback("internal-hl20") > fallback("uniform"),
               f"uniform {fallback('uniform'):.3f} -> "
               f"internal {fallback('internal-hl20'):.3f}"))

    n_pass = 0
    for nm, good, info in ok:
        print(f"  {'PASS' if good else 'FAIL'}  {nm}   [{info}]")
        n_pass += int(good)
    print(f"\n  {n_pass}/{len(ok)} 通过")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {k: [{"segs": r[0], "calls": r[1]} for r in v] for k, v in out.items()},
        ensure_ascii=False, indent=1))
    print(f"\n原始数据: {out_path}")
    return 0 if n_pass == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
