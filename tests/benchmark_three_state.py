#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_three_state.py — **三态 epistemic state(好/坏/未知)的验收台**。

## 为什么需要这个文件

leo 的判断: 当前最大的缺口不是精度, 是 **epistemic state** ——
系统必须能区分"好、坏、**未知**", 而在此之前 `advq == 0` 被一律读成"不坏"。

那个 `0` 在两种完全不同的情况下出现:

    (a) 证据充分, 真的没有显著差异           -> 好
    (b) **奖励流为空, Φ 学不到东西**
        (harm 故障下整臂全灭时 `Q_adv ≡ 0.0000` 精确零)  -> **未知**

把 (b) 读成"不坏"就是**自欺**。这个文件的核心判据就是**直接测这个自欺**:
在 agent 正在整体失败的场景里, 旧报告会不会说"一切正常"?

## 三个固定场景(leo 阶段一)

    normal   无故障        —— 正常 option 应该被保留(判定为"好")
    inert    惰性故障      —— 看着效率差, 但**不应误杀**
    harm     有害故障      —— 真正有害, 应该被检出(判"坏", 或至少不是"好")

## 八项固定指标(leo 阶段一)

    ① harm 检测延迟   ② inert 误杀率   ③ 退休后恢复时间   ④ fallback 比例
    ⑤ 未知状态占比    ⑥ 主任务成功率/步数
    ⑦ 证据是否持续增长   ⑧ **自欺率**(失败时仍报"好"的比例)

## 两个臂: 三态本身是一条消融轴

    b2-nostate  冻结基线(GVF + advq + 遗忘 + 原谅)但 `rr_three_state=False`
    b2-3state   同上 + `rr_three_state=True`(未知 -> 一等状态)

这样"把未知提升为一等状态到底改变了什么"是可归因的。
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

FAULT_OPT = "rr[door]"

# 固定场景: (regime 序列, 故障, 从第几段起施加故障)
SCENARIOS = {
    "normal": dict(chain=[DEFAULT_REGIMES[i] for i in (0, 1, 2)],
                   fault=None, fault_at=99),
    "inert":  dict(chain=[DEFAULT_REGIMES[i % 3] for i in range(6)],
                   fault="inert", fault_at=3),
    "harm":   dict(chain=[DEFAULT_REGIMES[i % 3] for i in range(6)],
                   fault="harm", fault_at=3),
    # ★★ "不稳定" 故障 —— 与 `harm`("稳定但错误")形成对照。
    #    leo 点名的否证实验就在这里: 如果 A 类只对 `chaos` 有反应、
    #    对 `harm` 无反应, 那它学到的是**可预测性**, 不是**任务价值**,
    #    因此只能当辅助/风险通道, 不能单独驱动退休或选择。
    "chaos":  dict(chain=[DEFAULT_REGIMES[i % 3] for i in range(6)],
                   fault="chaos", fault_at=3),
}


class _RandPolicy:
    """`chaos` 故障: **每步随机出动作** —— 动力学不可预测。

    与 `harm`(把 π_o 反向后仍然完全确定)正好构成一对:
      · `harm`  = 稳定 + 错误   -> 可预测性高, 任务价值负
      · `chaos` = 不稳定 + 无向 -> 可预测性低, 任务价值也是负
    两者对比可以把"熟悉度/可预测性"和"任务价值"**分离**开 —— 这是 A 类
    唯一能证明自己是值信号还是辅助信号的办法。
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def __getitem__(self, s):
        return int(self.rng.randint(2))

    def __len__(self):
        return 64

# ── 冻结基线(leo 契约): GVF + advq + 遗忘 + 原谅 ──────────────────────
FROZEN = dict(rr_select_rule="calibrated", rr_pot="gvf", rr_retire_rule="advq",
              rr_score="advq", rr_forget_hl=20, rr_forgive_p=0.10)

ARMS = {
    "uniform":    dict(rr_select_rule="uniform"),
    "b2-nostate": dict(FROZEN, rr_three_state=False),
    "b2-3state":  dict(FROZEN, rr_three_state=True),
    # ★ 阶段四 A: 三候选稠密动力学证据。其余**全部**沿用冻结契约,
    #   所以臂间差异只能归因到 cumulant 本身(leo 的归因纪律)。
    "a1": dict(FROZEN, rr_three_state=True, rr_dyn="A1"),   # 转移不确定性
    "a2": dict(FROZEN, rr_three_state=True, rr_dyn="A2"),   # 转移预测误差
    "a3": dict(FROZEN, rr_three_state=True, rr_dyn="A3"),   # 已知稳定区域
}


def apply_fault(ag, kind):
    """把故障加到目标 option 上。故障**只在行为上**, 不改台账键。"""
    st = next((s for s in ag.rr_lib_obj if getattr(s, "name", None) == FAULT_OPT), None)
    if st is None:
        return False
    if kind == "inert":                   # 看着坏: 达不成子目标...
        ag._rr_cap[FAULT_OPT] = 1         # ...但那一步仍是 π_o 的最优步
    elif kind == "harm":                  # 真的坏: 主动朝反方向走
        #   ★ 这个故障的性质是 **"稳定 + 错误"**: 转移完全确定、极易预测,
        #     但任务价值是负的。它同时是 A 类的**否证实验**(见判据⑧)。
        p = np.asarray(st.policy).copy()
        p2 = p.copy()
        p2[p == LEFT] = RIGHT
        p2[p == RIGHT] = LEFT
        st.policy = p2
    elif kind == "chaos":                 # 不稳定: 每步随机动作
        #   转移**不可预测**, 与 `harm` 的可预测性形成对照。
        st.policy = _RandPolicy(seed=12345)
        ag._rr_cap[FAULT_OPT] = 12        # 不让它无限跑
    else:
        raise ValueError(f"unknown fault: {kind}")
    return True


def run_arm(kw, scen, episodes_per=300, seed=0, thresh=0.8, window=20):
    sc = SCENARIOS[scen]
    mdp = KeyDoorMDP(n_pos=8, horizon=30, **sc["chain"][0])
    ag = RRSkillAgent(mdp, mode="rr", seed=seed, rr_lib="rr", opt_prob=1.0, **kw)
    segs = []
    for ci, reg in enumerate(sc["chain"]):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        ag._rr_ensure()
        faulted = (sc["fault"] is not None and ci >= sc["fault_at"])
        if faulted:
            apply_fault(ag, sc["fault"])
        n0 = ag.rr_outcome.get(FAULT_OPT, {}).get("n", 0)
        broken_at = None
        succ, steps_ok = [], []
        for _ in range(episodes_per):
            ok, stp = ag.run_episode()
            succ.append(1.0 if ok else 0.0)
            if ok:
                steps_ok.append(stp)
            if faulted and broken_at is None and FAULT_OPT in ag.rr_broken():
                broken_at = ag.rr_outcome[FAULT_OPT]["n"] - n0
        s = np.array(succ)
        t_adapt = episodes_per
        for e in range(window - 1, episodes_per):
            if float(s[e - window + 1:e + 1].mean()) >= thresh:
                t_adapt = e + 1
                break
        cs = ag.rr_calib_stats()
        o = cs["per_option"].get(FAULT_OPT, {})
        others = {k: v for k, v in cs["per_option"].items() if k != FAULT_OPT}
        segs.append({
            "seg": ci, "tc": reg.get("goal_pos"), "faulted": bool(faulted),
            "t_adapt": int(t_adapt),
            "succ": float(s[-window:].mean()),
            "steps_go": float(np.mean(steps_ok)) if steps_ok else float("nan"),
            # ① 检测: 目标是坏的
            "detect": broken_at,
            "bad_state": o.get("state"),
            # ② 误杀: 非目标的 option 被判坏
            "false_kill": sum(1 for v in others.values() if v["state"] == "bad"),
            # ③ 恢复: 目标从坏恢复过几次
            "recovered": o.get("recovered_n", 0),
            "retired": o.get("retired_n", 0),
            # ④ fallback = 退回基元层的**选择尝试占比**。分母必须是 `sel_attempts`
            #   (尝试总数), 不能用 `selections`(只在真的选中时 +1) —— 否则
            #   比值会 > 1(实测 6.827), 指标失去意义。
            "fallback": ag.rr_stats.get("starved_sel", 0)
            / max(1, ag.rr_stats.get("sel_attempts", 1)),
            # ⑤ 三态占比
            "states": cs["states"],
            "starved": cs["starved"],
            "since_rew": cs["since_rew"],
            # ⑦ 证据增长
            "evidence": float(sum(v["advq_n"] for v in cs["per_option"].values())),
            # ★ A 类通道: 转移数 / 基线压力 / 每 option 的证据量与风险标记
            "dyn": (ag.rr_dyn_stats() if hasattr(ag, "rr_dyn_stats") else None),
            "no_base": o.get("no_base", 0),
            "advq": o.get("advq", 0.0),
            "advq_lo": o.get("advq_lo", 0.0),
            "advq_n": o.get("advq_n", 0.0),
            "usage": o.get("n", 0) / max(1, sum(v["n"] for v in cs["per_option"].values())),
        })
    return segs, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--scenarios", default="normal,inert,harm")
    ap.add_argument("--arms", default="uniform,b2-nostate,b2-3state")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scen_names = a.scenarios.split(",")
    arms = {k: v for k, v in ARMS.items() if k in a.arms.split(",")}

    out, agents = {}, {}
    for sc in scen_names:
        for name, kw in arms.items():
            runs = [run_arm(kw, sc, a.episodes_per, sd) for sd in range(a.seeds)]
            out[(sc, name)] = [r[0] for r in runs]
            agents[(sc, name)] = runs[-1][1]

    # ── 打印 ────────────────────────────────────────────────────────
    for sc in scen_names:
        print("\n" + "=" * 108)
        print(f"场景 {sc}   故障={SCENARIOS[sc]['fault']}   从第 {SCENARIOS[sc]['fault_at']} 段起")
        print("=" * 108)
        for name in arms:
            runs = out[(sc, name)]
            print(f"\n── {name} " + "─" * max(0, 96 - len(name)))
            print(f"   {'段':>3}{'故障':>6}{'T_adapt':>9}{'步数':>8}{'成功率':>8}"
                  f"{'使用率':>8}{'advq':>9}{'下界':>9}{'advq_n':>8}"
                  f"{'判定':>8}{'好/坏/未知':>14}{'检测':>7}{'误杀':>6}")
            for si in range(len(SCENARIOS[sc]["chain"])):
                g = [r[si] for r in runs]
                m = lambda f: float(np.nanmean([x[f] for x in g]))
                dv = [x["detect"] for x in g if x["detect"] is not None]
                dl = (f"{np.mean(dv):.0f}" if dv
                      else ("未检测" if g[0]["faulted"] else "—"))
                st = m_states = g[0]["states"]
                st_s = f"{m_states['good']}/{m_states['bad']}/{m_states['unknown']}"
                print(f"   {si:>3}{'是' if g[0]['faulted'] else '否':>6}"
                      f"{m('t_adapt'):>9.1f}{m('steps_go'):>8.2f}{m('succ'):>8.3f}"
                      f"{m('usage'):>8.3f}{m('advq'):>9.4f}{m('advq_lo'):>9.4f}"
                      f"{m('advq_n'):>8.1f}{str(g[0]['bad_state']):>8}{st_s:>14}"
                      f"{dl:>7}{m('false_kill'):>6.0f}")

    # ── 事前判据 ────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("事前判据")
    print("=" * 108)
    ok = []

    def segs_(sc, nm):
        return out[(sc, nm)][0]

    def last_faulted(sc):
        ch = SCENARIOS[sc]["chain"]
        return [i for i in range(len(ch)) if i >= SCENARIOS[sc]["fault_at"]]

    # ★★ ① 自欺门: 任何"奖励流为空"的段, 都**不得报"好"**
    #    这是 leo 点的核心缺口的直接检验: `advq == 0` 在"证据充分无差异"和
    #    "奖励流为空根本没数据"两种情况下都出现; 前者可以报"好", 后者不行。
    #    注意这个判据对**所有臂**成立 —— 它是报告层的不变量, 与决策规则无关。
    lies = []
    for sc in scen_names:
        for nm in arms:
            for si, x in enumerate(segs_(sc, nm)):
                if x["starved"] and x["states"]["good"] > 0:
                    lies.append((sc, nm, si, x["states"], x["since_rew"]))
    ok.append(("① 自欺门: 奖励流为空时从不报'好'", len(lies) == 0,
               f"违约 {len(lies)} 处" + (f" 例: {lies[0]}" if lies else "")))

    # ★★ ② 决策有效性: 三态开着 -> agent 会因"坏/未知"停止使用有害 option 而**恢复**
    #    这是在测"把未知提升为一等状态"到底改变了什么: 不是报告更好看,
    #    是行为上能自救。
    if "harm" in scen_names and {"b2-nostate", "b2-3state"} <= set(arms):
        nf = [segs_("harm", "b2-nostate")[i] for i in last_faulted("harm")]
        tf = [segs_("harm", "b2-3state")[i] for i in last_faulted("harm")]
        su_n = float(np.mean([x["succ"] for x in nf]))
        su_t = float(np.mean([x["succ"] for x in tf]))
        ok.append(("② 三态使 agent 在失败场景下恢复 (成功率 3state 显著 > nostate)",
                   su_t > su_n + 0.2, f"{su_n:.3f} -> {su_t:.3f}"))

    # ③ 未知不得触发退休 / 不得增加误杀
    #    ★ 守卫必须有: 臂子集是常规用法(--arms a1,a2,a3), 缺守卫会直接
    #      KeyError 崩在**评测阶段** —— 数据全跑完了才崩, 最浪费的一种。
    if "harm" in scen_names and {"b2-nostate", "b2-3state"} <= set(arms):
        print(f"     harm 末段 误杀(非目标 option 被判坏): "
              f"nostate {segs_('harm','b2-nostate')[-1]['false_kill']} / "
              f"3state {segs_('harm','b2-3state')[-1]['false_kill']}")
        ok.append(("③ 三态不增加误杀",
                   segs_("harm", "b2-3state")[-1]["false_kill"]
                   <= segs_("harm", "b2-nostate")[-1]["false_kill"]))

    # ④ inert 不应被系统性误杀
    if "inert" in scen_names:
        for nm in ("b2-nostate", "b2-3state"):
            if nm in arms:
                fk = sum(segs_("inert", nm)[i]["false_kill"] for i in last_faulted("inert"))
                print(f"     inert 故障段累计误杀 {nm}: {fk}")
        ok.append(("④ inert 场景下三态臂不误杀",
                   all(segs_("inert", "b2-3state")[i]["false_kill"] == 0
                       for i in last_faulted("inert"))))

    # ⑤ 正常场景应判"好"(三态不能退化成一律未知)
    if "normal" in scen_names:
        c = segs_("normal", "b2-3state")[-1]
        print(f"     normal 末段: 三态 {c['states']}  (starved={c['starved']})")
        ok.append(("⑤ normal 场景下目标 option 最终判'好'", c["states"]["good"] >= 1))

    # ⑥ 证据持续增长
    for sc in scen_names:
        if "b2-3state" not in arms:
            continue
        ev = [x["evidence"] for x in segs_(sc, "b2-3state")]
        ok.append((f"⑥ [{sc}] 证据持续增长 (advq_n 单调)",
                   all(ev[i + 1] >= ev[i] - 1e-9 for i in range(len(ev) - 1))))

    # ★★ ⑨ **反熟悉度否证实验**(leo 点名的最重要一条; 编号接在既有 ⑧ 之后)
    #    对两个镜像故障各问一个问题:
    #      chaos(不稳定): A 类**应该**察觉到 -> 证明它测的是"可预测性"
    #      harm (稳定+错误): A 类**应该**察觉不到 -> 证明它测的**不是**
    #                        "任务价值", 因此只能当辅助/风险通道
    #    如果 A 在 harm 上也"察觉到了", 那是更可疑的情况(需另查来源);
    #    如果 A 在 chaos 上毫无反应, 那它连"可预测性"都没测到 -> 直接淘汰。
    if "chaos" in scen_names:
        def dyn_flag(sc, nm):
            """故障段里 A 类触发风险的 (命中, 总数)。"""
            hit = tot = 0
            for x in segs_(sc, nm):
                dy = x.get("dyn")
                if not dy or not x["faulted"]:
                    continue
                for v in dy["per_option"].values():
                    tot += 1
                    hit += int(v["risk"])
            return hit, tot
        for nm in ("a1", "a2", "a3"):
            if nm not in arms:
                continue
            hc, tc = dyn_flag("chaos", nm)
            hh, th = dyn_flag("harm", nm)
            ok.append((f"⑨ [{nm}] A 察觉'不稳定'(chaos) 而不察觉'稳定但错误'(harm)",
                       (hc > 0) and (hh == 0) and tc > 0 and th > 0,
                       f"chaos {hc}/{tc}   harm {hh}/{th}"))

    # ⑦ 主任务不劣于安全基线 —— **只在 uniform 真正能跑的场景比**
    #    (harm 下 uniform 整体失败 steps_go=nan, 那时比较没有意义;
    #     失败场景由判据 ② 用成功率来测)
    for sc in scen_names:
        if "uniform" not in arms or "b2-3state" not in arms:
            continue
        fu = np.nanmean([x["steps_go"] for x in segs_(sc, "uniform")])
        fs = np.nanmean([x["steps_go"] for x in segs_(sc, "b2-3state")])
        # ★ 安全基线的可用性必须**只在故障段上**判: harm 下 uniform 在故障段
        #   成功率 0.000 / 步数 nan(整体失败), 那时"比步数"没有意义 ——
        #   用全段均值会把健康段的成功混进来, 误以为基线可用。
        uf = [x for x in segs_(sc, "uniform") if x["faulted"]]
        us_f = np.mean([x["succ"] for x in uf]) if uf else 1.0
        if np.isnan(fu) or np.isnan(fs) or us_f < 0.5:
            ok.append((f"⑦ [{sc}] 主任务不劣于 uniform", True,
                       f"uniform 在故障段基线不可用 (成功率 {us_f:.3f}), 由判据②覆盖"))
        else:
            ok.append((f"⑦ [{sc}] 主任务步数不劣于 uniform (增幅 < 15%)",
                       fs <= fu * 1.15, f"{fu:.2f} -> {fs:.2f}"))

    n_pass = 0
    for item in ok:
        nm, v = item[0], item[1]
        extra = item[2] if len(item) > 2 else None
        n_pass += int(bool(v))
        print(f"  {'PASS' if v else 'FAIL'}  {nm}" + (f"   [{extra}]" if extra else ""))
    print(f"\n  {n_pass}/{len(ok)} 通过")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        ser = {f"{sc}|{nm}": runs for (sc, nm), runs in out.items()}
        Path(a.out).write_text(json.dumps(ser, indent=2, default=str, ensure_ascii=False))
        print(f"原始数据: {a.out}")
    return 0 if n_pass == len(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
