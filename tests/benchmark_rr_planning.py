# -*- coding: utf-8 -*-
"""OaK 的 Model→Planning 层在 KeyDoor 上的基准 —— 指标 = **look-ahead 操作数**。

论文(Sutton et al. 2022)用它自己的指标衡量这套机制:一个"让规划更省"的
抽象,只有在**规划代价**上才看得见收益(§9.2:"Use the source's own metric
for the mechanism")。本项目已在 gridworld 上复现了论文的双向结果:

    两房间 (36 状态)   primitive 1400 / shortest 1750 / **RR 875**
    四房间 (80 状态)   primitive 48872 / shortest 53710 / **RR 26200**

本文件把同一件事搬到 **KeyDoorMDP**(时序结构已证:A ∈ [−1,+1], 33% 显著)。

## 消融矩阵(相邻两臂只差**一个**变量)

  arm            c(cumulant)  z(stopping value)        作用
  ─────────────  ───────────  ────────────────────────  ──────────────────
  primitive      —            —                         下界基线(无 option)
  rr-exact       R            V_main + (w̄−w_i)x_i        **机制本体**
  rr-zeroV       R            0      + (w̄−0 )x_i        ← 削掉**价值函数**
  rr-nobonus     R            V_main + (0 )x_i   ≡ V_main ← 削掉停止奖励(已知退化)
  rr-nocum       0            V_main + (w̄−w_i)x_i        ← 削掉**沿途主任务奖励**
  rr-nofeat      R            V_main + (w̄−w_i)x_uniform  ← 削掉**特征语义**
  bottleneck     −1           0 @ 瓶颈                   **已知弱(论文点名)**
  random         R            V_main + (w̄−w_i)x_random   **随机目标(负对照)**

★ 判据**在跑之前写死**(写进本文件, 见 `INTERPRETATION_RULES`), 免得数字
  出来以后往想要的方向读。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from hibs_lnn.rr_options import (                            # noqa: E402
    KeyDoorTabular, build_bottleneck_library, build_random_goal_library,
    build_rr_library, feature_vectors, main_value,
)
from hibs_lnn.subtask_options import (                       # noqa: E402
    build_option_model, ops_to_tolerance,
)

TOL = 0.01
MAX_OPS = 400000

# ── 判据写死在此:跑完**先读这里再读数字** ────────────────────────────
INTERPRETATION_RULES = [
    "① 内部哨兵: primitive 臂必须复现历史值(两房间 1400 量级的同型行为)。"
    "若 primitive 臂与历史不符 -> 本 harness 自身有 bug, 其余数字全部作废。",
    "② 已知负向: bottleneck 臂**不得**优于 primitive。若它赢了 -> harness 有 bug"
    "(论文明确指出瓶颈子目标是已知弱基线)。",
    "③ 机制判据: rr-exact 优于 primitive 才叫'机制有效'。"
    "相等等不于'无效', 而是'这个 regime 上无可抽象的结构', 必须换 regime 再说。",
    "④ 价值函数假设(用户的猜测): 比较 rr-exact vs rr-zeroV。"
    "两者**逐位相同** = 该 regime 下价值函数不影响规划 -> 假设**未被这个试验台检验**"
    "(不是'被证伪');两者不同且 rr-exact 更省 = 价值函数是收益来源。",
    "⑤ 只在 tol=0.01 与 max_ops=400000 下读数:'打满上限'的臂记为不收敛, 不当作有限值比较。",
    "⑥ 单 regime 的结果只是**一个点**; 结论要在多 regime 上一致才成立。",
]


def arm_libraries(mdl, V_main):
    """构造矩阵里每个臂的 option 库。"""
    arms = {}
    arms["rr-exact"] = build_rr_library(mdl, V_main, bonus_weight=1.0)
    arms["rr-zeroV"] = build_rr_library(mdl, V_main, bonus_weight=1.0,
                                        z_mode="zeroV")
    arms["rr-nobonus"] = build_rr_library(mdl, V_main, bonus_weight=1.0,
                                          z_mode="nobonus")
    arms["rr-nocum"] = build_rr_library(mdl, V_main, bonus_weight=1.0,
                                        cumulant="zero")
    # 削掉特征语义: 特征用一个与任务无关的均匀特征(半数为 1)
    n = mdl.n_states
    x_unif = np.array([1.0 if (s % 2 == 0) else 0.0 for s in range(n)])
    from hibs_lnn.subtask_options import reward_respecting_subtask
    st_u = reward_respecting_subtask(mdl, x_unif, V_main, bonus_weight=1.0)
    st_u.name = "rr[nofeat]"
    st_u.feature = x_unif
    st_u.solve(mdl)
    arms["rr-nofeat"] = [st_u]
    arms["bottleneck"] = build_bottleneck_library(mdl)
    arms["random"] = build_random_goal_library(mdl, V_main, seed=0)
    return arms


def eval_arm(mdl, lib, v_star, s0=0):
    models = {}
    ks = []
    for st in lib:
        name = st.name
        if name in models:                      # 重名 -> 加后缀
            name = f"{name}#{len(models)}"
        M = build_option_model(mdl, st)
        models[name] = M
        # ★ 时长仪表: ops 指标可能被"option 有多长"主导而不是被"目标选得好不好"
        #   主导 —— 一个 k 步的 option 只花 1 次 backup 就传播 k 步的价值。
        #   记录每个 option 的期望终止步数, 才能把这个 confound 读出来。
        for s in range(mdl.n_states):
            for ent in M[s]:
                ks.append(ent[3])
    ops, errs, ops_curve, err_curve = ops_to_tolerance(
        mdl, models, v_star, s0, tol=TOL, max_ops=MAX_OPS)
    return {"ops": int(ops), "err": float(errs), "converged": bool(errs < TOL),
            "n_options": len(models), "n_sweeps": len(ops_curve),
            "ops_per_sweep": (float(ops) / max(1, len(ops_curve))),
            "mean_k": float(np.mean(ks)) if ks else 1.0,
            "max_k": float(np.max(ks)) if ks else 1.0,
            "beta_total": int(sum(int(np.sum(st.beta)) for st in lib))}


def main():
    regimes = [
        (2, 5, 7, "A: key=2 door=5 goal=7 (需全链)"),
        (1, 4, 7, "B: key=1 door=4 goal=7 (需全链)"),
        (3, 6, 7, "C: key=3 door=6 goal=7 (需全链)"),
        (2, 5, 3, "D: goal<door (免钥匙, 门无关)"),
    ]
    out = {"tol": TOL, "max_ops": MAX_OPS, "rules": INTERPRETATION_RULES,
           "regimes": []}

    print("=" * 78)
    print("KeyDoor 规划基准  |  指标 = look-ahead 操作数到 |V(s0)-V*| < %.2f" % TOL)
    print("=" * 78)

    for kp, dp, gp, label in regimes:
        from hibs_lnn.skill_mdp import KeyDoorMDP
        mdp = KeyDoorMDP(n_pos=8, key_pos=kp, door_pos=dp, goal_pos=gp, horizon=30)
        mdp.set_regime(kp, dp, gp)
        mdl = KeyDoorTabular(mdp, gamma=0.95)
        V_main = main_value(mdl)
        v_star = float(V_main[0])

        # ── ① 内部哨兵: primitive ────────────────────────────────────
        base = eval_arm(mdl, [], v_star)
        row = {"regime": (kp, dp, gp), "label": label,
               "v_star": v_star, "primitive": base, "arms": {}}

        print(f"\n{label}   V*(s0)={v_star:.6f}   |goal|={int(mdl.is_goal.sum())}")
        print(f"  {'arm':12s} {'ops':>7s} {'×prim':>6s} {'conv':>5s} "
              f"{'mean_k':>7s} {'max_k':>6s} {'n_opt':>5s} {'|β|':>5s} {'ops/扫':>8s}")
        print(f"  {'primitive':12s} {base['ops']:7d} {1.0:6.3f} "
              f"{str(base['converged']):>5s} {1.0:7.2f} {1.0:6.2f} {'-':>5s} "
              f"{'-':>5s} {base['ops_per_sweep']:8.1f}")

        for name, lib in arm_libraries(mdl, V_main).items():
            r = eval_arm(mdl, lib, v_star)
            r["ratio_vs_primitive"] = r["ops"] / max(1, base["ops"])
            row["arms"][name] = r
            print(f"  {name:12s} {r['ops']:7d} {r['ratio_vs_primitive']:6.3f} "
                  f"{str(r['converged']):>5s} {r['mean_k']:7.2f} {r['max_k']:6.2f} "
                  f"{r['n_options']:5d} {r['beta_total']:5d} {r['ops_per_sweep']:8.1f}")
        out["regimes"].append(row)

    # ── w̄ 扫描(dose-response: 中间值应当最好, 见配方 §9.3 末) ────────
    print("\n" + "=" * 78)
    print("w̄ 扫描(regime A)—— 论文: 中间值最好; w̄→大 退化成 shortest-path")
    print("=" * 78)
    kp, dp, gp = 2, 5, 7
    from hibs_lnn.skill_mdp import KeyDoorMDP
    mdp = KeyDoorMDP(n_pos=8, key_pos=kp, door_pos=dp, goal_pos=gp, horizon=30)
    mdp.set_regime(kp, dp, gp)
    mdl = KeyDoorTabular(mdp)
    V_main = main_value(mdl)
    base = eval_arm(mdl, [], float(V_main[0]))
    wbar = {}
    print(f"  primitive: ops={base['ops']}")
    for w in (0.0, 0.1, 0.3, 0.5, 0.78, 1.0, 3.0, 10.0, 100.0):
        lib = build_rr_library(mdl, V_main, bonus_weight=w)
        r = eval_arm(mdl, lib, float(V_main[0]))
        wbar[str(w)] = r
        print(f"  w̄={w:<6} ops={r['ops']:8d}  err={r['err']:.2e} "
              f"conv={r['converged']}  ×prim={r['ops']/max(1,base['ops']):.3f}  "
              f"|β|={r['beta_total']}")
    out["wbar_sweep_regimeA"] = {"primitive_ops": base["ops"], "arms": wbar}

    os.makedirs("results", exist_ok=True)
    with open("results/rr_planning_keydoor.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n→ results/rr_planning_keydoor.json")


if __name__ == "__main__":
    main()
