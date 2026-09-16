#!/usr/bin/env python3
"""benchmark_oak_repro.py — 在论文的 gridworld 上复现 Fig.1 的正反例。

## 这是本项目的 ground truth

论文 (arXiv 2202.03466) Fig.1 用两房间 gridworld 给出一个**明确的正反例**:

    只用 primitive actions                —— 基线
    + shortest-path (bottleneck) option   —— 论文实测**比基线还慢**
    + reward-respecting option            —— 论文实测**明显更快**

## 为什么这个基准对本项目是决定性的

本项目在 lm4 上测出「Options 有害」,但 lm4 的实测是 `A ≡ 0`(动作可交换、无时序结构)
—— 在那种环境里**任何类别的 option 都不可能有收益**,所以那个负结果测的是**基准的性质**。

在**论文这个有正反例的环境**里跑,就能回答一个完全不同、且可判定的问题:

    我的 option + option model + planning 管线, 到底写对没有?

判据(事前写死):
    ✓ 若复现出「shortest-path 不比 primitives 快」(或更慢)
    ✓ 且复现出「reward-respecting 明显更快」
    -> 管线是对的, 问题只在基准选择与 subtask 定义

    ✗ 若两者都无量级差别 -> 我的 planner/option model 还有实现问题

用法:
    python3 tests/benchmark_oak_repro.py                  # 两房间(确定性, Fig.1)
    python3 tests/benchmark_oak_repro.py --four-room      # 四房间(随机, Fig.6)
    python3 tests/benchmark_oak_repro.py --bonus 0.1 1 10 100   # 论文 §6 的 bonus 权重扫描
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hibs_lnn.gridworld import (FOUR_ROOM, TWO_ROOM, GridWorld, check_two_room,
                                find_bottlenecks, value_iteration)
from hibs_lnn.subtask_options import (build_option_model, planning_curve,
                                      ops_to_tolerance, reward_respecting_subtask,
                                      shortest_path_subtask)


def make_env(four_room: bool):
    if four_room:
        return GridWorld(FOUR_ROOM, gamma=0.99, stochastic=True), FOUR_ROOM
    return GridWorld(TWO_ROOM, gamma=0.99, stochastic=False), TWO_ROOM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--four-room", action="store_true")
    ap.add_argument("--bonus", type=float, nargs="+", default=[1.0],
                    help="reward-respecting 的 bonus 权重 w̄ (论文 §6: 0.1/1/10/100)")
    ap.add_argument("--tol", type=float, default=0.01, help="判定'已规划好'的容差")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env, layout = make_env(args.four_room)
    name = "four-room (stochastic)" if args.four_room else "two-room (deterministic)"
    print("=" * 74)
    print("论文 gridworld: %s" % name)
    print("=" * 74)
    print("  状态数 %d, 动作数 4, γ = %.2f, 随机 = %s" % (env.n_states, env.gamma, env.stochastic))

    # ── bottleneck(论文用的子目标候选) ────────────────────────────────
    bt = find_bottlenecks(env)
    cand = [(bt[s], s) for s in range(env.n_states)
            if s != env.start and not env.is_goal[s]]
    cand.sort(reverse=True)
    hall = cand[0][1]
    r, c = env.cells[hall]
    print("  bottleneck(介数最高) = 状态 %d  @(%d,%d), 介数 %.0f" % (hall, r, c, bt[hall]))

    # ── 真值 v*(s0):先把 primitive-only 收敛 ─────────────────────────
    V_star_all, _ = value_iteration(env, n_sweeps=500)
    v_star = float(V_star_all[env.start])
    print("\n  真值 v*(s0) = %.6f  (primitive-only VI 收敛)" % v_star)

    results = {}

    # ═══ 条件 A:只用 primitive ═══════════════════════════════════════
    print("\n" + "-" * 74)
    print("条件 A: 只用 primitive actions")
    print("-" * 74)
    ops_a, errs_a = planning_curve(env, {}, v_star, env.start, tol=args.tol, max_ops=300000)
    print("  收敛所需 look-ahead 操作数 = %d   最终误差 %.2e" % (ops_a[-1], errs_a[-1]))
    results["primitive"] = dict(ops=ops_a[-1], err=errs_a[-1],
                                n_sweeps=len(ops_a))

    # ═══ 条件 B: + shortest-path (bottleneck) option ═════════════════
    print("\n" + "-" * 74)
    print("条件 B: + shortest-path (bottleneck) option   [论文实测: 比 A 还慢]")
    print("-" * 74)
    sp = shortest_path_subtask(env, hall).solve(env)
    M_sp = build_option_model(env, sp)
    print("  subtask 解出的 option: π_o 非 STOP 的状态 %d 个, β_o=True 的状态 %d 个"
          % (int((sp.policy != 4).sum()), int(sp.beta.sum())))
    ops_b, errs_b = planning_curve(env, {"sp": M_sp}, v_star, env.start,
                                   tol=args.tol, max_ops=300000)
    print("  收敛所需 look-ahead 操作数 = %d   最终误差 %.2e" % (ops_b[-1], errs_b[-1]))
    ratio_b = ops_b[-1] / ops_a[-1]
    print("  相对 A 的比值 = %.3f  %s"
          % (ratio_b, "✓ 确实不更快(复现论文)" if ratio_b > 0.98 else "✗ 变快了(与论文不符)"))
    results["shortest_path"] = dict(ops=ops_b[-1], err=errs_b[-1], ratio=ratio_b)

    # ═══ 条件 C: + reward-respecting option ══════════════════════════
    print("\n" + "-" * 74)
    print("条件 C: + reward-respecting option   [论文实测: 明显更快]")
    print("-" * 74)
    feat = np.zeros(env.n_states)
    feat[hall] = 1.0                      # 特征 x_i = "在 bottleneck 处" 的一热指示
    for bw in args.bonus:
        rr = reward_respecting_subtask(env, feat, V_star_all, bonus_weight=bw).solve(env)
        M_rr = build_option_model(env, rr)
        ops_c, errs_c = planning_curve(env, {"rr": M_rr}, v_star, env.start,
                                       tol=args.tol, max_ops=300000)
        ratio_c = ops_c[-1] / ops_a[-1]
        n_stop = int(rr.beta.sum())
        print("  w̄ = %-6g  look-ahead = %7d   相对 A = %.3f  %s   (β_o=True 状态 %d 个)"
              % (bw, ops_c[-1], ratio_c,
                 "✓ 更快(复现论文)" if ratio_c < 0.98 else "✗ 没快", n_stop))
        results["reward_respecting_w%.4g" % bw] = dict(
            ops=ops_c[-1], err=errs_c[-1], ratio=ratio_c, bonus=bw, n_stop=n_stop)

    # ═══ 判读(事前写死的判据) ════════════════════════════════════════
    print("\n" + "=" * 74)
    print("判读(事前写死)")
    print("=" * 74)
    best_rr = min((v["ratio"] for k, v in results.items()
                   if k.startswith("reward_respecting")), default=None)
    ok_b = ratio_b > 0.98
    ok_c = best_rr is not None and best_rr < 0.98
    print("  ① shortest-path 不比 primitive 更快 : %s  (比值 %.3f)" % ("✓" if ok_b else "✗", ratio_b))
    print("  ② reward-respecting 明显更快        : %s  (最小比值 %s)"
          % ("✓" if ok_c else "✗", "%.3f" % best_rr if best_rr else "n/a"))
    if ok_b and ok_c:
        print("\n  -> 管线正确: option + option model + planning 的写法是对的。")
        print("     本项目在 lm4 上的负结果因此**确认**是基准选择 + subtask 定义的问题,")
        print("     而不是「option 这个机制没有价值」。")
    else:
        print("\n  -> 管线仍有实现问题, 先别对任何 benchmark 下 option 的结论。")

    payload = dict(env=name, n_states=env.n_states, gamma=env.gamma,
                   bottleneck=hall, v_star=v_star, tol=args.tol, results=results)
    out = args.out or ("results/oak_repro_%s.json" % ("four" if args.four_room else "two"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n  JSON -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
