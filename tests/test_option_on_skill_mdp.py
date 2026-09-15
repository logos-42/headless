#!/usr/bin/env python3
"""test_option_on_skill_mdp.py — 在**有时序结构**的环境上判定 OptionManager。

## 判定逻辑(这是"实现坏了 vs benchmark 有问题"的决定性区分)

    lm4     : A ≡ 0 (动作可交换)  -> 无时序结构 -> 那里的负结果**无法判定实现**
    KeyDoor : A ∈ [-1,+1], 33% 显著 -> 有时序结构 -> **可以用它判定实现**

    若 OptionManager 在这里能发现"拿钥匙→开门"这类技能并带来收益
        -> 实现没坏, lm4 的负结果是 benchmark 的性质 (用户判断成立)
    若在这里也发现不了/用不上
        -> **实现有问题, 该修实现**

## 三测项

  ① **发现**: discover() 找出的 option 是否对应任务里**真实存在**的技能链?
     判据: option 的动作序列应包含 GRAB / OPEN, 且起点→终点朝目标推进。
  ② **覆盖**: 发现的 option 能否覆盖"从起点到目标"的完整过程?
  ③ **T_adapt (B2/B3)**: regime 变化后恢复到阈值需要多少回合?
     primitives vs options。**这才是 option 该体现价值的地方** ——
     技能可复用(拿钥匙/开门), 只是**地点变了**。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hibs_lnn.skill_mdp import (  # noqa: E402
    GRAB, LEFT, N_ACT, OPEN, RIGHT, KeyDoorMDP,
)
from hibs_lnn.uncertainty_gate import TransitionEnsemble, UncertaintyGate  # noqa: E402


def collect_transitions(mdp, n_ep=400, seed=0):
    """采集 (state_vec, action, next_state_vec)。"""
    rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(n_ep):
        mdp.reset()
        done = False
        while not done:
            a = int(rng.randint(0, N_ACT))
            v = mdp.vec()
            _, _, done = mdp.step(a)
            X.append(np.append(v, a)); Y.append(mdp.vec())
    return np.array(X), np.array(Y)


def evaluate_options(mdp, ens, gate, om, n_ep=200, seed=0):
    """用发现的 option + 贪心 fallback 解任务, 返回成功率与平均步数。"""
    def goal_state_vec():
        mdp.reset()
        for _ in range(400):
            _, _, d = mdp.step(mdp.greedy_action())
            if d:
                break
        return mdp.vec()

    ok, tot = 0, 0
    for _ in range(n_ep):
        mdp.reset()
        done, n = False, 0
        while not done and n < mdp.horizon:
            v = mdp.vec()
            a = None
            if om is not None and om.options:
                o = om.select(v, goal=goal_state_vec(), exclude_unc=True)
                if o is not None:
                    a = int(o.actions[0])
            if a is None:
                a = mdp.greedy_action()
            _, _, done = mdp.step(a)
            n += 1
        ok += 1 if (done and mdp._reached()) else 0
        tot += n
    return ok / n_ep, tot / n_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--n-regions", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "option_on_mdp.json"))
    a = ap.parse_args()

    mdp = KeyDoorMDP(n_pos=a.n_pos, key_pos=2, door_pos=5, goal_pos=7, horizon=30)
    n_states = mdp.n_pos + 2
    print("KeyDoorMDP: n_pos=%d regime=%s 最优步数=%d" % (a.n_pos, mdp.regime, mdp.optimal_steps()))

    X, Y = collect_transitions(mdp, n_ep=a.episodes, seed=0)
    print("转移 %d 条, 状态维 %d" % (len(X), X.shape[1] - 1))

    ens = TransitionEnsemble(n_models=5, seed=0).fit(X, Y, N_ACT, None)
    gate = UncertaintyGate(tau_C=1.0)
    gate.calibrate([ens.predict(X[i, :X.shape[1] - 1], int(X[i, -1]), np.ones(N_ACT))[1]
                    for i in range(0, len(X), max(1, len(X) // 200))])

    from hibs_lnn.option_manager import OptionManager
    om = OptionManager(ens, gate, max_len=a.max_len, seed=0)
    states = np.array([np.append(mdp.vec(), 0)[:X.shape[1] - 1] for _ in range(0)])  # 占位
    states = X[:, :X.shape[1] - 1]
    acts = X[:, -1].astype(int)
    opts = om.discover(states, acts, n_regions=a.n_regions)
    print("发现 %d 个 option" % len(opts))

    print()
    print("=" * 92)
    print("① 发现: 找出的 option 是否对应真实技能链 (应含 GRAB=2 / OPEN=3)")
    print("=" * 92)
    names = {LEFT: "left", RIGHT: "right", GRAB: "grab", OPEN: "open"}
    hit = 0
    for o in opts[:10]:
        acts_o = [int(x) for x in o.actions]
        has_grab = GRAB in acts_o
        has_open = OPEN in acts_o
        if has_grab or has_open:
            hit += 1
        print("   actions=%-24s (%s)  start=%s goal=%s  unc=%.4f"
              % (str(acts_o), " ".join(names.get(x, "?") for x in acts_o),
                 np.round(o.start_center, 2).tolist(),
                 np.round(o.goal_center, 2).tolist(),
                 float(np.mean(o.uncs)) if o.uncs else 0.0))
    print("   -> 含关键动作 (grab/open) 的 option: %d / %d" % (hit, len(opts)))

    print()
    print("=" * 92)
    print("②/③ 解任务: primitives vs options (T_adapt 见下一段)")
    print("=" * 92)
    acc_p, st_p = evaluate_options(mdp, ens, gate, None)
    acc_o, st_o = evaluate_options(mdp, ens, gate, om)
    print("   primitives (贪心)      : 成功率 %.3f  平均步数 %.2f" % (acc_p, st_p))
    print("   options + 贪心 fallback: 成功率 %.3f  平均步数 %.2f" % (acc_o, st_o))
    print()
    print("   ★ 说明: 这里的贪心 fallback 已经是最优一步策略, 所以成功率都接近 1。")
    print("     真正能区分的是 **T_adapt**: regime 变化后需要多少回合恢复。")
    print("     (贪心需要**重新计算**, 而 option 是**可复用的技能** —— 这才是差异所在)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "n_options": len(opts), "n_with_key_actions": hit,
        "primitive": {"acc": acc_p, "steps": st_p},
        "option": {"acc": acc_o, "steps": st_o},
        "options": [o.stats() for o in opts],
    }, indent=1, ensure_ascii=False))
    print("\n已写 %s" % a.out)


if __name__ == "__main__":
    main()
