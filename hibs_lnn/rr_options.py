# -*- coding: utf-8 -*-
"""Reward-respecting subtask options on `KeyDoorMDP`(OaK 的第一步)。

论文:Sutton et al. 2022, *Reward-Respecting Subtasks for Model-Based
Reinforcement Learning*(arXiv:2202.03466)。

## 层序(不可颠倒, 见 `docs/oak_diagnosis_from_papers.md`)

    SubTask ──> Option ──> Option model ──> Planning
    (c, z)      (π_o,β_o)    SMDP model      planner

本项目此前一直从**转移图**出发(bottleneck / 介数中心性 / 最短路),那正是论文
点名的**最差类别**:"option models from ... shortest path options based on
bottleneck states ... are much less likely to be useful in planning"。
这个模块把驱动层换回 **SubTask**。

## 子任务的定义(论文式(4))

    c(s,a) = R(s,a)                          ← **与主任务完全相同的奖励**
    z_i(s) = V_main(s) + (w̄_i − w_i)·x_i(s)   ← 只在 x_i(s)=1 处允许停
             └────────── stopping bonus ──────────┘

`w_i` 是**特征 i 在主任务价值函数里的权重**(一热特征下 = 该特征为 1 诸状态上
`V_main` 的均值),不是常数 1。硬写 `w_i = 1` 会让 bonus 恒 0 → `z ≡ V_main`
→ `β_o` 处处为空 → option 永不停止 → 与 shortest-path 给出逐位相同的曲线。
""" 

import numpy as np

from hibs_lnn.skill_mdp import N_ACT
from hibs_lnn.subtask_options import (
    STOP,
    Subtask,
    reward_respecting_subtask,
    shortest_path_subtask,
)


# ══════════════════════════════════════════════════════════════════════
# 1. KeyDoor 的表格视图 —— 给 Subtask.solve 提供 (T, R, is_goal)
# ══════════════════════════════════════════════════════════════════════

class KeyDoorTabular:
    """`KeyDoorMDP` 的**表格**表示。

    KeyDoor 的动力学完全确定性, 所以 `T[s][a] = [(1.0, s')]`;唯一随 regime
    改变的是 `is_goal`(目标判定含 `goal_pos < door_pos` 的免钥匙情形)与 `R`
    (进目标得 +1)。**因此本对象必须按 regime 重建**, 不能跨 regime 复用。

    接口刻意与 `subtask_options.py` 里 gridworld 的环境对象**逐字一致**
    (`n_states` / `T` / `R` / `is_goal` / `gamma`), 这样 `Subtask.solve`
    与 `reward_respecting_subtask` 可以原样复用, 不必为 KeyDoor 另写一份。
    """

    def __init__(self, mdp, gamma=0.95):
        self.gamma = float(gamma)
        self.regime = tuple(mdp.regime)
        self.n_pos = int(mdp.n_pos)
        self.n_states = int(mdp.n_states)
        key_pos, door_pos, goal_pos = self.regime

        # 目标状态: 与 KeyDoorMDP._reached() 同一判据
        def is_term(s):
            pos, _hk, do = s // 4, (s // 2) % 2, s % 2
            return pos == goal_pos and (do or goal_pos < door_pos)

        self.is_goal = np.array([is_term(s) for s in range(self.n_states)],
                                dtype=bool)

        self.T = [[None] * N_ACT for _ in range(self.n_states)]
        self.R = np.zeros((self.n_states, N_ACT), dtype=float)
        for s in range(self.n_states):
            pos, hk, do = s // 4, (s // 2) % 2, s % 2
            for a in range(N_ACT):
                np_, nhk, ndo = pos, hk, do
                if a == 0:                      # LEFT
                    np_ = max(0, pos - 1)
                elif a == 1:                    # RIGHT
                    np_ = min(self.n_pos - 1, pos + 1)
                elif a == 2:                    # GRAB
                    if pos == key_pos and not hk:
                        nhk = 1
                elif a == 3:                    # OPEN
                    if pos == door_pos and hk and not do:
                        ndo = 1
                ns = np_ * 4 + nhk * 2 + ndo
                self.T[s][a] = [(1.0, ns)]
                self.R[s][a] = 1.0 if self.is_goal[ns] else 0.0
        self.regime_key = self.regime


# ══════════════════════════════════════════════════════════════════════
# 2. 特征向量与主任务价值
# ══════════════════════════════════════════════════════════════════════

def feature_vectors(mdl: KeyDoorTabular):
    """KeyDoor 的**子成就特征**: 「拿到钥匙」与「打开门」。

    这两个是任务链上真正的中间成果(key → door → goal), 对应论文里
    "the features are the sub-achievements of the task" 的取法。
    """
    n = mdl.n_states
    x_key = np.array([(s // 2) % 2 for s in range(n)], dtype=float)
    x_door = np.array([s % 2 for s in range(n)], dtype=float)
    return {"key": x_key, "door": x_door}


def main_value(mdl: KeyDoorTabular, max_sweeps=5000, tol=1e-12):
    """主任务的最优价值 `V_main`(精确价值迭代)。

    ★ 与 `Subtask.solve` 用**同一套约定**: 目标状态价值为 0, 其余
      `V(s) = max_a [ R(s,a) + γ V(s') ]`。两处约定一致是硬要求 ——
      否则 `z_i(s) = V_main(s) + bonus` 里的 `V_main` 与子任务自身
      的价值不在同一把尺子上, `β_o` 会在错误的地方触发。
    """
    n = mdl.n_states
    V = np.zeros(n)
    for _ in range(max_sweeps):
        delta = 0.0
        Vn = V.copy()
        for s in range(n):
            if mdl.is_goal[s]:
                Vn[s] = 0.0
                continue
            best = -np.inf
            for a in range(N_ACT):
                if mdl.T[s][a] is None:
                    continue
                val = mdl.R[s][a] + mdl.gamma * V[mdl.T[s][a][0][1]]
                if val > best:
                    best = val
            Vn[s] = best
            delta = max(delta, abs(Vn[s] - V[s]))
        V = Vn
        if delta < tol:
            break
    return V


def optimal_policy(mdl: KeyDoorTabular):
    """主任务最优策略(诊断用: 给 T_adapt 阈值与 sanity check)。"""
    V = main_value(mdl)
    n = mdl.n_states
    pol = np.full(n, STOP, dtype=int)
    for s in range(n):
        if mdl.is_goal[s]:
            continue
        best_a, best_v = STOP, -np.inf
        for a in range(N_ACT):
            val = mdl.R[s][a] + mdl.gamma * V[mdl.T[s][a][0][1]]
            if val > best_v:
                best_a, best_v = a, val
        pol[s] = best_a
    return V, pol


# ══════════════════════════════════════════════════════════════════════
# 3. 子任务库的构造
# ══════════════════════════════════════════════════════════════════════

def build_rr_library(mdl: KeyDoorTabular, V_main, bonus_weight=1.0,
                     feats=("key", "door"), cumulant="reward",
                     z_mode="rr"):
    """构造 reward-respecting 子任务库并**解出** option。

    参数就是消融的旋钮 —— 每个旋钮对应矩阵里的一个臂:

      `z_mode`    "rr"      → `z = V_main + (w̄ − w_i)x_i`   (机制本体)
                  "zeroV"   → `z = 0     + (w̄ − 0 )x_i`   (削掉价值函数)
                  "nobonus" → `w̄ = w_i` → bonus ≡ 0        (削掉停止奖励)
      `cumulant`  "reward"  → `c = R`(reward-respecting)
                  "zero"    → `c = 0`(纯"到达特征", 丢掉沿途目标)
    """
    feats_all = feature_vectors(mdl)
    lib = []
    for name in feats:
        x = feats_all[name]
        if z_mode == "rr":
            st = reward_respecting_subtask(mdl, x, V_main,
                                           bonus_weight=bonus_weight)
        elif z_mode == "zeroV":
            # 削掉价值函数: V_main ≡ 0。此时 z 只剩 bonus 项,
            # "停"与"继续"不再由主任务价值定价 —— 这正是用户猜测的
            # "可能是价值函数的原因" 的**直接检验臂**。
            z0 = np.zeros_like(np.asarray(V_main, dtype=float))
            st = reward_respecting_subtask(mdl, x, z0,
                                           bonus_weight=bonus_weight)
        elif z_mode == "nobonus":
            # w̄ = w_i → bonus ≡ 0 → z ≡ V_main → 论文警告的退化情形。
            # 这一臂**预期**表现为 option 永不提前停(β_o 空), 是 harness
            # 的"已知负向"校准点(Pitfall 35 / §9.3 第三条)。
            on = np.asarray(x, dtype=float) > 0
            wi = float(np.mean(np.asarray(V_main, dtype=float)[on])) if on.any() else 0.0
            st = reward_respecting_subtask(mdl, x, V_main, bonus_weight=wi)
        else:
            raise ValueError(f"unknown z_mode: {z_mode}")

        if cumulant == "zero":
            st.c_fn = (lambda s, a: 0.0)      # 丢掉沿途主任务奖励
        elif cumulant != "reward":
            raise ValueError(f"unknown cumulant: {cumulant}")

        st.name = f"rr[{name}]"
        st.feature = x
        st.solve(mdl)
        lib.append(st)
    return lib


def build_bottleneck_library(mdl: KeyDoorTabular, subgoal_state=None):
    """**已知弱基线(负对照)**: 从瓶颈/子目标状态出发的 shortest-path subtask。

    `c = −1`, `z = 0` 只在子目标处 —— 它优化的是"**到达**",完全无视任务目标。
    它在这里的**正确行为就是不比什么都不做好**, 所以它同时是 harness 的
    负向校准点(§9:一个已知负向的臂必须复现为负)。
    """
    if subgoal_state is None:
        # 瓶颈 = 门所在的格(拿钥匙后必须经过) —— 或用割点思想选
        door_pos = mdl.regime[1]
        subgoal_state = min(mdl.n_states - 1, door_pos * 4 + 2)   # (door, has_key, closed)
    st = shortest_path_subtask(mdl, subgoal_state)
    st.name = f"bottleneck[{subgoal_state}]"
    st.feature = np.zeros(mdl.n_states)
    st.solve(mdl)
    return [st]


def build_random_goal_library(mdl: KeyDoorTabular, V_main, seed=0,
                              n_goals=2, bonus_weight=1.0):
    """**随机目标(负对照)**: 目标状态随机取, 其余同 reward-respecting。

    论文 Fig.1 的第三条曲线。用来检验"收益是不是来自目标**选得好**,
    而不是来自'有一个可停止的子任务'这件事本身"。
    """
    rng = np.random.RandomState(seed)
    lib = []
    cand = [s for s in range(mdl.n_states) if not mdl.is_goal[s]]
    for i in range(n_goals):
        g = int(cand[rng.randint(len(cand))])
        x = np.zeros(mdl.n_states)
        x[g] = 1.0
        st = reward_respecting_subtask(mdl, x, V_main,
                                       bonus_weight=bonus_weight)
        st.name = f"random[{g}]"
        st.feature = x
        st.solve(mdl)
        lib.append(st)
    return lib


# ══════════════════════════════════════════════════════════════════════
# 4. 启动集 I_o 与选择
# ══════════════════════════════════════════════════════════════════════

def initiation_ok(st: Subtask, s: int) -> bool:
    """`I_o(s)`: 只有当**特征尚未达成**且**在 s 停止不是最优**时才允许启动。

    "特征已达成"= `x_i(s) > 0`(如 has_key=1)。已经拿到钥匙还去执行
    "拿钥匙"的 option 是空转;论文的 initiation set 正是排除这类状态。
    """
    x = getattr(st, "feature", None)
    if x is None:
        return True
    return float(x[s]) <= 0.0


def option_gain(st: Subtask, s: int, V_main) -> float:
    """选择用的**增益**: 该 option 承诺的价值超出主任务价值多少。

    只在可启动处有意义;不可启动返回 `-inf`。选择 = `argmax` 这个量,
    这就是"哪个子任务现在最值得追"。
    """
    if st.value is None or not initiation_ok(st, s):
        return -np.inf
    v = float(st.value[s])
    if not np.isfinite(v):
        return -np.inf
    return v - float(np.asarray(V_main)[s])
