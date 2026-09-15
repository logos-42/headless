"""skill_mdp.py — 一个有**已知 option 结构**的 MDP, 用来判定:

    (a) OptionManager 到底能不能发现并利用 option?
    (b) E9/E10 的负结果 是**实现坏了**, 还是 **lm4 benchmark 的性质**?

## 为什么需要它

lm4 的测量已经证明那个 benchmark **无时序结构**(不对称性 A ≡ 0, 动作可交换)。
在那种环境下 option 无论怎么实现都不会有用 —— 所以 lm4 上的负结果
**无法判定实现好坏**。

要判定实现, 必须换到一个**option 客观上应该有用**的环境。
这就是经典的「钥匙—门」链式任务: 最优解要求一段**有严格顺序**的多步行为。
若 OptionManager 在这里也发现不了/用不上 option, 那才是实现的问题。

## 任务结构

    state = (pos, has_key)            pos ∈ {0..N-1},  has_key ∈ {0,1}
    actions:
        0  left     位置 -1
        1  right    位置 +1
        2  grab     仅在 key_pos 且 !has_key 时拿到钥匙
        3  open     仅在 door_pos 且 has_key 时开门 (此后 goal 可达)

    reward:
        到达 goal 且门已开  -> +1, 回合结束
        步数超过上限        -> 0,  回合结束

    ★ **顺序依赖是内生的**:
        - "go_right 到 key" 只在**还没拿钥匙**时有用
        - "grab" 只在 key 位置有用
        - "open" 只在**已拿钥匙**时有用
      -> A(a,b) ≠ 0, 且不对称 (go_key → grab ≠ grab → go_key)
      -> 这正是 lm4 里**完全缺失**的结构

## regime (用于 A→B→A 复用测试)

    一个 regime = (key_pos, door_pos, goal_pos)。
    换 regime = 同样的技能 (拿钥匙/开门) 仍然有效, 但**地点变了**
    -> 这正是"可复用行为结构"的定义, 也是 option 该体现价值的地方。
"""
from __future__ import annotations

import numpy as np

LEFT, RIGHT, GRAB, OPEN = 0, 1, 2, 3
N_ACT = 4


class KeyDoorMDP:
    """钥匙—门 链式 MDP。`state = (pos, has_key)` 编码成整数 pos*2 + has_key。"""

    def __init__(self, n_pos=8, key_pos=2, door_pos=5, goal_pos=7, horizon=30):
        self.n_pos = int(n_pos)
        self.regime = (int(key_pos), int(door_pos), int(goal_pos))
        self.horizon = int(horizon)
        self.n_states = self.n_pos * 2
        self.t = 0
        self.pos, self.has_key, self.door_open = 0, 0, 0

    # ── 状态编码 ────────────────────────────────────────────────────
    def state(self):
        return self.pos * 2 + self.has_key

    def decode(self, s):
        return int(s) // 2, int(s) % 2

    def vec(self):
        """给知识层用的连续表示: one-hot(pos) ⊕ has_key ⊕ door_open。"""
        v = np.zeros(self.n_pos + 2)
        v[self.pos] = 1.0
        v[self.n_pos] = float(self.has_key)
        v[self.n_pos + 1] = float(self.door_open)
        return v

    def set_regime(self, key_pos, door_pos, goal_pos):
        self.regime = (int(key_pos), int(door_pos), int(goal_pos))

    def reset(self):
        self.t = 0
        self.pos, self.has_key, self.door_open = 0, 0, 0
        return self.state()

    # ── 动力学 ──────────────────────────────────────────────────────
    def step(self, a):
        key_pos, door_pos, goal_pos = self.regime
        self.t += 1
        if a == LEFT:
            self.pos = max(0, self.pos - 1)
        elif a == RIGHT:
            self.pos = min(self.n_pos - 1, self.pos + 1)
        elif a == GRAB:
            if self.pos == key_pos and not self.has_key:
                self.has_key = 1
        elif a == OPEN:
            if self.pos == door_pos and self.has_key and not self.door_open:
                self.door_open = 1
        # 到达目标
        done = False
        r = 0.0
        if self.pos == goal_pos and (self.door_open or goal_pos < door_pos):
            r, done = 1.0, True
        elif self.t >= self.horizon:
            done = True
        return self.state(), r, done

    def _reached(self):
        key_pos, door_pos, goal_pos = self.regime
        return self.pos == goal_pos and (self.door_open or goal_pos < door_pos)

    def greedy_action(self):
        """这个任务的最优一步策略 (只为给 baseline/交互测量用)。"""
        key_pos, door_pos, goal_pos = self.regime
        if self.pos != key_pos and not self.has_key and key_pos >= 0:
            return RIGHT if key_pos > self.pos else LEFT
        if self.pos == key_pos and not self.has_key:
            return GRAB
        if not self.door_open and goal_pos >= door_pos:
            if self.pos != door_pos:
                return RIGHT if door_pos > self.pos else LEFT
            if self.has_key:
                return OPEN
        return RIGHT if goal_pos > self.pos else LEFT

    def optimal_steps(self):
        """最优解需要多少步 (用于设 T_adapt 阈值)。"""
        key_pos, door_pos, goal_pos = self.regime
        n = abs(key_pos - 0) + 1          # 走到 key + grab
        if goal_pos >= door_pos:
            n += abs(door_pos - key_pos) + 1   # 走到 door + open
            n += abs(goal_pos - door_pos)
        else:
            n += abs(goal_pos - key_pos)
        return n


def interaction_asymmetry(mdp, a, b, n_ep=200, seed=0):
    """A(a,b) = I(a,b) − I(b,a), 用**走到目标的步数**度量。

        S_seq = 从初始态做完整序列再贪心走到目标的**总步数**
        I(a,b) = S_b − S_ab      (先做 a 再做 b, 比只做 b 少用多少步)

    ★ 为什么不能用「成功率」: 贪心策略从**任何**状态都能解出任务,
      于是成功率恒为 1.000, 前缀完全不影响 —— 测量饱和, 什么也测不出。
      (这是本项目第二次踩同一个坑; 上一次是 lm4 的 `b_only = 1.0000`。)
      步数才对前缀敏感。
    """
    def steps(seq):
        tot = 0
        for _ in range(n_ep):
            mdp.reset()
            n = 0
            done = False
            for x in seq:
                _, _, done = mdp.step(x)
                n += 1
                if done:
                    break
            while not done and n < mdp.horizon * 3:
                _, _, done = mdp.step(mdp.greedy_action())
                n += 1
            tot += n
        return tot / n_ep

    S_ab = steps([a, b]); S_ba = steps([b, a])
    S_a = steps([a]); S_b = steps([b])
    # 基线: 什么都不强迫
    S_none = steps([])
    I_ab, I_ba = S_b - S_ab, S_a - S_ba
    return dict(a=a, b=b, S_none=S_none, S_a=S_a, S_b=S_b, S_ab=S_ab, S_ba=S_ba,
                I_ab=I_ab, I_ba=I_ba, asym=I_ab - I_ba)


def has_temporal_structure(mdp, n_ep=200, seed=0, tol=0.5):
    """对全部有序动作对测 A(a,b), 返回是否**存在**顺序依赖。"""
    import itertools
    rows = [interaction_asymmetry(mdp, a, b, n_ep=n_ep, seed=seed)
            for a, b in itertools.permutations(range(N_ACT), 2)]
    A = np.array([r["asym"] for r in rows])
    frac = float((np.abs(A) > tol).mean())
    return rows, A, frac
