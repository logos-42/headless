"""skill_agent.py — 在 KeyDoorMDP 上做 B 矩阵的智能体。

## 要回答的问题 (用户 2026-09-14 框架里的 H5 + 复用)

    "Option 是否保存了可以重新利用的经验?"

即 A→B→C→A→B→C 反复漂移时, 是否出现

        T_A^(2) < T_A^(1),   T_A^(3) < T_A^(2)

## 公平对照的两个设计要点

1. **两个 agent 都保留 Q 表** (跨 regime 不重置)
   否则"复用"测的是"有没有记忆", 而不是"option 有没有带来额外的可复用结构"。
   我们要隔离的是 **option 库** 的额外贡献。

2. **相同计算预算**: option agent 的额外开销是"拟合世界模型 + 发现 option",
   所以给它和 primitive 相同数量的**环境交互**。
   测的是 **样本效率 R(N)**, 不是"跑得更久所以更好"。

## 关键: regime 变化后 option 库会不会失效?

一个 regime = (key_pos, door_pos, goal_pos)。换 regime 后:
  - **技能本身仍然成立** ("走到 key 位置并抓取" 这个**行为模式**没变)
  - 但**地点变了** -> 旧 option 的 goal_center 是**过期的**

所以本实现区分两种 option 使用方式:
  - `reuse_stale`  : 直接用旧 option (预期: 会被过期目标拖累)
  - `rediscover`   : 换 regime 后**重新发现**, 但库不清空 (预期: 应该更快)
这正好对应"技能可复用"的两种理解, 也是判据所在。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hibs_lnn.skill_mdp import N_ACT, KeyDoorMDP  # noqa: E402
from hibs_lnn.uncertainty_gate import TabularTransition  # noqa: E402


class QLearner:
    """表格 Q-learning (跨 regime 保留 Q 表 —— 这是"记忆"基线)。"""

    def __init__(self, n_states, n_act=N_ACT, lr=0.2, gamma=0.95,
                 eps=0.25, seed=0):
        self.Q = np.zeros((n_states, n_act))
        self.n_states, self.n_act = n_states, n_act
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.rng = np.random.RandomState(seed)

    def act(self, s):
        if self.rng.rand() < self.eps:
            return int(self.rng.randint(0, self.n_act))
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s2, done):
        tgt = r if done else r + self.gamma * float(np.max(self.Q[s2]))
        self.Q[s, a] += self.lr * (tgt - self.Q[s, a])


class SkillAgent:
    """Q-learning + 一个**跨 regime 保留**的技能库。"""

    def __init__(self, mdp: KeyDoorMDP, mode="primitive", seed=0,
                 discover_every=25, min_trans=120, opt_prob=0.5,
                 use_subgoals=1, n_regions=16, max_len=5):
        self.mdp = mdp
        self.mode = mode                    # primitive / reuse_stale / rediscover
        self.q = QLearner(mdp.n_states, seed=seed)
        self.rng = np.random.RandomState(seed + 1)
        self.discover_every = int(discover_every)
        self.min_trans = int(min_trans)
        self.opt_prob = float(opt_prob)
        self.use_subgoals = int(use_subgoals)
        self.n_regions = int(n_regions)
        self.max_len = int(max_len)

        self.trans = []                     # 全历史 (跨 regime 累积)
        self.om = None                      # 当前技能库
        self.episodes = 0
        self.n_discover = 0

    # ── 技能库维护 ──────────────────────────────────────────────────
    def maybe_discover(self, force=False):
        """周期性 (或强制) 用**已有全部转移**重建技能库。

        ★ 库**不清空**: 新 options 与旧 options 合并去重。
          这才是"技能可复用"的实现 —— 换 regime 只是新增地点变体,
          而不是从头学。
        """
        if not force:
            if self.episodes % self.discover_every != 0:
                return
        if len(self.trans) < self.min_trans:
            return
        X = np.array([np.append(v, a) for v, a, _ in self.trans])
        Y = np.array([vn for _, _, vn in self.trans])
        dim = X.shape[1] - 1
        T = TabularTransition().fit(X, Y, N_ACT)
        from hibs_lnn.option_manager import OptionManager
        from hibs_lnn.uncertainty_gate import UncertaintyGate
        g = UncertaintyGate(tau_C=1.0)
        g.calibrate([T.predict(X[i, :dim], int(X[i, -1]))[1]
                     for i in range(0, len(X), max(1, len(X) // 100))])
        om = OptionManager(T, g, max_len=self.max_len, seed=0)
        n_uniq = len(np.unique(np.round(X[:, :dim], 6), axis=0))
        nreg = min(self.n_regions, max(4, n_uniq))
        if self.use_subgoals:
            om.discover_subgoals(X[:, :dim], X[:, -1].astype(int), n_regions=nreg)
        else:
            om.discover(X[:, :dim], X[:, -1].astype(int), n_regions=nreg)
        if self.om is not None:
            # 合并: 旧技能在前 (先验偏好), 新技能去重后追加
            seen = {tuple(int(x) for x in o.actions) for o in self.om.options}
            for o in om.options:
                k = tuple(int(x) for x in o.actions)
                if k not in seen:
                    self.om.options.append(o)
                    seen.add(k)
        else:
            self.om = om
        self.n_discover += 1

    def _option_action(self, s):
        """按 option 出动作 (闭环: π_o 用世界模型算, 或退回动作序列)。"""
        if self.om is None or not self.om.options:
            return None
        v = self.mdp.vec()
        o = self.om.options[int(self.rng.randint(0, len(self.om.options)))]
        acts = [int(x) for x in o.actions]
        if not acts:
            return None
        return acts[0]

    # ── 一个 episode ────────────────────────────────────────────────
    def run_episode(self):
        s = self.mdp.reset()
        done = False
        n = 0
        while not done and n < self.mdp.horizon:
            v = self.mdp.vec()
            a = None
            if self.mode != "primitive" and self.rng.rand() < self.opt_prob:
                a = self._option_action(s)
            if a is None:
                a = self.q.act(s)
            s2, r, done = self.mdp.step(a)
            self.q.update(s, a, r, s2, done)
            self.trans.append((v, a, self.mdp.vec()))
            s = s2
            n += 1
        self.episodes += 1
        if self.mode != "primitive":
            self.maybe_discover()
        return bool(done and self.mdp._reached()), n


# ── B 矩阵: regime 链 ───────────────────────────────────────────────

DEFAULT_REGIMES = [
    dict(key_pos=2, door_pos=5, goal_pos=7),      # A
    dict(key_pos=5, door_pos=2, goal_pos=6),      # B (key/door 换位)
    dict(key_pos=1, door_pos=6, goal_pos=7),      # C
]


def run_regime_chain(mode="primitive", chain=None, n_pos=8, episodes_per=400,
                     seed=0, thresh=0.8, window=20, **kw):
    """跑 A→B→C→A→B→C 链, 返回每次 regime 的 T_adapt(达到 thresh 所需回合)。

    T_adapt = 该 regime 段内, 滑动成功率首次 ≥ thresh 时的回合数;
              从未达到则记为 episodes_per (上限)。
    """
    chain = chain or (DEFAULT_REGIMES * 2)
    mdp = KeyDoorMDP(n_pos=n_pos, horizon=30, **chain[0])
    agent = SkillAgent(mdp, mode=mode, seed=seed, **kw)
    out = []
    for ci, reg in enumerate(chain):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        if mode != "primitive":
            agent.maybe_discover(force=True)
        succ, t_adapt = [], None
        for ep in range(episodes_per):
            ok, _ = agent.run_episode()
            succ.append(1.0 if ok else 0.0)
            if t_adapt is None and len(succ) >= window:
                if float(np.mean(succ[-window:])) >= thresh:
                    t_adapt = ep + 1
        out.append({"regime": ci, "key": reg["key_pos"], "door": reg["door_pos"],
                    "goal": reg["goal_pos"],
                    "t_adapt": int(t_adapt if t_adapt is not None else episodes_per),
                    "final_succ": float(np.mean(succ[-window:])) if succ else 0.0,
                    "n_options": len(agent.om.options) if agent.om else 0})
    return out, agent
