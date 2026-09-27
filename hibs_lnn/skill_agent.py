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
                 eps=0.15, seed=0, q_init=1.0):
        # ★ **乐观初始化** —— 稀疏奖励下这是必需的, 不是调参技巧。
        #   若 Q 全为 0, `argmax` 恒返回动作 0, agent 会**一直做同一个动作**
        #   (实测: 在 KeyDoor 上一直 left, 卡在 pos 0), 只有 ε-随机步才动,
        #   于是几乎永远到不了目标 -> 600 回合成功率仍为 0.00。
        #   初值取正 (q_init=+1) 使每个 (s,a) 都先被尝一次, 由更新把它拉回真值。
        self.Q = np.full((n_states, n_act), float(q_init))
        self.n_states, self.n_act = n_states, n_act
        self.lr, self.gamma, self.eps = lr, gamma, eps
        self.q_init = float(q_init)
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
                 use_subgoals=1, n_regions=16, max_len=5, term_eps=None):
        self.mdp = mdp
        self.mode = mode                    # primitive / reuse_stale / rediscover / macro
        self.q = QLearner(mdp.n_states, seed=seed)
        self.rng = np.random.RandomState(seed + 1)
        self.discover_every = int(discover_every)
        self.min_trans = int(min_trans)
        self.opt_prob = float(opt_prob)
        self.use_subgoals = int(use_subgoals)
        self.n_regions = int(n_regions)
        self.max_len = int(max_len)
        # ★ β_o 的"目标达成"阈值必须与**状态空间尺度**匹配。
        #   KeyDoor 的 vec() = one-hot(pos) ⊕ has_key ⊕ door_open (n_pos+2 维)。
        #   **两个不同位置之间的 L2 距离是 √2 ≈ 1.414**, 而到达"某个目标区域"
        #   的合理判据是"落进该区域", 该区域半径就是 OptionManager 的
        #   `init_radius`(按区域最近邻距离中位数 * 0.75 自适应推断)。
        #   固定 eps=0.05 (给"归一化距离"用的量级) 在这里**几乎永不满足**
        #   -> β_o 只在预算耗尽时触发 -> success 恒 0 -> 技能库从不学习。
        #   这是本项目第 5 次同型错误("阈值与状态空间尺度不匹配"),
        #   故此处**默认 None = 运行时取 om.init_radius**, 不再手写常数。
        self.term_eps = None if term_eps is None else float(term_eps)

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

    def _select_option(self, v):
        """选一个可启动的 option(与 _option_action 同源, 抽出复用)。"""
        if self.om is None or not self.om.options:
            return None
        goal_vec = None
        try:
            key_pos, door_pos, goal_pos = self.mdp.regime
            gv = np.zeros_like(v)
            gv[goal_pos] = 1.0
            gv[self.mdp.n_pos] = 1.0      # has_key = 1
            gv[self.mdp.n_pos + 1] = 1.0  # door_open = 1
            goal_vec = gv
        except Exception:
            pass
        return self.om.select(v, goal=goal_vec, exclude_unc=True)

    def _option_action(self, s):
        """按 option 出动作 —— **必须尊重启动集 I_o**。

        ★ 原来的写法是「从技能库里**均匀随机**挑一个 option, 取它的第一个动作」。
          这既不看 `I_o`(启动集), 也不管目标是否相关, 等于往动作里注入
          一半的噪声。实测后果 (KeyDoor B 矩阵):
            技能库涨到 54~60 个后, rediscover 在全部 6 段上都差于 primitive,
            其中两段**永不收敛** (T_adapt = 上限 300)。

          正确做法 (用户给的 `Option=(I_o,g_o,π_o,β_o)`):
            用 `OptionManager.select(s, goal=...)` —— 它**会**检查 `initiable(s)`,
            也就是"当前状态是否落在该 option 的启动集里"。这才是 `I_o` 的语义。

        ★★ 本函数是**逐步顾问**语义 (每步问一次, 执行一个 primitive 动作)。
            它不是时间抽象 —— option 的多步结构完全没用上, 成败也无从统计
            (`Option.visits/success` 因此永远是 0, `select()` 里
            `rate = success/visits` 恒取默认 0.5, 技能库**从不学习哪些 option 管用**)。
            真正的宏执行见 `_macro_step`。
        """
        o = self._select_option(self.mdp.vec())
        if o is None:
            return None          # 没有可启动的技能 -> 交回 Q (这才是正确的退让)
        acts = [int(x) for x in o.actions]
        if not acts:
            return None
        # 闭环: 用世界模型算"朝该 option 目标前进最多"的动作; 失败则退回动作序列
        a = None
        if self.om.T is not None:
            a = self._goal_action(self.mdp.vec(), o.goal_center)
        if a is None:
            a = acts[0]
        return int(a)

    def _macro_step(self, v, active, steps):
        """把一个 option 当**宏动作**执行。

        返回 `(动作, 新的 active, 新的 steps)`。`active is None` 表示本步先选一个
        option 再出动作; 每步都重算 `π_o` (闭环), 当 `β_o` 命中
        (目标达成 / 步数上限) 时结算成败并交回 base policy。

        ## 为什么要这个

        原 `rediscover` 把 option 当**单步动作顾问**: 每步按概率问一次 "现在该做
        什么动作", option 的 `(I_o, g_o, π_o, β_o)` 里只有 `π_o` 的一步被用到。
        后果有两个, 都是实测到的:

          1. **成败无从统计** -> `visits/success` 恒 0 -> `select()` 的
             `rate = success/visits` 恒为默认 0.5 -> 54~60 个 option 在启动集内
             **同分**, 选择等于抛硬币 -> 全部 6 段都差于 primitive。
          2. **不是时间抽象** -> 逐步咨询等于给 base policy 注入噪声,
             没有任何"一段行为被当作一个决策单位"的效果。
        """
        # 目标达成阈值: 未显式给定时, 用 OptionManager 自适应推断的
        # `init_radius`(区域尺度) —— 同一把尺子量启动集与目标区。
        eps = self.term_eps
        if eps is None:
            eps = float(getattr(self.om, "init_radius", 0.75))
        if active is None:
            active = self._select_option(v)
            steps = 0
            if active is None:
                return None, None, 0
        a = self._goal_action(v, active.goal_center) if self.om.T is not None else None
        if a is None:
            a = int(active.actions[0]) if active.actions else None
        steps += 1
        # β_o: 目标达成 或 预算耗尽 -> 结算
        try:
            reached = active.goal_distance(v) < eps
        except Exception:
            reached = False
        if reached or steps >= max(1, int(self.max_len)):
            active.visits += 1
            if reached:
                active.success += 1
            active = None
            steps = 0
        return a, active, steps

    def _goal_action(self, v, g):
        """π_o(s) = argmax_a [ ‖g−s‖ − ‖g−T(s,a)‖ ], 每步重算。"""
        g = np.asarray(g, dtype=float)
        d_now = float(np.linalg.norm(v - g))
        best_a, best_gain = None, -np.inf
        for a in range(N_ACT):
            sn, unc = self.om.T.predict(v, a)
            if not np.isfinite(unc):
                continue
            gain = d_now - float(np.linalg.norm(np.asarray(sn, dtype=float) - g))
            if gain > best_gain:
                best_a, best_gain = a, gain
        return best_a

    # ── 一个 episode ────────────────────────────────────────────────
    def run_episode(self):
        s = self.mdp.reset()
        done = False
        n = 0
        # ── macro 模式的跨步状态: 当前正在执行的宏动作 ──────────────
        active, o_steps = None, 0
        while not done and n < self.mdp.horizon:
            v = self.mdp.vec()
            a = None
            if self.mode == "macro":
                # 宏执行: 选一个 option -> 跑到 β_o 触发 -> 结算成败
                if self.rng.rand() < self.opt_prob or active is not None:
                    a, active, o_steps = self._macro_step(v, active, o_steps)
            elif self.mode != "primitive" and self.rng.rand() < self.opt_prob:
                # rediscover / reuse_stale: 逐步顾问语义 (保持不变, 作为对照)
                a = self._option_action(s)
            if a is None:
                active, o_steps = None, 0     # 交回 base policy 时宏动作结束
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

    def option_bookkeeping(self):
        """技能库的成败台账(诊断用: 证明 visits/success 真的被回填了)。"""
        if self.om is None:
            return {"n_options": 0, "n_used": 0, "total_visits": 0, "total_success": 0}
        vs = [o.visits for o in self.om.options]
        ss = [o.success for o in self.om.options]
        return {"n_options": len(self.om.options),
                "n_used": int(sum(1 for v_ in vs if v_ > 0)),
                "total_visits": int(sum(vs)), "total_success": int(sum(ss))}


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
