# -*- coding: utf-8 -*-
"""`RRSkillAgent` —— OaK 第一步(SubTask → Option)在 KeyDoor 上的 agent 侧实现。

**为什么是子类**:`SkillAgent.run_episode` 是所有实验臂的公共路径。把 rr 逻辑
写进父类会改动正在跑的管线;写成子类则:

  * `mode != "rr"` 时**逐字委托** `super().run_episode()` → primitive /
    rediscover / macro 三个对照臂与原来**逐位相同**(对照干净);
  * 部署只是**新增一个文件**, 不修改任何现有文件。

## 与 OpenLoop 的区别(§1, Pitfall 35)

    a_t = π_o(s_t)     ← **每一步都按当前状态重算**(本文件)
    a_{t+i} = a_i^o    ← 固定动作序列replay(不是时间抽象)

`π_o` 来自子任务的价值迭代解,存成表;执行时**每个状态查一次表**, 所以
`recompute-count == 执行步数`。

## 四条终止判据(§6, 全部必需)

    ① goal_reached  特征达成(x_i(s)=1)        ← 子任务的**目的**
    ② beta_stop     表中 β_o(s)=True(停车最优)
    ③ expired       **硬到期**(由 option 自身的名义时长导出, 不是拍脑袋常数)
    ④ stall         连续 stall_patience 步状态不变(撞墙空转)

    另有 ⑤ devalued: 主任务价值追上后增益 ≤ 0 → 主动放弃。

到期是**独立的第四条**而不是前三条的兜底 —— 前三条各自会因自己的原因不触发
(容差尺度错 / 台账空 / 阈值没到), 到期把损害封顶。
"""
import numpy as np

from hibs_lnn.rr_options import (
    KeyDoorTabular, build_bottleneck_library, build_random_goal_library,
    build_rr_library, initiation_ok, main_value, option_gain,
)
from hibs_lnn.skill_agent import SkillAgent
from hibs_lnn.subtask_options import STOP


class RRSkillAgent(SkillAgent):
    """在 `SkillAgent` 上增加 `mode="rr"`(reward-respecting subtask options)。

    消融旋钮(每个对应矩阵里的一臂):
        rr_v_main   "exact" | "learned"   价值函数来源(精确 VI / agent 自己的 Q)
        rr_z_mode   "rr" | "zeroV" | "nobonus"
        rr_cumulant "reward" | "zero"
        rr_feats    ("key","door") | ("key",) ...
        rr_bonus    w̄(浮点, 扫录用)
        rr_lib      "rr" | "bottleneck" | "random"   换整库(负对照)
    """

    def __init__(self, mdp, mode="rr", seed=0, rr_v_main="exact",
                 rr_z_mode="rr", rr_cumulant="reward", rr_feats=("key", "door"),
                 rr_bonus=1.0, rr_lib="rr", rr_max_duration=None,
                 stall_patience=4, rr_select_rule="uniform", **kw):
        super().__init__(mdp, mode=mode, seed=seed, **kw)
        self.rr_v_main = rr_v_main
        self.rr_z_mode = rr_z_mode
        self.rr_cumulant = rr_cumulant
        self.rr_feats = tuple(rr_feats)
        self.rr_bonus = float(rr_bonus)
        self.rr_lib = rr_lib
        # ★ 选择规则**所有臂共用同一个** —— 这样臂间的差异只能来自"子任务内容",
        #   而不是来自某个选择启发式。规则本身是**另一条消融轴**。
        #     "uniform" 在可启动的 option 里均匀抽(每个库都有同等曝光)
        #     "gain"    argmax(子任务价值 − V_main), 只在 > 0 时启动
        #   §3 的反面教训: 用 `V_main` 做增益基线时, `c=−1` 的瓶颈 option
        #   价值为负 -> 永不启动 -> 负对照臂变成**空臂**(不是"更差")。
        self.rr_select_rule = rr_select_rule
        self.rr_max_duration = None if rr_max_duration is None else int(rr_max_duration)
        self.stall_patience = int(stall_patience)

        self.rr_lib_obj = None
        self.rr_mdl = None
        self._rr_regime = None
        self._rr_cap = {}          # option 名 -> 硬到期上限
        # ── 计数器(§3: 模式的机制差异必须在**跑了之后**可被证伪) ──
        self.rr_stats = {
            "recomputes": 0,        # π_o 重算次数(闭环执行 ≈ 执行步数)
            "selections": 0,        # 启动次数
            "exec_steps": 0,        # 在 option 内的总步数
            "terms": {},            # 终止原因台账(**必须非空**)
            "durations": [],        # 每次执行的时长(检 lock-in)
            "stall_baseline": [],   # 停滞统计的基线样本
            "rebuilds": 0,
        }

    # ── 技能库(每个 regime 重建一次 —— 目标随 regime 移动) ──────────
    def _rr_ensure(self):
        reg = tuple(self.mdp.regime)
        if self._rr_regime == reg and self.rr_lib_obj is not None:
            return
        mdl = KeyDoorTabular(self.mdp, gamma=self.q.gamma)
        self.rr_mdl = mdl
        if self.rr_v_main == "exact":
            V_main = main_value(mdl)
        elif self.rr_v_main == "learned":
            # agent 自己的 Q 表 —— 在线版, 不需要 oracle
            V_main = np.max(self.q.Q, axis=1).copy()
        else:
            raise ValueError(f"unknown rr_v_main: {self.rr_v_main}")
        self._rr_v_main_arr = V_main

        if self.rr_lib == "rr":
            lib = build_rr_library(mdl, V_main, bonus_weight=self.rr_bonus,
                                   feats=self.rr_feats,
                                   cumulant=self.rr_cumulant,
                                   z_mode=self.rr_z_mode)
        elif self.rr_lib == "bottleneck":
            lib = build_bottleneck_library(mdl)
        elif self.rr_lib == "random":
            lib = build_random_goal_library(mdl, V_main, seed=self.rng.randint(1 << 30))
        else:
            raise ValueError(f"unknown rr_lib: {self.rr_lib}")

        self.rr_lib_obj = lib
        self._rr_regime = reg
        self.rr_stats["rebuilds"] += 1
        # 每个 option 的硬到期上限: **由它自己的名义时长导出**。
        # 名义时长 = 从起始状态出发、按 π_o 走、直到终止判据命中所需步数
        # (模型确定性, 直接模拟)。
        self._rr_cap = {}
        for st in lib:
            nom = self._rr_nominal(st)
            cap = int(self.rr_max_duration) if self.rr_max_duration else max(3, 2 * nom)
            self._rr_cap[st.name] = cap

    def _rr_nominal(self, st):
        """模拟 π_o 从**回合起始状态 0** 出发到自己终止要多少步。

        模型确定性, 直接前向模拟。这个数是 `expired` 上限的依据 ——
        §6 要求上限由 option 自身的名义时长导出, 不是拍脑袋常数。
        """
        s = 0
        for k in range(1, self.mdp.horizon + 1):
            if st.policy[s] == STOP:
                return k
            nxt = self.rr_mdl.T[s][st.policy[s]][0][1]
            if self.rr_mdl.is_goal[nxt] or float(getattr(st, "feature")[nxt]) > 0:
                return k
            if st.beta[nxt]:
                return k
            if nxt == s:
                return k
            s = nxt
        return self.mdp.horizon

    # ── 选择 ────────────────────────────────────────────────────────
    def _rr_select(self, s):
        """选一个 option。规则由 `rr_select_rule` 决定(所有臂共用)。

        "uniform" —— 在**可启动**(I_o 成立)的 option 里均匀抽。
                      每个库都有同等曝光, 所以臂间差异只能来自子任务内容。
        "gain"    —— `argmax`(子任务价值 − V_main), 且增益必须 **> 0**。
                      这是"哪个子任务现在最值得追"的贪心读法, 本身是一条
                      消融轴(更强的先验 = 更多 oracle 信息)。
        """
        cand = [st for st in self.rr_lib_obj if initiation_ok(st, s)]
        if not cand:
            return None
        if self.rr_select_rule == "gain":
            best, best_g = None, 0.0
            for st in cand:
                g = option_gain(st, s, self._rr_v_main_arr)
                if g > best_g:
                    best, best_g = st, g
            return best
        return cand[int(self.rng.randint(len(cand)))]

    # ── 一个回合(rr 路径; 其他模式逐字委托父类) ──────────────────
    def run_episode(self):
        if self.mode != "rr":
            return super().run_episode()          # 对照臂路径**逐字不变**
        self._rr_ensure()
        s = self.mdp.reset()
        done, n = False, 0
        active, o_steps, prev_s, stuck = None, 0, None, 0
        while not done and n < self.mdp.horizon:
            v0 = self.mdp.vec()        # ★ **步前**状态 —— 与父类 run_episode 同约定。
            a = None
            if self.rng.rand() < self.opt_prob or active is not None:
                # 启动
                if active is None:
                    active = self._rr_select(s)
                    o_steps, prev_s, stuck = 0, s, 0
                    if active is not None:
                        self.rr_stats["selections"] += 1
                # 出动作: **每步按当前状态重算 π_o**(闭环)
                if active is not None:
                    if int(active.policy[s]) != STOP:
                        a = int(active.policy[s])
                        self.rr_stats["recomputes"] += 1
                    else:
                        self._rr_settle(active, o_steps, "beta_stop")
                        active, o_steps = None, 0
            if a is None:
                if active is not None:
                    self._rr_settle(active, o_steps, "devalued")
                    active, o_steps = None, 0
                a = self.q.act(s)
            s2, r, done = self.mdp.step(a)
            self.q.update(s, a, r, s2, done)
            n += 1
            self.trans.append((v0, a, self.mdp.vec()))    # (步前, 动作, 步后)
            # ── 结算判据: 全部在**新状态**上评估(正确语义) ──────────
            if active is not None:
                o_steps += 1
                self.rr_stats["exec_steps"] += 1
                reason = None
                # ★ 主判据 = **option 自己的 β_o**(由 stopping value z 导出)。
                #   这是机制本体 —— 之前把"特征达成"排在 β 之前, 导致所有
                #   变体在特征亮起的同一瞬间就停, z 永远轮不到起作用,
                #   5 个 rr 变体的台账**逐位相同**(Pitfall-26 tell)。
                #   "恰好停在特征上"现在作为 β_o 的一个**子理由**计数,
                #   这样既能证明机制在跑, 又能看出它停在哪。
                if bool(active.beta[s2]):
                    reason = ("beta_at_feature" if float(active.feature[s2]) > 0
                              else "beta_elsewhere")
                elif done:
                    reason = "horizon_end"
                elif o_steps >= self._rr_cap.get(active.name, self.mdp.horizon):
                    reason = "expired"             # ③ 硬到期
                elif s2 == prev_s:
                    stuck += 1
                    self.rr_stats["stall_baseline"].append(stuck)
                    if stuck >= self.stall_patience:
                        reason = "stall"           # ④ 撞墙空转
                else:
                    stuck = 0
                if reason:
                    self._rr_settle(active, o_steps, reason)
                    active, o_steps = None, 0
                prev_s = s2
            s = s2
        if active is not None:                     # 回合结束仍在执行
            self._rr_settle(active, o_steps, "horizon_end")
        self.episodes += 1
        return bool(done and self.mdp._reached()), n

    def _rr_settle(self, st, steps, reason):
        self.rr_stats["terms"][reason] = self.rr_stats["terms"].get(reason, 0) + 1
        self.rr_stats["durations"].append(int(steps))

    # ── 台账 ────────────────────────────────────────────────────────
    def rr_bookkeeping(self):
        d = list(self.rr_stats["durations"])
        base = self.rr_stats["stall_baseline"]
        return {
            "recomputes": self.rr_stats["recomputes"],
            "selections": self.rr_stats["selections"],
            "exec_steps": self.rr_stats["exec_steps"],
            "terms": dict(self.rr_stats["terms"]),
            "rebuilds": self.rr_stats["rebuilds"],
            "n_options": 0 if self.rr_lib_obj is None else len(self.rr_lib_obj),
            "mean_duration": float(np.mean(d)) if d else 0.0,
            "max_duration": int(np.max(d)) if d else 0,
            "horizon": int(self.mdp.horizon),
            # ★ lock-in 检: 平均执行时长接近整回合 = 不是抽象而是锁死。
            #   阈值取自 option 自身名义时长, 所以也一并报出来。
            "lockin_ratio": (float(np.mean(d)) / max(1, self.mdp.horizon)) if d else 0.0,
            "max_cap": (int(max(self._rr_cap.values())) if self._rr_cap else 0),
            "stall_trigger_frac": (float(np.sum([1 for b in base if b >= self.stall_patience])) / max(1, len(base))) if base else 0.0,
            "beta_nonempty": 0 if self.rr_lib_obj is None else
                             int(sum(int(np.sum(o.beta)) for o in self.rr_lib_obj)),
        }
