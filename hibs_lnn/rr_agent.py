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
                 stall_patience=4, rr_select_rule="uniform",
                 rr_min_launches=6, rr_ucb_c=1.0, rr_calib_min_rate=0.5,
                 rr_persist=True, rr_forget_hl=float("inf"),
                 rr_score="rate", rr_pot="learned",
                 rr_calib_min_adv=0.0, rr_retire_rule="internal",
                 rr_calib_min_advq=0.0, rr_forgive_p=0.0,
                 rr_lr_v=0.2, **kw):
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
        # ══ 自我校准闭环(L2 缺的那一格)══════════════════════════════
        #  这个 dict 是**按 option 归属**的台账。在此之前整个仓库唯一的
        #  更新点是 skill_agent.py:240(旧 macro 路径)的全局计数器 ——
        #  `_rr_settle(self, st, steps, reason)` 收下 `st` 然后**完全忽略它**,
        #  所以库级台账对 rr 路径不存在, "这个 option 从不达成目标" 这个
        #  事实无处安放(L2 的 57 启动 / 56 expired / 1 reached 就是这样
        #  被埋掉的: 没有任何东西因为该事实而改变)。
        #  信号取 **option 自己的达成率**(β_at_feature), 不取主奖励 ——
        #  KeyDoor 的奖励是稀疏的(只在终点 +1), 拿主奖励做 option 级信号
        #  会退化成"所有 option 都得 0"。
        self.rr_outcome = {}        # option 名 -> {"n", "reached", "steps"}
        self.rr_min_launches = int(rr_min_launches)   # 判"坏了"所需的最少启动数
        self.rr_ucb_c = float(rr_ucb_c)               # 校准选择的探索常数
        self.rr_calib_min_rate = float(rr_calib_min_rate)  # 达成率低于此 = 机制坏了
        # ══ 持续性的两个旋钮(「持续自我校准」的"持续")══════════════════
        #  rr_persist     台账是否跨 regime 存活。
        #                 False -> 每次库重建就清空(只在单 regime 内校准)
        #                 True  -> 账随 agent 走(校准成为**跨任务**的知识)
        #  rr_forget_hl   **忘记半衰期**(以启动次数计)。inf = 不忘记 = 终身平均。
        #
        #  ★ 为什么持久化必须配遗忘, 否则持久化是**有害**的:
        #    无遗忘时 n 会跨 regime 累积到几百, 于是
        #      (a) 老的成功把新 regime 的失败稀释掉 —— 一个在第 2 个 regime
        #          已经坏掉的 option, 终身达成率仍可能是 0.9, 检测不出来;
        #      (b) UCB 探索项 c·sqrt(ln N/(n+1)) 被大 n 压死 —— agent 不再
        #          去重新检验它, 陈旧的判断被永久锁住。
        #    「持续校准」必须跟踪的是**当前胜任度**, 不是历史平均。
        #    实现: 有效计数按 decay = 0.5^(1/hl) 指数衰减。
        #    hl=inf -> decay=1.0 -> n_eff ≡ n, 逐位复现旧行为(向后兼容)。
        self.rr_persist = bool(rr_persist)
        self.rr_forget_hl = float(rr_forget_hl)
        #  rr_score  选择分数用什么信号
        #     "rate" 达成率(option 自己的成功)         —— 机制内部视角
        #     "eff"  达成次数 / 消耗步数               —— **把主任务的成本接进来**
        #            奖励稀疏时主奖励做不了 option 级信号, 但**时间是通用货币**:
        #            视野有限, 白走的路严格是损失。所以"每步产出多少达成"是
        #            一个不需要 oracle、又对主任务有意义的信号。
        self.rr_score = rr_score
        if rr_score not in ("rate", "eff", "adv", "advq"):
            raise ValueError(f"unknown rr_score: {rr_score}")
        # ══ 主任务信号(§8 明写的那个缺口, 现在被实测证据逼出来了)══════
        #  为什么必须有: 只用 **option 自己的判据**(达成率)做校准是**错的**。
        #  实测: 把一个 option 的上限压到 1 步 -> 它"达不成自己的子目标"
        #  (达成率 0.2), 但**它执行的那一步仍然是 π_o 的最优步**, 对主任务
        #  有益。按内部判据把它淘汰 -> 白损失 π_o 的好步 -> 步数反而差
        #  1.0 步(3 seed 一致)。**校准到错误的信号上, 会淘汰有用的东西。**
        #
        #  主任务信号取 **势函数差分 / 步数**:
        #      progress = Φ(终止状态) − Φ(起始状态),  adv = Σprogress / Σsteps
        #  这是 SMDP 视角下"这个 option 每消耗一步, 把主任务推进了多少"。
        #  Φ 取 **agent 自己的 max_a Q[s,a]** —— 在线、不需要 oracle。
        #  (早期 Q 是乐观常数 -> adv≈0 -> 判定"无信息", 这是诚实的行为;
        #   它随学习变得有信息。`rr_pot="exact"` 用模型 VI 的 V_main 作对照,
        #   用来分离"信号质量"与"机制结构"两个因素。)
        self.rr_pot = rr_pot
        if rr_pot not in ("learned", "exact", "gvf"):
            raise ValueError(f"unknown rr_pot: {rr_pot}")
        self.rr_calib_min_adv = float(rr_calib_min_adv)
        #  rr_retire_rule  用哪条判据决定淘汰
        #    "internal" 只看 option 自己的达成率      (旧行为)
        #    "adv"      只看主任务推进量
        #    "both"     两条都判坏才淘汰             (保守)
        #    "either"   任一条判坏就淘汰             (激进)
        self.rr_retire_rule = rr_retire_rule
        if rr_retire_rule not in ("none", "internal", "adv", "advq", "both",
                                  "either", "all"):
            raise ValueError(f"unknown rr_retire_rule: {rr_retire_rule}")
        self.rr_regime_idx = -1        # 已见过的 regime 数(诊断用)
        self.rr_calib_hist = []        # [{'regime': i, 'broken': [...], 'usage': {...}}]
        # ══ option 的**学习价值**(SMDP)══════════════════════════════════
        #  ★ 这是对"在线可用势函数"问题的正确解 —— 不是找一个更好的 Φ,
        #    而是**取消 Φ 这个中间量**,直接学 option 自己的价值。
        #
        #  为什么 Φ 这条路走不通:
        #    `adv = [Φ(s_end) − Φ(s_start)] / steps` 把两件事混在一起 ——
        #      (a) option 把我移动了
        #      (b) agent 在学习, 所以 Φ 自己变了
        #    而且乐观初始化下 Φ(s_start) = Φ(s_end) = 常数 -> 恒等于 0。
        #
        #  正确的量是 **advantage**:
        #      A(s, o) = Q_opt(s, o) − V_base(s),   V_base(s) = max_a Q[s, a]
        #  它直接回答"从 s 执行这个 option, 比贪心走基元动作好多少" ——
        #  这正是"要不要留它"该问的问题。阈值 0 在**同一奖励尺度**上有意义,
        #  而且两边**从同一个初值出发**(都 q_init), 所以早期 A ≡ 0 =
        #  "还没有信息"(正确行为, 不是 bug)。
        #
        #  SMDP 更新(每次 option 结算时):
        #      Q_opt[s0, o] += lr · [r_opt + γ^k · V(s1) − Q_opt[s0, o]]
        #  r_opt = option 执行期间累计的**主任务**奖励, k = 执行步数。
        #  部分执行(到期/停滞/被中断)按截断处理, 仍是合法 bootstrap。
        #
        #  ★ 判别器为什么**从数学里自己掉出来**(不是手设计的阈值):
        #    inert(上限压到1步, 但那1步是好的): Q_opt ≈ γ·V(s') 且 V(s') ≈ V(s)/γ
        #                                        => A ≈ 0     -> 不淘汰 ✓
        #    harm (反转 π_o, 朝反方向走):        Q_opt ≈ γ·V(s_back), V(s_back) ≪ V(s)
        #                                        => A ≪ 0     -> 淘汰   ✓
        self.q_opt = None              # (n_states, n_opts) —— 惰性建表
        self._rr_oidx = {}             # option 名 -> 列号(跨 regime 稳定的身份)
        self.rr_calib_min_advq = float(rr_calib_min_advq)
        # ★ GVF 势函数: **中性初始化**(全 0)的状态价值表。
        #   为什么不能借用 `self.q.Q` 的 max: 那张表是乐观初始化,
        #   未试过的动作会把 max 顶在初值上 —— 实测 V(s0) 逐位 1.0000。
        self.rr_V = np.zeros(self.mdp.n_states, dtype=float)
        self.rr_lr_v = float(rr_lr_v)
        # ★ "犯错是可以原谅的" = **退休可逆**。被判坏的 option 以 p 的概率被重新检验;
        #   若其统计量恢复(它用的是同一批阈值判据), 下次 `rr_broken` 自然不再点名它。
        self.rr_forgive_p = float(rr_forgive_p)

    # ── 技能库(每个 regime 重建一次 —— 目标随 regime 移动) ──────────
    def _rr_ensure(self):
        reg = tuple(self.mdp.regime)
        if self._rr_regime == reg and self.rr_lib_obj is not None:
            return
        # ── regime 变了: 台账怎么处理 = 「持续」这条轴 ──────────────────
        if self._rr_regime is not None:            # 不是第一次
            self.rr_regime_idx += 1
            self.rr_calib_hist.append({
                "regime": self.rr_regime_idx,
                "broken": sorted(self.rr_broken()),
                "usage": self._rr_usage(),
            })
            if not self.rr_persist:
                # 只在本 regime 内校准 —— 每换一次世界, 经验清零
                self.rr_outcome = {}
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

        "uniform"    —— 在**可启动**(I_o 成立)的 option 里均匀抽。
                        每个库都有同等曝光, 所以臂间差异只能来自子任务内容。
        "gain"       —— `argmax`(子任务价值 − V_main), 且增益必须 **> 0**。
                        这是"哪个子任务现在最值得追"的贪心读法, 本身是一条
                        消融轴(更强的先验 = 更多 oracle 信息)。
        "calibrated" —— **自我校准**: 只用**自己的实测达成率**排序,
                        不用任何假设的价值。UCB1 形式:
                            score = reached/n + c·sqrt(ln(N+1)/(n+1))
                        并且**排除被判"坏了"的 option**(见 `rr_broken`)。
                        这是"犯错 -> 发现 -> 改正"的闭环本身, 与其它两条
                        规则的关键区别是: 它**不引入任何外部先验**,
                        "哪个 option 好"完全由 agent 自己的经验决定。
        """
        cand = [st for st in self.rr_lib_obj if initiation_ok(st, s)]
        if not cand:
            return None
        rule = self.rr_select_rule
        if rule == "gain":
            best, best_g = None, 0.0
            for st in cand:
                g = option_gain(st, s, self._rr_v_main_arr)
                if g > best_g:
                    best, best_g = st, g
            return best
        if rule == "calibrated":
            broken = self.rr_broken()
            live = [st for st in cand if getattr(st, "name", None) not in broken]
            # ★★ **原谅**(`rr_forgive_p`)—— "犯错是可以原谅的"的机制含义:
            #   被判坏的 option **不是被处决**, 而是以 p 的概率被**重新检验**。
            #
            #   没有这条路就有一个**死锁**: 早期 V 还没学到东西(bootstrap 暂态),
            #   option 的价值天然偏低 -> 判坏 -> 不再使用 -> 统计量永不更新 ->
            #   永远坏。实测就是这么崩的: advq 恒 -0.004, fallback 0.999。
            #   **退休必须可逆, 否则一次误判 = 永久损失。**
            dea = [st for st in cand if getattr(st, "name", None) in broken]
            if dea and self.rr_forgive_p > 0 and self.rng.rand() < self.rr_forgive_p:
                self.rr_stats["forgiven"] = self.rr_stats.get("forgiven", 0) + 1
                return dea[int(self.rng.randint(len(dea)))]
            if not live:                      # 全坏了 -> 退回基元层(诚实行为)
                return None
            N = sum(o.get("n_eff", 0.0) for o in self.rr_outcome.values())
            best, best_sc = None, -np.inf      # adv 分数可为负 -> 不能用 -1.0 做初值
            for st in live:
                o = self.rr_outcome.get(getattr(st, "name", None))
                if o is None:                 # 没试过的 option: 无限乐观, 先试它
                    return st
                n = o.get("n_eff", 0.0)
                sc = self._rr_score_of(o) + self.rr_ucb_c * float(
                    np.sqrt(np.log(N + 1.0) / (n + 1.0)))
                if sc > best_sc:
                    best, best_sc = st, sc
            return best
        return cand[int(self.rng.randint(len(cand)))]

    # ── 一个回合(rr 路径; 其他模式逐字委托父类) ──────────────────
    def run_episode(self):
        if self.mode != "rr":
            return super().run_episode()          # 对照臂路径**逐字不变**
        self._rr_ensure()
        s = self.mdp.reset()
        done, n = False, 0
        active, o_steps, prev_s, stuck, o_start = None, 0, None, 0, None
        o_rew = 0.0            # option 执行期间的**主任务**累计奖励(SMDP 更新用)
        while not done and n < self.mdp.horizon:
            v0 = self.mdp.vec()        # ★ **步前**状态 —— 与父类 run_episode 同约定。
            a = None
            if self.rng.rand() < self.opt_prob or active is not None:
                # 启动
                if active is None:
                    active = self._rr_select(s)
                    o_steps, prev_s, stuck, o_start, o_rew = 0, s, 0, s, 0.0
                    if active is not None:
                        self.rr_stats["selections"] += 1
                # 出动作: **每步按当前状态重算 π_o**(闭环)
                if active is not None:
                    if int(active.policy[s]) != STOP:
                        a = int(active.policy[s])
                        self.rr_stats["recomputes"] += 1
                    else:
                        self._rr_settle(active, o_steps, "beta_stop", o_start, s,
                                        r_opt=o_rew)
                        active, o_steps, o_start = None, 0, None
            if a is None:
                if active is not None:
                    self._rr_settle(active, o_steps, "devalued", o_start, s,
                                    r_opt=o_rew)
                    active, o_steps, o_start = None, 0, None
                a = self.q.act(s)
            s2, r, done = self.mdp.step(a)
            self.q.update(s, a, r, s2, done)
            # ★ GVF 势函数: **每个转移都喂**, 不论那一步是 option 出的还是基元出的。
            #   这就是 option 绕过 Q 也不影响它的原因 —— 它学的是"走到哪值多少",
            #   不是"哪个动作好"。
            self._rr_vstep(s, r, s2, done)
            n += 1
            self.trans.append((v0, a, self.mdp.vec()))    # (步前, 动作, 步后)
            # ── 结算判据: 全部在**新状态**上评估(正确语义) ──────────
            if active is not None:
                o_steps += 1
                o_rew += float(r)      # 主任务奖励记到**当前这个 option** 头上
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
                    self._rr_settle(active, o_steps, reason, o_start, s2,
                                    r_opt=o_rew, done=bool(done))
                    active, o_steps, o_start = None, 0, None
                prev_s = s2
            s = s2
        if active is not None:                     # 回合结束仍在执行
            self._rr_settle(active, o_steps, "horizon_end", o_start, s,
                            r_opt=o_rew, done=True)
        self.episodes += 1
        return bool(done and self.mdp._reached()), n

    # ── 有效计数(遗忘)──────────────────────────────────────────────
    def _rr_decay(self):
        """每次启动后, 旧统计量保留的比例。`hl=inf` -> 1.0(不遗忘)。"""
        hl = self.rr_forget_hl
        if hl is None or not np.isfinite(hl) or hl <= 0:
            return 1.0
        return float(0.5 ** (1.0 / hl))

    @staticmethod
    def _rr_rate(o):
        """当前胜任度 = reached_eff / n_eff。不遗忘时逐位等于终身平均。"""
        ne = o.get("n_eff", float(o.get("n", 0)))
        return (o.get("reached_eff", float(o.get("reached", 0))) / ne) if ne > 0 else 0.0

    @staticmethod
    def _rr_eff(o):
        """**效率** = 达成次数 / 消耗步数(每步产出多少达成)。

        这是不依赖 oracle 的"主任务成本"信号: 视野有限, 白走的路严格是损失。
        与 `rate` 的区别: 一个"总能达成但每次都绕远路"的 option,
        `rate` 高而 `eff` 低 —— 后者才是对主任务真正有意义的量。
        """
        return o.get("reached_eff", 0.0) / max(1.0, o.get("steps_eff", 1.0))

    def _rr_score_of(self, o):
        s = self.rr_score
        if s == "rate":
            return self._rr_rate(o)
        if s == "eff":
            return self._rr_eff(o)
        if s == "advq":
            return self._rr_advq_of(o)
        return self._rr_adv(o)

    @staticmethod
    def _rr_adv(o):
        """主任务推进量 = Σ势函数差分 / Σ步数(每步把主任务推进多少)。"""
        return o.get("adv_sum", 0.0) / max(1.0, o.get("adv_steps", 1.0))

    def _rr_pot(self, s):
        """势函数 Φ(s)。**必须是一个独立学习、中性初始化的状态价值预测**(GVF)。

        三个来源, 依次说明为什么只有第三个在线可用:

        `learned` —— `max_a Q[s,a]`, agent 的动作价值表取最大。
            实测**不可用**: 该表是**乐观初始化**(q_init=+1.0), 只要某个动作
            还没试过, `max_a` 就还是 1.0。实测 option 的启动状态上
            `V(s0)` **逐位等于 1.0000**(初值), 而 `γ^k·V(s1) = 0.717` ->
            advantage ≡ −0.275 —— **量出来的是初值 artifact, 不是 option 好坏**。
            (32 个状态里只有 14 个有任一 Q 离开初值, 但"最大值"几乎都还在初值上。)

        `exact` —— 模型 VI 的 `V_main`。干净, 但**是 oracle**(用了真转移模型),
            只能当"信号质量"的对照, 不能当在线方案。

        `gvf`   —— ★ **对"在线可用势函数"的答案**: 一张中性初始化(全 0)的
            状态价值表, 用 TD(0) 从 agent **自己走的每一步**学, 不管那一步是
            基元动作还是 option 内部的动作:

                V[s] += lr_v · [r + γ·V[s'] − V[s]]

            为什么这样就成了:
              · **中性初值** -> 没有"未试过的动作把 max 顶在初值"这个病;
              · 奖励稀疏(+1 只在终点) -> `V(s)` 自然收敛成 γ^(到目标的步数),
                这**正是**我们要的"离目标多远"的势函数;
              · **每个状态都被喂** -> option 绕过 Q 也没关系, 因为喂 V 的是
                每一步的真实转移, 不是动作选择;
              · 不用转移模型、不用目标位置的坐标 -> **无 oracle**。
        """
        if s is None:
            return 0.0
        if self.rr_pot == "exact":
            return float(self._rr_v_main_arr[s])
        if self.rr_pot == "gvf":
            return float(self.rr_V[s])
        return float(np.max(self.q.Q[s]))

    def _rr_vstep(self, s, r, s2, done):
        """GVF 势函数的一步 TD —— 每个转移都喂, 无论那一步是谁出的动作。"""
        tgt = float(r) if done else float(r) + float(self.q.gamma) * self.rr_V[s2]
        self.rr_V[s] += self.rr_lr_v * (tgt - self.rr_V[s])

    def _rr_has_base(self, s):
        """**势函数在该状态上是否有证据**(离开中性初值)。

        为什么必须有这个门: 在没有任何经验的区域上比较"option 比基线好不好",
        比出来的是**初值的形状**, 不是经验。这不是调阈值, 是判断的前提条件 ——
        "要判断这里用 option 划不划算, 前提是在这里有过经验"。
        """
        if s is None:
            return False
        if self.rr_pot == "gvf":
            return bool(abs(float(self.rr_V[s])) > 1e-12)
        return bool(np.any(self.q.Q[s] != self.q.q_init))

    # ── option 的学习价值(SMDP)──────────────────────────────────────
    def _rr_oindex(self, name):
        """option 名 -> `q_opt` 的列号。名跨 regime 稳定, 所以身份稳定。"""
        if name not in self._rr_oidx:
            self._rr_oidx[name] = len(self._rr_oidx)
        n_col = len(self._rr_oidx)
        if self.q_opt is None:
            # 与基元 Q **同一初值** —— 这样早期 advantage ≡ 0 = "还没有信息",
            # 而不是"所有 option 都很差"。
            self.q_opt = np.full((self.mdp.n_states, max(n_col, 1)),
                                 float(self.q.q_init))
        elif self.q_opt.shape[1] < n_col:
            pad = np.full((self.mdp.n_states, n_col - self.q_opt.shape[1]),
                          float(self.q.q_init))
            self.q_opt = np.hstack([self.q_opt, pad])
        return self._rr_oidx[name]

    def _rr_vbase(self, s):
        """V_base(s) = max_a Q[s, a] —— **只用基元动作**的价值。

        这是 advantage 的比较基线: "不用这个 option, 我靠基元动作能拿到多少"。
        它必须是**基元专属**的, 不能把 option 的价值混进来, 否则
        `Q_opt − V` 恒 <= 0(自己不可能超过包含自己的最大值)。
        """
        return float(np.max(self.q.Q[s]))

    def _rr_vall(self, s):
        """V_all(s) = max(基元, option) —— agent **实际行为**(含 option)的价值。

        ★ SMDP 的 bootstrap 必须用它, 不能用 `_rr_vbase`: option 之间是可以
          串联的(先拿钥匙再开门), 如果 option 从"基元专属"价值 bootstrap,
          option 链上的价值就传不下去 —— 每次传播只能走一层。
        """
        v = float(np.max(self.q.Q[s]))
        if self.q_opt is not None and self.q_opt.shape[1]:
            v = max(v, float(np.max(self.q_opt[s])))
        return v

    @staticmethod
    def _rr_advq_of(o):
        """advantage = Q_opt − Φ(s0) 的(遗忘加权)均值。正 = 比基线好。"""
        return o.get("advq_sum", 0.0) / max(1.0, o.get("advq_n", 1.0))

    @staticmethod
    def _rr_advq_se(o):
        """上面那个均值的**标准误**(遗忘加权)。没有它就不能说"显著更差"。"""
        n = max(1.0, o.get("advq_n", 0.0))
        m = o.get("advq_sum", 0.0) / n
        v = max(0.0, o.get("advq_sq", 0.0) / n - m * m)
        return float(np.sqrt(v / n))

    def _rr_settle(self, st, steps, reason, s0=None, s1=None,
                   r_opt=0.0, done=False):
        self.rr_stats["terms"][reason] = self.rr_stats["terms"].get(reason, 0) + 1
        self.rr_stats["durations"].append(int(steps))
        # ★ 归属到**这个 option**(此前 `st` 收下即弃 —— 库级台账不存在)
        o = self.rr_outcome.setdefault(getattr(st, "name", id(st)),
                                       {"n": 0, "reached": 0, "steps": 0,
                                        "n_eff": 0.0, "reached_eff": 0.0,
                                        "steps_eff": 0.0,
                                        "adv_sum": 0.0, "adv_steps": 0.0,
                                        "advq_sum": 0.0, "advq_n": 0.0})
        dec = self._rr_decay()
        hit = 1.0 if reason == "beta_at_feature" else 0.0
        # 终身计数(永不衰减, 供报告与判据门限用)
        o["n"] += 1
        o["steps"] += int(steps)
        o["reached"] += int(hit)
        # 有效计数(按 rr_forget_hl 指数遗忘; dec=1.0 时逐位等于终身计数)
        o["n_eff"] = dec * o["n_eff"] + 1.0
        o["reached_eff"] = dec * o["reached_eff"] + hit
        o["steps_eff"] = dec * o["steps_eff"] + int(steps)
        # 势函数差分(旧方案; 保留作对照)
        prog = self._rr_pot(s1) - self._rr_pot(s0)
        o["adv_sum"] = dec * o["adv_sum"] + prog
        o["adv_steps"] = dec * o["adv_steps"] + max(1, int(steps))
        # ★ SMDP 更新 + advantage 记账(新方案; 在线、无 oracle)
        if s0 is not None and s1 is not None:
            name = getattr(st, "name", None)
            if name is not None:
                j = self._rr_oindex(name)
                qo = self.q_opt
                assert qo is not None          # _rr_oindex 已建表
                g, k = float(self.q.gamma), max(1, int(steps))
                # ★ bootstrap 用**势函数** Φ: 它才是"agent 现在认为走到某状态值多少"
                #   的在线估计。`gvf` 时它是中性初始化独立学的状态价值,
                #   没有"未试过的动作把 max 顶在初值"这个病。
                boot = 0.0 if done else (g ** k) * self._rr_pot(s1)
                tgt = float(r_opt) + boot
                qo[s0, j] += self.q.lr * (tgt - qo[s0, j])
                # ★ 只在**势函数对该状态有证据**时记账 —— 否则比的是初值的形状
                if self._rr_has_base(s0):
                    advq = float(qo[s0, j]) - self._rr_pot(s0)
                    o["advq_sum"] = dec * o["advq_sum"] + advq
                    o["advq_sq"] = dec * o.get("advq_sq", 0.0) + advq * advq
                    o["advq_n"] = dec * o["advq_n"] + 1.0
                else:
                    o["no_base"] = o.get("no_base", 0) + 1

    # ── 自我校准: 检测 + 淘汰 ───────────────────────────────────────
    def rr_broken(self, names=None):
        """哪些 option 的**终止机制坏了**(而不是"暂时运气差")。

        判据(事前定义, 只看 option 自己的达成率, 不看主任务表现):
            启动 >= `rr_min_launches` 次 且 达成率 < `rr_calib_min_rate`

        ## 为什么不是 `reached == 0`

        最初的写法是"从没达成过"(绝对零)。这是错的 —— **连 L2 的真案都
        抓不到**: L2 是 57 启动 / 1 次 goal_reached, 达成率 0.018, 不是 0。
        绝对零判据会放过它。

        ## 0.5 这个线的依据(来自实测的两个簇, 不是拍脑袋)

            坏的:  L2 真案 0.018 (56/57 expired)  |  本轮制造器 0.250
            好的:  rr[key] 1.000 (确定性 MDP + 模型导出的 π_o 应当几乎必成)
                   rr-zeroV 0.595 (健康但偏早停的臂)

        两个簇之间有一大段空档, 0.5 落在空档里。**确定性 MDP 上模型导出的
        option 本该接近 1.0**, 明显低于 1.0 就意味着终止机制或 π_o 有问题。
        """
        out = set()
        for k, o in self.rr_outcome.items():
            if names is not None and k not in names:
                continue
            if o.get("n_eff", float(o.get("n", 0))) < self.rr_min_launches:
                continue
            bad_int = self._rr_rate(o) < self.rr_calib_min_rate
            bad_adv = self._rr_adv(o) < self.rr_calib_min_adv
            # advq 额外要求**基线证据够** + **显著更差**:
            #   ★ 判据必须是 "上置信界 < 0"(显著更差), 不是 "均值 < 0"。
            #   实测量到 Q_adv = −0.0002 —— 那是**零的噪声范围之内**,
            #   严格 `< 0` 会把每一个 option 都判坏(fallback 0.994)。
            #   证据不足时不许处决 —— 这正是"犯错是可以原谅的"的统计版本,
            #   而且它不需要任何手工 epsilon: 尺度由数据自己的方差给出。
            advq_mean = self._rr_advq_of(o)
            bad_q = (o.get("advq_n", 0.0) >= self.rr_min_launches
                     and advq_mean + self.rr_ucb_c * self._rr_advq_se(o) < 0.0)
            rule = self.rr_retire_rule
            if rule == "none":              # 诊断臂: 只报数, 不淘汰
                hit = False
            elif rule == "internal":
                hit = bad_int
            elif rule == "adv":
                hit = bad_adv
            elif rule == "advq":            # ★ 在线可用: 只需 agent 自己的 Q
                hit = bad_q
            elif rule == "both":
                hit = bad_int and bad_q
            elif rule == "all":
                hit = bad_int and bad_adv and bad_q
            else:
                hit = bad_int or bad_q
            if hit:
                out.add(k)
        return out

    def rr_calib_stats(self):
        """校准台账: 每个 option 的启动数 / **当前胜任度** / 效率 + 谁被判坏了。

        注意 `rate` 报的是**有效达成率**(按 `rr_forget_hl` 遗忘后的当前胜任度),
        `rate_life` 才是终身平均 —— 持久化的意义就在这两者的差别上。
        """
        per = {}
        for k, o in sorted(self.rr_outcome.items()):
            ne = o.get("n_eff", float(o.get("n", 0)))
            per[k] = {"n": o["n"], "reached": o["reached"],
                      "n_eff": round(ne, 2),
                      "rate": round(self._rr_rate(o), 4),
                      "rate_life": round(o["reached"] / o["n"], 4) if o["n"] else 0.0,
                      "eff": round(self._rr_eff(o), 4),
                      "adv": round(self._rr_adv(o), 4),
                      "advq": round(self._rr_advq_of(o), 4),
                      "mean_steps": round(o["steps"] / o["n"], 2) if o["n"] else 0.0}
        broke = self.rr_broken()
        return {"per_option": per, "broken": sorted(broke),
                "n_broken": len(broke)}

    def _rr_usage(self):
        """各 option 的**使用占比**(校准的直接观测量)。"""
        tot = sum(o["n"] for o in self.rr_outcome.values())
        if not tot:
            return {}
        return {k: round(o["n"] / tot, 4) for k, o in sorted(self.rr_outcome.items())}

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
