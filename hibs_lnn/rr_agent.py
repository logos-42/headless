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
import math

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
                 rr_lr_v=0.2, rr_starve_steps=None, rr_unk_p=0.5,
                 rr_three_state=False, rr_dyn="none", rr_dyn_c=1.0,
                 rr_evidence=False, rr_ev_n_min=10, rr_ev_z_bad=2.0,
                 rr_ev_stale_hl=200, **kw):
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
        # ══ 三态 epistemic state(好 / 坏 / 未知)══════════════════════════════
        #  ★ 为什么必须有: 实测发现, `advq == 0` 在两种完全不同的情况下出现 ——
        #      (a) 证据充分且真的没有显著差异  -> 好
        #      (b) **奖励流为空, Φ 学不到东西**(harm 故障下整臂全灭时
        #          `Q_adv ≡ 0.0000` 精确零, 因为账压根没记)  -> **未知**
        #    把 (b) 读成"不坏", 就是**自欺**。系统的最大缺口不是精度, 是它
        #    分不清这两种零。所以"未知"必须是一等状态, 不能被 0 顶上。
        #
        #  "奖励流为空"的在线判据: 连续多少步没有拿到任何正奖励。
        #  默认 3×horizon = "连着三个回合一分没拿" -> 势函数不可能在学东西。
        self.rr_starve_steps = (int(rr_starve_steps) if rr_starve_steps
                                else 3 * int(getattr(mdp, "horizon", 30)))
        self.rr_since_rew = 0        # 距上次拿到正奖励的步数
        self.rr_starve_ever = 0      # 曾经进入"奖励流为空"的次数(诊断)
        self.rr_unk_p = float(rr_unk_p)   # "未知"状态的探索概率(有限度探索)
        # ★ 三态开关。默认 **False** = 旧行为逐位不变(向后兼容 + 它本身是
        #   一条消融轴: "把未知提升为一等状态, 到底改变了什么?")。
        #   True 时启用三件事: 缓存为空 -> 退回基元层; 未知 -> 有限度探索;
        #   未知 -> 永不算坏。
        self.rr_three_state = bool(rr_three_state)
        # ── 阶段五: 证据库(evidence bank) ──────────────────────────────
        # ★ 默认关闭 ⇒ 旧行为**逐位不变**(与 rr_three_state 同一条纪律)。
        # ★ 语义: 证据**只降级(好->坏), 不升级** —— 与 A 类的反熟悉度约束同一条。
        #   理由: 预测误差低只说明"我预测得准", 不说明"这件事有价值";
        #   而**新颖度会拉高误差** ⇒ 允许升级就等于允许新颖度伪造"好"。
        self.rr_evidence = bool(rr_evidence)
        self.rr_ev_n_min = int(rr_ev_n_min)
        self.rr_ev_z_bad = float(rr_ev_z_bad)
        self.rr_ev_stale_hl = int(rr_ev_stale_hl)
        self.rr_bank = None
        self.rr_ev_step = 0
        self.rr_ev_last = None          # 最近一次的 support() 引用清单
        self.rr_ev_bad_n = 0            # 证据库投了几次 bad(诊断用)
        if self.rr_evidence:
            from hibs_lnn.evidence_bank import EvidenceBank
            self.rr_bank = EvidenceBank(
                dim=len(self.mdp.vec()), seed=seed,
                n_min=self.rr_ev_n_min, stale_hl=self.rr_ev_stale_hl,
                z_bad=self.rr_ev_z_bad)
        # ── A 类动力学证据(阶段四)─────────────────────────────────
        #   `rr_cumulant`(上面那个)是 **option 构造期**参数 —— 它是 OaK 的
        #   `(I_o, g_o, π_o, β_o)` 里的 `g_o`。A 类是**每步的动力学读数**,
        #   是另一回事, 所以单开一个通道。
        #
        #   ★ 方向约定: 三个 cumulant 都**统一成"压力"(越大越差)** ——
        #     A1 转移不确定性(归一化熵) / A2 转移预测误差(1−p(s2|s,a)) /
        #     A3 离开已知稳定动力学区域(1−熟悉度)。
        #     这样它们**只能当风险证据**, 不可能被读成"值"。
        self.rr_dyn = str(rr_dyn)
        if self.rr_dyn not in ("none", "N", "A1", "A2", "A3"):
            # "N" = 新颖度**控制量** —— 与 A1/A2/A3 走同一套管道,
            #   用来回答"A 类到底是动力学探测器, 还是只是新颖度探测器"。
            raise ValueError(f"unknown rr_dyn: {rr_dyn}")
        self.rr_dyn_c = float(rr_dyn_c)
        self._rr_T = {}          # (s,a) -> {s2: n}   在线转移计数(只用真实转移)
        self._rr_vis = {}        # s -> n             访问次数
        self._rr_dense = {}      # option -> [n, Σ, Σ²]  动力学证据台账
        self._rr_dseg = [0.0, 0.0, 0.0]   # 全段基线 [n, Σ, Σ²] —— "显著更差"的参照
        self._rr_dn = 0          # A 类喂了多少个转移
        self._rr_prev_state = {}     # option -> 上次判定(用来数退休/恢复次数)

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
        # 选择**尝试**总数 —— fallback 占比的分母必须是它, 不能是
        # `selections`(那只在真的选中时加一, 比值会 > 1, 实测 6.827)。
        self.rr_stats["sel_attempts"] = self.rr_stats.get("sel_attempts", 0) + 1
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
            # ★ 奖励流为空 -> **退回安全基元策略**。这不是放弃, 是承认"现在
            #   什么都判不了": Φ 学不到东西, 任何判断都是自欺。等主任务拿到
            #   奖励、Φ 重新有内容, 再回来判。
            if self.rr_three_state and self.rr_starved():
                self.rr_stats["starved_sel"] = self.rr_stats.get("starved_sel", 0) + 1
                return None
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
            # ★ 三态驱动的选择(leo 契约: "未知应触发有限度探索"):
            #   未知 -> 以 `rr_unk_p` 的概率**优先挑它** —— 不确定的 option
            #           得先攒到证据, 才可能从"未知"变成"好/坏"。没有这条,
            #           它永远停在未知(不被选 -> 没证据 -> 不被选)。
            #   好   -> 正常 UCB 参与选择。
            #   坏   -> 只在原谅概率下被重新检验。
            unk = [st for st in live
                   if self.rr_epistemic(self.rr_outcome.get(
                       getattr(st, "name", None), {})) == "unknown"]
            if unk and self.rr_three_state and self.rng.rand() < self.rr_unk_p:
                self.rr_stats["unk_probe"] = self.rr_stats.get("unk_probe", 0) + 1
                return unk[int(self.rng.randint(len(unk)))]
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
            # ★ A 类动力学证据: **先读数(用更新前的表)再更新** —— 这是
            #   "先预测、再观测"的正确在线语义。**失败时这行照样执行**:
            #   到不了终点也有真实转移, 这正是 A 类作为候选的全部理由。
            if self.rr_dyn != "none":
                _dv = self._rr_dval(s, a, s2)
                self._rr_dstep(s, a, s2)
                self._rr_dacc(active.name if active is not None
                              else "__primitive__", _dv)
            n += 1
            # ★ 奖励流监控: "连续多少步一分没拿"。Φ 从奖励流学, 奖励流为空
            #   时 Φ 恒为初值 —— 此时任何"不坏"的结论都是自欺。
            self.rr_since_rew = 0 if float(r) > 0 else (self.rr_since_rew + 1)
            # ── 阶段五: 喂证据库 ────────────────────────────────────────
            # ★ `phi=v0, phi_next=self.mdp.vec()` —— **不能两个都传 `self.mdp.vec()`**。
            #   `self.mdp` 在 `step()` 里已经前进, 两个都传当前 vec 会让 `v_next == v`,
            #   TD 的 `gamma*v_next` 项与 `v` 抵消 ⇒ 学的是"瞬时误差"不是值。
            #   (`hibs_lnn/oak_proposer.py:338` 的 `observe_gvf(phi, cums, phi)` 正是
            #    这个形态; 本模块刻意避开, 见 docs/stage5_evidence_bank_spec.md §0。)
            # 四个 cumulant = 这一步**四个不同的可观测事实**:
            #   prediction → "有 option 正在执行"   reward → 即时奖励
            #   transition → "状态没变(自环)"        regime → "本回合结束"
            if self.rr_bank is not None:
                _cums = (1.0 if active is not None else 0.0,
                         float(r),
                         1.0 if s2 == s else 0.0,
                         1.0 if done else 0.0)
                self.rr_ev_step += 1
                self.rr_bank.observe(v0, _cums, self.mdp.vec(),
                                     step=self.rr_ev_step,
                                     regime=self.rr_regime_idx)
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
        self._rr_sync_states()          # 三态判定 + 退休/恢复计数(每回合一次)
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

    # ── 三态 epistemic state ────────────────────────────────────────
    # ── A 类动力学 cumulant(阶段四)────────────────────────────────
    def _rr_dval(self, s, a, s2):
        """A 类 cumulant 的**行前读数**(用更新前的表 —— 先预测再观测)。

        三条共同契约(leo 的四条要求里最硬的两条):
          · 每一步都能在线获得 —— 只看刚发生的真实转移;
          · **不用目标坐标、不用真实模型、不用未来标签** —— 只有
            `(s, a, s2)` 三元组的计数, agent 自己走出来的;
          · 统一方向 = "压力", 越大越差, ⇒ 天然不可能被读成"值"。
        """
        if self.rr_dyn == "none":
            return 0.0
        d = self._rr_T.get((int(s), int(a)))
        n = sum(d.values()) if d else 0
        if self.rr_dyn == "N":
            # ★ **控制量: 新颖度**。`1/(1+n(s,a))`, 用**更新前**的计数。
            #   它必须与 A1/A2/A3 **走同一套管道**(同一个 _rr_dval/_rr_dacc/
            #   基线与判据), 否则"A 与 N 有没有分开"比的是实现差异而不是语义。
            #   方向同样统一成"压力": 越新 -> 越大。
            return float(1.0 / (1.0 + n))
        if self.rr_dyn == "A1":
            # ★ A1 = **转移预测误差**(按 leo 的 A′ 规格)。
            #   语义: 用**更新前**的模型预测这一步会落在哪, 再看实际落在哪。
            #   ⇒ "意外度" = 1 − p̂_prev(实际后继 | s,a)。
            #   ★ 这是**可预测性**, 不是任务价值 —— 一个稳定把你带向
            #     错误方向的动作照样很容易预测(leo 点名的那条)。
            #   ★ 与 A2 的分工: A1 对**分布翻转**敏感(A2 不敏感),
            #     A2 对**支撑集扩张**敏感(A1 也敏感) —— 两者在翻转环境上
            #     **必然分歧**, 这正是检验"它们是不是同一个量"的机会。
            #
            #   注意: 本分支与 A2 早先**写反了**(旧 A1 = 熵, 旧 A2 = 1−p̂)。
            #   在 A′ 里按 leo 规格对齐: A1 = 预测误差, A2 = 不确定性。
            if n < 2:
                return 1.0          # 无先验 ⇒ 没有"预测"可供做错, 报未知
            return float(1.0 - d.get(int(s2), 0) / n)
        if self.rr_dyn == "A2":
            # ★ A2 = **转移不确定性** = 观测到的后继分布归一化熵。
            #   ★ 关键区别: 熵对 **{0.9,0.1} 与 {0.1,0.9} 给出同一个数**,
            #     所以分布翻转时 A2 **不跳** —— 这是 A1 ≠ A2 的结构性证据。
            #   ★ 但 A2 对"支撑集扩张"(0.9/0.1 -> 0.5/0.5) 敏感。
            if n < 2:
                return 1.0          # 见过的样本不足 ⇒ 报最大不确定
            p = np.array(list(d.values()), float)
            p /= p.sum()
            return float(-(p * np.log(p + 1e-12)).sum()
                         / np.log(max(2, getattr(self.mdp, "n_states", 32))))
        # A3 离开已知稳定动力学区域: 落点的**出发边平均熵**(归一化)。
        #   ★ 方向必须与 A1/A2 一致 = "压力"(越大越差)。第一版写成了
        #     `1 - 熵`, 返回的是**熟悉度** —— 与通道约定相反, 于是读数恒
        #     1.000(确定性 MDP 里熵恒 0)、永不触发。这是"通道方向写反"
        #     的典型样子: 指标看起来在动, 但基线与判据读的是相反的量。
        ns = int(s2)
        if self._rr_vis.get(ns, 0) == 0:
            return 1.0                                  # 全新状态 -> 最大压力
        ents = []
        # 用**观测到的**出边动作, 不硬编码 (0,1) —— 第一版漏掉了 GRAB/OPEN,
        # 而 KeyDoor 的动作是 LEFT/RIGHT/GRAB/OPEN 四个。
        acts = {a2 for (ns2, a2) in self._rr_T if ns2 == ns}
        for a2 in (sorted(acts) if acts else (0, 1)):
            dd = self._rr_T.get((ns, int(a2)))
            nn = sum(dd.values()) if dd else 0
            if nn >= 2:
                pp = np.array(list(dd.values()), float)
                pp /= pp.sum()
                ents.append(float(-(pp * np.log(pp + 1e-12)).sum()))
        if not ents:
            return 1.0                                  # 出边一条都没摸过
        return float(min(1.0, np.mean(ents)
                         / np.log(max(2, getattr(self.mdp, "n_states", 32)))))

    def _rr_dstep(self, s, a, s2):
        """把这一步的真实转移记进在线转移表 —— **失败时这行照样执行**。"""
        k = (int(s), int(a))
        d = self._rr_T.get(k)
        if d is None:
            d = self._rr_T[k] = {}
        d[int(s2)] = d.get(int(s2), 0) + 1
        self._rr_vis[int(s2)] = self._rr_vis.get(int(s2), 0) + 1
        self._rr_dn += 1

    def _rr_dacc(self, name, d):
        """把读数记到 **具体 option** 的台账 + 全段基线。"""
        L = self._rr_dense.get(name)
        if L is None:
            L = self._rr_dense[name] = [0.0, 0.0, 0.0]
        L[0] += 1.0
        L[1] += d
        L[2] += d * d
        B = self._rr_dseg
        B[0] += 1.0
        B[1] += d
        B[2] += d * d

    def _rr_dyn_bad(self, name):
        """A 类风险判据: 该 option 的动力学压力是否**显著高于本段基线**。

        尺度**完全由数据自己的方差给出**(两样本 SE), 不引入手工 ε ——
        与 leo 在 inert 上要求的"'接近零'必须使用健康段自身的方差定义"同一条。
        """
        if self.rr_dyn == "none":
            return False
        L = self._rr_dense.get(name)
        if L is None or L[0] < self.rr_min_launches:
            return False
        Bn, Bs, Bs2 = self._rr_dseg
        if Bn < 20 or Bn <= L[0]:
            return False
        m = L[1] / L[0]
        v = max(0.0, L[2] / L[0] - m * m)
        bm = Bs / Bn
        bv = max(0.0, Bs2 / Bn - bm * bm)
        se = math.sqrt(v / L[0] + bv / Bn)
        return (m - bm) > self.rr_dyn_c * se

    def rr_dyn_stats(self):
        """A 类通道的报告: 证据量 / 均值 / 是否触发风险。"""
        Bn, Bs, Bs2 = self._rr_dseg
        bm = Bs / Bn if Bn else 0.0
        per = {}
        for k in sorted(self._rr_dense):
            L = self._rr_dense[k]
            m = L[1] / L[0] if L[0] else 0.0
            per[k] = {"n": int(L[0]), "mean": round(m, 4),
                      "risk": bool(self._rr_dyn_bad(k))}
        return {"kind": self.rr_dyn, "transitions": int(self._rr_dn),
                "base_mean": round(bm, 4), "base_n": int(Bn), "per_option": per}

    def rr_starved(self):
        """奖励流是否为空 —— 势函数不可能在学东西。

        ★ 这是本轮最重要的那个门。实测: harm 故障下整臂全灭时
        `Q_adv ≡ 0.0000` **精确为零** —— 不是"没有显著差异", 是"根本没记账"
        (到不了终点 -> 无奖励 -> Φ 恒为 0 -> 证据门 `_rr_has_base` 挡掉一切)。
        不加这个门, 系统会把"我不知道"读成"没问题" —— **自欺**。
        """
        return self.rr_since_rew >= self.rr_starve_steps

    def rr_epistemic(self, o):
        """**三态判定**: `"good"` / `"bad"` / `"unknown"`。

        与旧 `rr_broken` 的关键区别: **"未知"不是"好"**。
        旧代码 `advq = 0 -> 不 < 0 -> 不淘汰 -> 当作健康`, 而在奖励流为空时
        `advq` 恰好是精确的 0 —— 于是"没有证据"被系统性地读成"没问题"。

        判据(与 leo 定的契约一致):
          未知  —— 奖励流为空 / 势函数未离开初值 / 证据量不足(< min_launches)
          坏    —— 证据充分, 且**上置信界 < 0**(显著更差)
          好    —— 证据充分, 且没有显著负向信号
        """
        if self.rr_starved():
            return "unknown"                      # Φ 学不到 -> 无有效预测证据
        n = o.get("advq_n", 0.0)
        if n < self.rr_min_launches:
            return "unknown"                      # 证据不足
        m = self._rr_advq_of(o)
        if m + self.rr_ucb_c * self._rr_advq_se(o) < 0.0:
            return "bad"
        # ★★ **反熟悉度保证**: A 类动力学证据**只能降级(好 -> 坏)**,
        #    **永远不能升级(未知 -> 好)**。也就是说"可预测"买不到"好" ——
        #    否则一个稳定把你带向错误方向的 option 会因为好预测而被判好。
        #    这条不是调出来的, 是设计约束: A 类是**风险通道**, 不是值信号。
        #    (它到底能不能当值信号, 由"稳定但错误"否证实验来判。)
        nm = o.get("name")
        if nm is not None and self._rr_dyn_bad(nm):
            return "bad"
        # ── 阶段五: 证据库(★ **只降级, 不升级**)──────────────────────
        #   与 A 类同一条不对称约束: 证据可以投 `bad`, 但**永远不能**把状态升成
        #   `good`。理由: 预测误差低只说明"我预测得准", 不说明"这件事有价值";
        #   而**新颖度会拉高误差** ⇒ 允许升级就等于允许新颖度伪造"好"。
        #   ⇒ 证据缺席不改变判定, 证据示警才降级。
        if self.rr_bank is not None:
            self.rr_ev_last = self.rr_bank.support(step=self.rr_ev_step,
                                                   regime=self.rr_regime_idx)
            if self.rr_ev_last.get("verdict") == "bad":
                self.rr_ev_bad_n += 1
                return "bad"
        return "good"

    def rr_ev_support(self):
        """★ 最近一次证据判定背后的**引用清单** —— 哪条支持 bad / 哪条缺席。

        这是"证据可追溯"的直接接口: 调用方拿到的不是分数, 是**来源**。
        """
        if self.rr_bank is None:
            return {"enabled": False}
        s = self.rr_bank.support(step=self.rr_ev_step, regime=self.rr_regime_idx)
        s["enabled"] = True
        s["n_bad_ever"] = int(self.rr_ev_bad_n)
        return s

    def rr_complementarity(self, label="reward"):
        """★ 这组 GVF 到底有没有产生**互补证据**。

        `label`: 用哪个 cumulant 的二值化当"变化/未变化"标签。
        默认 `"reward"` ⇒ 问的是「这组证据在**奖励到达**这件事上互不互补」。

        ⚠ 口径限制(必须写明): 标签本身就是四个 cumulant 之一,
        所以对 `reward` 那条 GVF 天然有利。读 `complementarity` 时要同时看
        `auc_i`(**单条**的 AUC)与 `corr_err`(冗余度), 不能只看差值。

        **允许结论是"冗余"** —— 那就照实报冗余。
        """
        if self.rr_bank is None:
            return {"enabled": False}
        from hibs_lnn.evidence_bank import complementarity
        h = self.rr_bank.z_history()
        if label not in h["cumulant"]:
            return {"enabled": True, "error": f"unknown label {label}",
                    "labels": list(h["cumulant"])}
        lab = (h["cumulant"][label] > 0).astype(int)
        if lab.min() == lab.max():
            return {"enabled": True, "error": "标签全为同一值, 无法算 AUC",
                    "label": label, "n": int(lab.size)}
        r = complementarity(h["z"], lab, z_bad=self.rr_ev_z_bad)
        r.update({"enabled": True, "label": label, "n": int(lab.size),
                  "pos_rate": float(lab.mean())})
        return r

    def rr_epistemic_all(self, names=None):
        return {k: self.rr_epistemic(o) for k, o in self.rr_outcome.items()
                if names is None or k in names}

    def _rr_sync_states(self):
        """把三态判定同步进台账, 并**计数退休/恢复**。

        为什么必须分开记: leo 的契约里 `当前状态` 用于**决策**,
        `历史状态` 用于**审计** —— 一个 option 被判坏过几次、又恢复过几次,
        是"持续学习是否真的在发生"的直接证据, 不能被当前状态覆盖掉。
        """
        for k, o in self.rr_outcome.items():
            st = self.rr_epistemic(o)
            old = self._rr_prev_state.get(k)
            if old is not None and st != old:
                if st == "bad":
                    o["retired_n"] = o.get("retired_n", 0) + 1
                elif old == "bad":
                    o["recovered_n"] = o.get("recovered_n", 0) + 1
            self._rr_prev_state[k] = st
            o["state"] = st
        if self.rr_starved():
            self.rr_starve_ever += 1

    def _rr_settle(self, st, steps, reason, s0=None, s1=None,
                   r_opt=0.0, done=False):
        self.rr_stats["terms"][reason] = self.rr_stats["terms"].get(reason, 0) + 1
        self.rr_stats["durations"].append(int(steps))
        # ★ 归属到**这个 option**(此前 `st` 收下即弃 —— 库级台账不存在)
        o = self.rr_outcome.setdefault(getattr(st, "name", id(st)),
                                       {"n": 0, "reached": 0, "steps": 0,
                                        "name": getattr(st, "name", None),
                                        "n_eff": 0.0, "reached_eff": 0.0,
                                        "steps_eff": 0.0,
                                        "adv_sum": 0.0, "adv_steps": 0.0,
                                        "advq_sum": 0.0, "advq_n": 0.0,
                                        "advq_sq": 0.0})
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
            # ★ 三态: "未知"**永远不算坏** —— 证据不足 / 奖励流为空都不处决。
            #   这是"退休必须保守"的原则: 宁可留着不确定的, 不可错杀。
            if self.rr_three_state and self.rr_epistemic(o) == "unknown":
                bad_q = False
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
            m = self._rr_advq_of(o)
            se = self._rr_advq_se(o)
            per[k] = {
                # ── 当前判定(用于决策)──────────────────────────────
                "state": self.rr_epistemic(o),
                # ── 证据量 ──────────────────────────────────────────
                "n": o["n"],
                "n_eff": round(ne, 2),
                "advq_n": round(o.get("advq_n", 0.0), 2),
                "no_base": o.get("no_base", 0),   # 有启动但没进比较的次数
                # ── 近期状态(遗忘加权)─────────────────────────────
                "rate": round(self._rr_rate(o), 4),
                "advq": round(m, 4),
                "advq_se": round(se, 4),
                "advq_lo": round(m - self.rr_ucb_c * se, 4),   # 下置信界
                "advq_hi": round(m + self.rr_ucb_c * se, 4),   # 上置信界
                "eff": round(self._rr_eff(o), 4),
                "adv": round(self._rr_adv(o), 4),
                # ── 历史状态(用于审计; 与当前状态分开)────────────
                "reached": o["reached"],
                "rate_life": round(o["reached"] / o["n"], 4) if o["n"] else 0.0,
                "retired_n": o.get("retired_n", 0),      # 被判坏过几次
                "recovered_n": o.get("recovered_n", 0),  # 从坏恢复过几次
                "mean_steps": round(o["steps"] / o["n"], 2) if o["n"] else 0.0,
            }
        broke = self.rr_broken()
        states = [v["state"] for v in per.values()]
        return {
            "per_option": per, "broken": sorted(broke), "n_broken": len(broke),
            # ★ 三态汇总 + 自欺门的状态 —— 报告层必须能一眼看出
            #   "这次是真的没问题" 还是 "这次根本没数据"。
            "states": {"good": states.count("good"),
                       "bad": states.count("bad"),
                       "unknown": states.count("unknown")},
            "starved": bool(self.rr_starved()),
            # ★ 阶段五: 证据库 —— 报的是**引用清单**不是分数
            "evidence": (self.rr_ev_support() if self.rr_evidence else {"enabled": False}),
            "since_rew": int(self.rr_since_rew),
            "starve_steps": int(self.rr_starve_steps),
            "starve_ever": int(self.rr_starve_ever),
            "sel_starved": self.rr_stats.get("starved_sel", 0),
            "sel_unk_probe": self.rr_stats.get("unk_probe", 0),
            "sel_forgiven": self.rr_stats.get("forgiven", 0),
        }

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
