"""oak_proposer.py — OaK 提议器: 把 InternalKnowledge + Options 接进主回路。

## 状态/动作的定义 (在这个问题里)

    state  = 各细区间的**能力画像** (维度 = n_fine)   "我现在会什么"
    action = 选哪个细区间训练 (0..n_fine-1)
    next   = 训练后的新能力画像
    reward = Δ(any-time 准确率)

这就是真实的持续学习闭环: **动作是"去练哪块", 环境是"练完能力怎么变"。**
`T(s,a) -> s'` 因此是一个**真正可学、数据里天然存在**的转移 —— 不是硬凑的。

## 结构

    OAKProposer
      ├── base    : RLProposer        (Level 3 价值/策略知识, 复用已验证机制)
      └── know    : InternalKnowledge (Level 1/2/4 + 三类元知识)
            ├── predictions  GVFBank
            ├── dynamics     TransitionEnsemble
            ├── options      OptionManager
            ├── uncertainty  UncertaintyGate
            ├── coverage     Coverage
            └── plasticity   Plasticity

## act() 的决策顺序

    ① 有 option 在跑且未终止  -> 沿 option 的动作序列走
    ② 否则取 base 的候选      -> 过**门控**; 被拒则退回已覆盖动作
    ③ 记录 (state, action)    -> 供 dynamics/option 发现用

## E9 vs E10 就是一个 flag

    use_options=False  -> E9  (+coverage/uncertainty 门控)
    use_options=True   -> E10 (+Options)
    两者共用同一 base 与同一门控, 差异**只**在于是否使用时间抽象。
"""
from __future__ import annotations

import numpy as np

from hibs_lnn.knowledge import InternalKnowledge


class OAKProposer:
    def __init__(self, fine_desc, n_fine, k=3, seed=0, tau=0.5,
                 mu=0.05, alpha0=0.2, explore_w=0.5, algo="autostep",
                 n_know=0, use_options=True, use_gate=True,
                 n_models=5, refresh_every=4, min_transitions=40,
                 n_regions=4, max_opt_len=3, opt_frac=1.0,
                 opt_mode="fixed", term_eps=0.05, stall_tol=0.0,
                 dyn_model="linear", n_regions_opt=16, use_subgoals=0):
        from hibs_lnn.rl_proposer import RLProposer
        self.n_fine = int(n_fine)
        self.n = self.n_fine
        self.k = int(k)
        self.seed = seed
        self.use_options = bool(use_options)
        self.use_gate = bool(use_gate)
        self.refresh_every = int(refresh_every)
        self.min_transitions = int(min_transitions)
        self.n_regions = int(n_regions)
        self.max_opt_len = int(max_opt_len)
        # ★ 诊断用: 只在一部分轮次允许启用 option。
        #   若"Options 有害"确实来自 option 执行本身 (提交一段固定动作序列 ->
        #   减少自适应性 / 降低探索多样性), 那么伤害应当**随 opt_frac 单调放大**。
        #   若伤害与用量无关, 说明另有来源 (如 discovery 本身在扰动状态)。
        self.opt_frac = float(opt_frac)
        self._opt_round = 0
        # ★ opt_mode: Option 的**执行方式** (用户 2026-09-14 指导的 K1 矩阵)
        #   "fixed"              O1  固定动作序列 = open-loop (旧行为, D1 的病灶)
        #   "goal"               O2  目标条件 + **每步重算动作** (闭环)
        #   "goal_term"          O3  O2 + 自适应终止 β_o(s)
        #   "goal_term_override" O4  O3 + 不确定性抢占 (Option 从属于当前证据)
        self.opt_mode = str(opt_mode)
        self.term_eps = float(term_eps)
        self.stall_tol = float(stall_tol)
        self._opt_stall_d = None
        self.override_count = 0
        self.term_reasons = {}
        self.replan_steps = 0
        # ★ 世界模型 (实测: 线性 ridge 表达不了结构化动力学 —— 在 KeyDoor 这类
        #   离散/条件性动力学上四个动作预测到**同一个**下一状态, 转移图退化成
        #   几乎无边, option 发现为 0。计数式表格模型修掉了这个。)
        self.dyn_model_kind = str(dyn_model)
        self.n_regions_opt = int(n_regions_opt)
        self.use_subgoals = int(use_subgoals)

        # Level 3: 复用已验证的 RL 提议器
        self.base = RLProposer(fine_desc, tau=tau, k=k, mu=mu, alpha0=alpha0,
                               explore_w=explore_w, algo=algo, n_know=n_know,
                               seed=seed)
        # ★ 内部知识**惰性创建**: 状态维度在构造时未知。
        #   主回路的状态是 `row` = 每**粗域**准确率 (长度 = --domains),
        #   而动作数是细区间数 n_fine —— 两个维度**不同**。
        #   (实测: 按 n_fine 建 GVF 会在 observe 时撞
        #    `size 3 is different from 6`)。
        self._know_cfg = dict(n_models=n_models, seed=seed, alpha0=alpha0, mu=mu)
        self.know = None
        self.dim_state = None

        # 转移日志: (state_t, action, state_{t+1})
        self.trans = []
        self._prev_state = None
        self._prev_pick = None
        self._last_acc = {}
        self._opt_queue = []          # 正在执行的 option 剩余动作
        self._opt_active = None
        self.refresh_count = 0
        self.untrusted_picks = 0      # 提议落在不可信区的次数 (仅诊断)
        self.fallback_count = 0       # 保留字段 (旧: 门控重定向次数, 已废弃)
        self.option_steps = 0         # 由 option 决定的步数
        self.option_starts = 0        # option 被启动的次数

    # ── 惰性初始化 ──────────────────────────────────────────────────
    def _ensure_know(self, dim_state):
        if self.know is not None:
            return self.know
        self.dim_state = int(dim_state)
        self.know = InternalKnowledge(self.n_fine, self.dim_state, **self._know_cfg)
        if self.dyn_model_kind == "tabular":
            from hibs_lnn.uncertainty_gate import TabularTransition
            self.know.dynamics_model = TabularTransition()
        self.know.register_value_fn(
            lambda s, a: float(self._last_acc.get(int(a), np.nan)),
            policy="rl-base")
        return self.know

    # ── 记录环境反馈 ────────────────────────────────────────────────
    def set_state(self, state):
        """每轮训练后由主回路喂入**新**能力画像。"""
        st = np.asarray(state, dtype=float)
        self._ensure_know(st.shape[0])
        if self._prev_state is not None and self._prev_pick is not None:
            for a in (self._prev_pick if isinstance(self._prev_pick, (list, tuple))
                      else [self._prev_pick]):
                self.trans.append((self._prev_state.copy(), int(a), st.copy()))
                self.know.observe_action(int(a))
        self._prev_state = st
        self._last_acc = {i: float(v) for i, v in enumerate(st)
                          if np.isfinite(v)}

    def set_pick(self, pick):
        """记录本轮真正执行的动作。"""
        self._prev_pick = list(pick) if isinstance(pick, (list, tuple)) else [pick]

    # ── 动作选择 ───────────────────────────────────────────────────
    def act(self):
        # ① option 在执行中
        if self.use_options and self._opt_active is not None and self.opt_mode != "fixed":
            o = self._opt_active
            s_now = self._prev_state
            if s_now is None:
                self._opt_active = None
            else:
                _, unc_a = self.know.predict(s_now, int(o.actions[0]))
                tau = self.know.uncertainty.tau_U
                if self.opt_mode == "goal_term_override":
                    if (not np.isfinite(unc_a)) or (tau and unc_a >= tau):
                        self.override_count += 1
                        self.term_reasons["override"] = self.term_reasons.get("override", 0) + 1
                        self._opt_active = None
                        return list(self.base.act())
                if self.opt_mode in ("goal_term", "goal_term_override"):
                    done, why = o.terminated(
                        s_now, unc=unc_a, eps=self.term_eps,
                        tau_U=(tau if self.opt_mode == "goal_term_override" else None),
                        stall=self.stall_tol)
                    if done:
                        self.term_reasons[why] = self.term_reasons.get(why, 0) + 1
                        self._opt_active = None
                        return list(self.base.act())
                a = self._goal_action(o)
                if a is None:
                    self._opt_active = None
                else:
                    self.replan_steps += 1
                    self.option_steps += 1
                    return [int(a)]
        if self.use_options and self._opt_queue and self.opt_mode == "fixed":
            a = int(self._opt_queue.pop(0))
            self.option_steps += 1
            if not self._opt_queue:
                self._opt_active = None
            return [a]

        pick = list(self.base.act())

        # ② ★ 门控**不用于 action selection**。
        #   早先版本在这里把零覆盖动作重定向到 `argmin(visits)` 的已覆盖动作,
        #   结果是灾难性的: 一旦某个动作被覆盖, 门控就把**所有**提议都指向它
        #   (实测 `per_action = [0,9,0,0,0,0]`, 9 轮只练了 1 个区间, 探索崩塌)。
        #
        #   语义纠正 (Q8 原文): "目标策略必须在现有经验中得到足够的信息支持" ——
        #   这约束的是**模型驱动的规划决策** (凭预测去选), **不是**阻止 base
        #   policy 去探索。往零覆盖区探索恰恰是想要的: 覆盖度只能这样长起来。
        #   所以门控只作用在**模型被咨询**的地方 (option 选择 / rollout),
        #   那两处已由 OptionManager.select(exclude_unc=True) 与
        #   ens.rollout(gate=...) 各自保证。
        if self.use_gate and self.know is not None:
            # 只**记录**当前提议是否落在不可信区 (诊断用, 不做干预)
            for a in pick:
                if int(a) < self.n_fine:
                    _, unc = self.know.predict(self._prev_state
                                               if self._prev_state is not None
                                               else np.zeros(self.dim_state or 1), int(a))
                    tau = self.know.uncertainty.tau_U
                    if (not np.isfinite(unc)) or (tau and unc >= tau):
                        self.untrusted_picks += 1

        # ③ 起一个新 option (受 opt_frac 限制)
        self._opt_round += 1
        _allow_opt = (self.opt_frac >= 1.0
                      or (self.opt_frac > 0 and
                          (self._opt_round % max(1, int(round(1.0 / self.opt_frac)))) == 0))
        if (_allow_opt and self.use_options and self.know is not None
                and self.know.options is not None and self._prev_state is not None):
            o = self.know.options.select(self._prev_state, exclude_unc=True)
            if o is not None and len(getattr(o, "actions", [])) > 1:
                self._opt_active = o
                # ★ 返回 option 的**第一个**动作。早先实现假设"起点的那个动作
                #   已经执行过了", 于是只排队 actions[1:] —— 结果 option 的语义
                #   只被用了一半, option_steps 实测只有 1~2 步。
                self._opt_queue = [int(x) for x in o.actions[1:]]
                self.option_starts += 1
                if self.opt_mode == "fixed":
                    self._opt_queue = [int(x) for x in o.actions[1:]]
                    return [int(o.actions[0])]
                a = self._goal_action(o)          # 闭环: 立刻按当前状态算
                if a is None:
                    self._opt_active = None
                    return pick
                self.replan_steps += 1
                self.option_steps += 1
                return [int(a)]
        return pick

    def _goal_action(self, opt):
        """π_o(s_t) —— **按当前状态重算**动作, 而不是回放固定序列。

            a = argmax_a [ ‖g−s‖ − ‖g−T(s,a)‖ ]
        即"哪一步动作让我朝子目标前进最多"。

        ★ 用户指出的核心修正: **抽象的是目标, 不是动作**。
          固定序列执行 = "我发现了一段过去有效的脚本, 现在重放它",
          状态变化不再触发动作重算 -> 适应性下降 (D1 实测 r=-0.80)。
        """
        s_now = self._prev_state
        g = np.asarray(opt.goal_center, dtype=float)
        d_now = float(np.linalg.norm(np.asarray(s_now, dtype=float) - g))
        best_a, best_gain = None, -np.inf
        for a in range(self.n_fine):
            if self.use_gate:
                ok, _ = self.know.uncertainty.allow(s_now, a, self.know.coverage.n)
                if not ok:
                    continue
            s_next, unc = self.know.predict(s_now, a)
            if not np.isfinite(unc):
                continue
            gain = d_now - float(np.linalg.norm(np.asarray(s_next, dtype=float) - g))
            if gain > best_gain:
                best_a, best_gain = a, gain
        return best_a

    # ── 奖励 → 参数更新 (+ 知识维护) ────────────────────────────────
    def observe(self, pick, acc, any_time):
        self.base.observe(pick, acc=acc, any_time=any_time)

    def update(self, pick):
        self.base.update(pick)
        # 预测知识: cumulant = 各区间能力本身 (GVF 学"能力会怎么走")
        if self.know is not None and self._prev_state is not None:
            phi = self._prev_state
            cums = [float(np.mean(phi)),
                    float(np.max(phi) - np.min(phi)),
                    float(np.std(phi)),
                    float(np.mean(np.diff(np.sort(phi)[::-1][:3])) if self.n_fine >= 3 else 0.0)]
            self.know.observe_gvf(phi, cums, phi)
        # 周期性重建 world model + 重新发现 option
        if (self.know is not None and len(self.trans) >= self.min_transitions
                and self.refresh_count < len(self.trans) // max(1, self.refresh_every)):
            self.refresh()
            self.refresh_count += 1

    def refresh(self):
        """把转移日志灌进 dynamics ensemble -> 标定门控 -> 重新发现 options。"""
        if self.know is None:
            return
        try:
            X = np.array([np.append(s, a) for s, a, _ in self.trans])
            Y = np.array([sp for _, _, sp in self.trans])
            self.know.fit_dynamics(X, Y)
            from hibs_lnn.option_manager import OptionManager
            om = OptionManager(self.know.dynamics_model, self.know.uncertainty,
                               max_len=self.max_opt_len, seed=self.seed)
            states = np.array([s for s, _, _ in self.trans])
            acts = np.array([a for _, a, _ in self.trans])
            # ★ 表格模型下按**真实状态**建图 (不是聚类中心)
            n_uniq = len(np.unique(np.round(states, 6), axis=0))
            nreg = (min(self.n_regions_opt, max(4, n_uniq))
                    if self.dyn_model_kind == "tabular"
                    else min(self.n_regions, max(2, len(states) // 10)))
            if self.use_subgoals:
                om.discover_subgoals(states, acts, n_regions=nreg)
            else:
                om.discover(states, acts, n_regions=nreg)
            self.know.register_options(om)
        except Exception as e:      # 知识重建失败不该拖垮主实验
            print("!! knowledge refresh 失败:", e, flush=True)

    # ── 诊断 ───────────────────────────────────────────────────────
    def stats(self):
        s = self.base.proposer_stats() if hasattr(self.base, "proposer_stats") else {}
        # ★ 真步长统计: 取自 base 的最近一条 trace (RLProposer 在那里记录 α)
        _tr = getattr(self.base, "trace", None) or []
        if _tr:
            s["alpha_mean"] = _tr[-1].get("alpha_mean")
            s["alpha_std"] = _tr[-1].get("alpha_std")
            s["h_norm"] = _tr[-1].get("h_norm")
        if self.know is None:
            s.update({"n_trans": len(self.trans), "refreshes": 0, "n_options": 0,
                      "option_steps": self.option_steps,
            "opt_frac": self.opt_frac,
            "opt_mode": self.opt_mode,
            "dyn_model": self.dyn_model_kind,
            "use_subgoals": self.use_subgoals,
            "replan_steps": self.replan_steps,
            "override_count": self.override_count,
            "term_reasons": dict(self.term_reasons),
            "option_starts": self.option_starts, "untrusted_picks": self.untrusted_picks,
                      "coverage_total": 0, "tau_U": None, "gvf_steps": [], "alpha_std": 0.0})
            return s
        s.update({
            "n_trans": len(self.trans),
            "refreshes": self.refresh_count,
            "n_options": len(self.know.skills()),
            "option_steps": self.option_steps,
            "option_starts": self.option_starts,
            "opt_frac": self.opt_frac,
            "opt_mode": self.opt_mode,
            "dyn_model": self.dyn_model_kind,
            "use_subgoals": self.use_subgoals,
            "replan_steps": self.replan_steps,
            "override_count": self.override_count,
            "term_reasons": dict(self.term_reasons),
            "untrusted_picks": self.untrusted_picks,
            "coverage_total": int(self.know.coverage.n_total),
            "tau_U": self.know.uncertainty.tau_U,
            "gvf_steps": [g.steps for g in self.know.predictions.gvfs],
            # ★ 不要放在 "alpha_std" 这个键上 —— 它会**覆盖** base.proposer_stats()
            #   里真正的步长统计。早先版本就是这么写的, 结果 JSON 里几个不同算法
            #   (idbd / idbd-raw / cidbd) 的 alpha_std 全是同一个**未被更新**的死对象
            #   的值 (2.78e-17), 看起来像"算法不生效"。真正在用的 α 在 base.trace。
            "know_alpha_std": float(self.know.plasticity.alpha.std()),
        })
        return s

    def stats_extra(self):
        """主实验落盘用。"""
        if self.know is None:
            return {"knowledge": {"n_options": 0, "coverage": {}, "plasticity": {},
                                  "tau_U": None, "options": []},
                    "oak": {"use_options": self.use_options, "use_gate": self.use_gate,
                            "n_trans": len(self.trans), "refreshes": 0,
                            "option_steps": self.option_steps,
            "option_starts": self.option_starts,
                            "untrusted_picks": self.untrusted_picks}}
        d = self.know.describe()
        return {"knowledge": {
            "n_options": d["Level4_abstraction"]["n_options"],
            "coverage": d["meta_coverage"],
            "plasticity": d["meta_plasticity"],
            "tau_U": d["meta_uncertainty"]["tau_U"],
            "options": d["Level4_abstraction"]["options"],
        }, "oak": {"use_options": self.use_options, "use_gate": self.use_gate,
                   "n_trans": len(self.trans), "refreshes": self.refresh_count,
                   "option_steps": self.option_steps,
            "option_starts": self.option_starts,
                   "untrusted_picks": self.untrusted_picks}}
