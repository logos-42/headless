"""knowledge.py — InternalKnowledge: Agent 的**内部知识层**。

## 核心立场 (用户给的)

    Knowledge ≠ Parameter

参数只是**承载**知识的一种方式。OaK 式系统希望内部知识是**可组合的计算对象**,
而不是 `y = f_θ(x)` 里一堆"你甚至不知道 θ 代表什么知识"的权重。

所以这个类**不暴露 θ / W / α**,而是暴露**可计算的知识查询**:

    "未来会发生什么?"  -> predict()
    "我做什么会导致什么?" -> dynamics()
    "什么结果更好?"      -> value()
    "哪些过程是一个整体技能?" -> skills()
    "我对 X 有多确定?"   -> confidence()
    "我有多少经验?"      -> coverage_of()
    "我该多快改变?"      -> plasticity()

## 四层知识 (用户给的编号)

    Level 1  预测知识    K_P   GVF / predictors           "未来会发生什么"
    Level 2  因果/转移   K_T   T(s,a)                    "我做什么会导致什么"
    Level 3  价值/策略   K_V   V / Q / π                 "什么结果更好"
    Level 4  抽象知识    K_A   Options                    "哪些过程是一个整体技能"

外加两类**元知识**:

    uncertainty/   confidence           "我对自己的预测有多确定"
    coverage/      state_action 覆盖     "我对这个状态/动作到底有没有经验"
    plasticity/    α / β               "我应该多快改变这个知识" (接 IDBD)

## 为什么 α 也算内部知识

    w_i   = knowledge                  "我知道什么"
    α_i   = knowledge about plasticity "我应该多快改变这个知识"
    β_i   = meta-knowledge              "我的学习速度应该如何变化"

这与 IDBD 主线直接接上。

## 关键能力: 知识是**留下来的**

`to_dict()` / `load_dict()` 让 K 可以持久化比对 —— 这是"持续构建内部知识"的
可判定形式: 不是"模型又训了一轮",而是**能指出知识具体变了什么**。
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Level 1: 预测知识 (GVF / Horde 式)
# ──────────────────────────────────────────────────────────────────────────


class GVF:
    """General Value Function: 预测某个 cumulant 的折扣累积量。

        V(s) = E[ Σ γ^k λ^k c_{t+k+1} | s_t = s ]

    `c` = cumulant (要预测的信号), `γ` = 折扣, `λ` = 迹衰减。
    Horde (Sutton 2011) 的做法: 一堆 GVF 并行学, 它们的预测**进入状态**,
    构成 agent 对世界的"理解"—— 这正是 OaK 说的 internal knowledge。
    """

    def __init__(self, name, dim, gamma=0.9, lam=0.8, lr=0.05, seed=0):
        self.name = name
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.w = np.zeros(dim)
        self.trace = np.zeros(dim)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def predict(self, phi):
        return float(self.w @ phi)

    def update(self, phi, cumulant, phi_next):
        """TD(λ) 更新。"""
        v, v_next = self.w @ phi, self.w @ phi_next
        delta = cumulant + self.gamma * v_next - v
        self.trace = self.gamma * self.lam * self.trace + phi
        self.w += self.lr * delta * self.trace
        self.steps += 1
        return abs(delta)

    def stats(self):
        return {"name": self.name, "w_norm": float(np.linalg.norm(self.w)),
                "steps": self.steps, "gamma": self.gamma, "lam": self.lam}


class GVFBank:
    """一组 GVF。它们的预测拼起来就是 agent 的"世界理解"向量。"""

    def __init__(self, dim, spec=None, seed=0):
        self.spec = spec or [
            ("next_acc", 0.95, 0.9), ("acc_trend", 0.9, 0.7),
            ("forget_risk", 0.85, 0.8), ("regime_shift", 0.7, 0.5),
        ]
        self.gvfs = [GVF(n, dim, g, l, seed=seed + i)
                     for i, (n, g, l) in enumerate(self.spec)]

    def outputs(self, phi):
        """预测向量 —— 这就是"预测到的世界状态", 可作为下游特征。"""
        return np.array([g.predict(phi) for g in self.gvfs])

    def update(self, phi, cumulants, phi_next):
        errs = []
        for g, c in zip(self.gvfs, cumulants):
            errs.append(g.update(phi, float(c), phi_next))
        return float(np.mean(errs)) if errs else 0.0

    def stats(self):
        return {"n_gvf": len(self.gvfs),
                "gvfs": [g.stats() for g in self.gvfs],
                "out_norm": float(np.linalg.norm(self.outputs(np.ones_like(self.gvfs[0].w)))),
                "mean_steps": float(np.mean([g.steps for g in self.gvfs])) if self.gvfs else 0.0}


# ──────────────────────────────────────────────────────────────────────────
# 元知识: 覆盖度 / 可塑性
# ──────────────────────────────────────────────────────────────────────────


class Coverage:
    """state-action 覆盖度 —— "我对这个 (s,a) 到底有没有经验"。"""

    def __init__(self, n_dom):
        self.n_dom = n_dom
        self.n = np.zeros(n_dom)        # 每动作被访问次数
        self.n_total = 0

    def observe(self, a):
        if 0 <= a < self.n_dom:
            self.n[a] += 1
        self.n_total += 1

    def of(self, a):
        return float(self.n[a]) if 0 <= a < self.n_dom else 0.0

    def gate(self, a):
        """g = 1/√(1+n): 零覆盖 -> 1, 经验多 -> 0。"""
        return float(1.0 / np.sqrt(1.0 + self.of(a)))

    def stats(self):
        nz = self.n[self.n > 0]
        return {"total": int(self.n_total), "per_action": self.n.astype(int).tolist(),
                "min": float(nz.min()) if len(nz) else 0.0,
                "max": float(nz.max()) if len(nz) else 0.0,
                "zero_actions": int((self.n == 0).sum())}


class Plasticity:
    """元知识: α (该多快改变某个参数) 与 β (学习速度本身的变化)。

    这是 IDBD 主线在"内部知识"里的位置 —— 它**不是**知识本身,
    而是"关于知识该如何改变的"元知识。
    """

    def __init__(self, dim, alpha0=0.05, mu=0.01):
        self.alpha0 = alpha0
        self.mu = mu
        self.beta = np.zeros(dim)
        self.h = np.zeros(dim)
        self.updates = 0

    @property
    def alpha(self):
        return np.exp(np.clip(self.beta, -20, 20)) * self.alpha0

    def update(self, phi, delta):
        g = delta * phi * self.h
        self.h = np.maximum(0.0, self.h * (1.0 - np.clip(self.alpha * phi ** 2, 0, 1)) + self.alpha * phi * delta)
        self.beta = np.clip(self.beta + self.mu * g, -20, 20)
        self.updates += 1

    def stats(self):
        a = self.alpha
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "ratio_max_min": float(a.max() / max(a.min(), 1e-12)),
                "updates": self.updates}


# ──────────────────────────────────────────────────────────────────────────
# InternalKnowledge: 统一的知识容器
# ──────────────────────────────────────────────────────────────────────────


class InternalKnowledge:
    """K = {predictions, dynamics, value, options, uncertainty, coverage, plasticity}

    ★ 不暴露参数, 只暴露**知识查询**。
    """

    def __init__(self, n_dom, dim_x, n_models=5, seed=0, alpha0=0.05, mu=0.01):
        self.n_dom = int(n_dom)
        self.dim_x = int(dim_x)
        # Level 1 预测知识 K_P
        self.predictions = GVFBank(dim_x, seed=seed)
        # Level 2 因果/转移知识 K_T
        from hibs_lnn.uncertainty_gate import TransitionEnsemble
        self.dynamics_model = TransitionEnsemble(n_models=n_models, seed=seed)
        self._dyn_fitted = False
        # Level 3 价值/策略知识 K_V  (由外部 proposer 提供, 这里只登记)
        self.value_fn = None
        self.policy = None
        # Level 4 抽象知识 K_A
        self.options = None
        # 元知识
        from hibs_lnn.uncertainty_gate import UncertaintyGate
        self.uncertainty = UncertaintyGate(tau_C=1.0)
        self.coverage = Coverage(self.n_dom)
        self.plasticity = Plasticity(dim_x, alpha0=alpha0, mu=mu)
        self.seed = seed

    # ── 写入 ──────────────────────────────────────────────────────────
    def fit_dynamics(self, X, Y):
        self.dynamics_model.fit(X, Y, self.n_dom, None)
        self._dyn_fitted = True
        # 标定不确定性阈值
        visits = self.coverage.n
        n = len(X)
        step = max(1, n // 100)
        self.uncertainty.calibrate(
            [self.dynamics_model.predict(X[i, :self.dim_x], int(X[i, -1]), visits)[1]
             for i in range(0, n, step)])
        return self

    def register_value_fn(self, fn, policy=None):
        """登记价值/策略知识 (K_V)。fn(s, a) -> V。"""
        self.value_fn = fn
        self.policy = policy
        return self

    def register_options(self, option_manager):
        """登记抽象知识 (K_A)。"""
        self.options = option_manager
        return self

    def observe_gvf(self, phi, cumulants, phi_next):
        return self.predictions.update(phi, cumulants, phi_next)

    def observe_action(self, a):
        self.coverage.observe(a)

    def observe_plasticity(self, phi, delta):
        self.plasticity.update(phi, delta)

    # ── 查询 (知识接口) ────────────────────────────────────────────────
    def predict(self, s, a):
        """未来会发生什么 (Level 1+2 合成)。"""
        if not self._dyn_fitted:
            return np.asarray(s, dtype=float).copy(), float("inf")
        return self.dynamics_model.predict(s, a, self.coverage.n)

    def value(self, s, a):
        """什么结果更好 (Level 3)。

        ★ **对未知输入必须返回 NaN, 而不是抛异常**。
        知识层被问到一个它没见过的 (s,a) 时, 正确反应是"我不知道", 不是崩。
        (这条是实测抓到的: 测试里对从未出现过的动作 7 查询, value_fn 抛 KeyError
        一路炸穿 knowledge_of —— 而"承认不知道"恰恰是这一层的核心语义。)
        """
        if self.value_fn is None:
            return float("nan")
        try:
            return float(self.value_fn(s, a))
        except Exception:
            return float("nan")

    def skills(self):
        """哪些过程是一个整体技能 (Level 4)。"""
        if self.options is None:
            return []
        return [o.stats() for o in getattr(self.options, "options", [])]

    def confidence(self, s, a):
        """我对自己的预测有多确定 —— 返回 [0,1], 1 = 最确定。"""
        _, unc = self.predict(s, a)
        if not np.isfinite(unc):
            return 0.0
        tau = self.uncertainty.tau_U
        if tau is None or not np.isfinite(tau) or tau <= 0:
            return 0.5
        return float(1.0 / (1.0 + unc / tau))

    def coverage_of(self, a):
        return self.coverage.of(a)

    def knowledge_of(self, s, a):
        """★★ 统一查询: 对任意 (s,a), agent 到底知道什么。

        这是 Knowledge≠Parameter 的直接体现 —— 问的是**内容**不是参数。
        """
        s_next, unc = self.predict(s, a)
        val = self.value(s, a)
        novel = bool(self.coverage_of(a) == 0)
        known = bool(not novel and np.isfinite(unc))
        return {
            "state": np.round(np.asarray(s, dtype=float), 4).tolist(),
            "action": int(a),
            "predicted_next": np.round(np.asarray(s_next), 4).tolist(),
            "uncertainty": float(unc) if np.isfinite(unc) else None,
            "confidence": self.confidence(s, a),
            "value": (val if np.isfinite(val) else None),
            "coverage": self.coverage_of(a),
            "coverage_gate": self.coverage.gate(a),
            "trustworthy": bool(self.uncertainty.allow(s, a, self.coverage.n)[0]),
            "novel": novel,
            # ★ 这一项是整层的立命之处: agent 明确区分「我知道」与「我不知道」
            "known": known,
            "verdict": ("我不知道" if not known else
                        ("不太确定" if self.confidence(s, a) < 0.5 else "我知道")),
        }

    # ── 摘要 / 持久化 ─────────────────────────────────────────────────
    def describe(self) -> dict:
        """分层摘要: 这是我"知道了什么"的完整画像。"""
        return {
            "Level1_predictions": self.predictions.stats(),
            "Level2_dynamics": {"fitted": self._dyn_fitted,
                                "n_models": len(getattr(self.dynamics_model, "Ws", []) or []),
                                "dim_x": self.dim_x, "n_dom": self.n_dom},
            "Level3_value": {"registered": self.value_fn is not None,
                             "policy_registered": self.policy is not None},
            "Level4_abstraction": {"n_options": len(self.skills()),
                                   "options": self.skills()[:6]},
            "meta_uncertainty": {"tau_U": self.uncertainty.tau_U,
                                 "tau_C": self.uncertainty.tau_C},
            "meta_coverage": self.coverage.stats(),
            "meta_plasticity": self.plasticity.stats(),
        }

    def to_dict(self) -> dict:
        """知识可以被**留下来** —— 持久化后能与旧版本比对。"""
        return {
            "coverage_n": self.coverage.n.tolist(),
            "coverage_total": self.coverage.n_total,
            "gvf_w": [g.w.tolist() for g in self.predictions.gvfs],
            "gvf_steps": [g.steps for g in self.predictions.gvfs],
            "dyn_W": [W.tolist() for W in (getattr(self.dynamics_model, "Ws", None) or [])],
            "tau_U": self.uncertainty.tau_U,
            "plasticity_beta": self.plasticity.beta.tolist(),
            "plasticity_updates": self.plasticity.updates,
            "n_options": len(self.skills()),
        }

    def growth(self, old_dict) -> dict:
        """★ 与旧版本比对: 知识**具体**变了什么。

        这是"持续构建内部知识"的可判定形式 —— 不是"又训了一轮",
        而是能逐项指出知识的变化量。
        """
        if not old_dict:
            return {"note": "no prior knowledge"}
        out = {}
        # 覆盖度变化
        old_n = np.array(old_dict.get("coverage_n", [0] * self.n_dom), dtype=float)
        out["coverage_delta"] = (self.coverage.n - old_n).astype(int).tolist()
        out["coverage_total_delta"] = int(self.coverage.n_total - old_dict.get("coverage_total", 0))
        # GVF 权重变化 (预测知识变了多少)
        gw_old = old_dict.get("gvf_w")
        if gw_old is not None and len(gw_old) == len(self.predictions.gvfs):
            d = [float(np.linalg.norm(np.array(g.w) - np.array(o)))
                 for g, o in zip(self.predictions.gvfs, gw_old)]
            out["gvf_weight_delta"] = np.round(d, 6).tolist()
            out["gvf_weight_delta_mean"] = float(np.mean(d))
        # 转移模型变化
        dw_old = old_dict.get("dyn_W")
        cur = getattr(self.dynamics_model, "Ws", None)
        if dw_old and cur and len(dw_old) == len(cur):
            d = [float(np.linalg.norm(np.array(W) - np.array(o))) for W, o in zip(cur, dw_old)]
            out["dyn_weight_delta_mean"] = float(np.mean(d))
        # 可塑性元知识变化
        pb_old = old_dict.get("plasticity_beta")
        if pb_old is not None and len(pb_old) == len(self.plasticity.beta):
            out["plasticity_beta_delta"] = np.round(
                self.plasticity.beta - np.array(pb_old), 6).tolist()
        out["n_options_delta"] = len(self.skills()) - old_dict.get("n_options", 0)
        return out

    def stats(self):
        return {"n_dom": self.n_dom, "dim_x": self.dim_x,
                "dyn_fitted": self._dyn_fitted, "n_options": len(self.skills()),
                "coverage_total": int(self.coverage.n_total)}
