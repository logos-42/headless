"""dual_proposer.py — 把两个学习信号严格分开 (Q15: 「两个不要混成一个东西」)。

## 为什么需要这个

Q15 明确要求两类学习信号各自独立:

    Prediction error   δ^pred = x_{t+1} − x̂_{t+1}          用于学习**环境**
    Control/TD error   δ^RL   = r + γV(s_{t+1}) − V(s_t)    用于学习**价值/控制**

而此前的 `RLProposer` **只有一个 reward = Δ(any-time)**, 没有独立预测通路 ——
预测信息只能通过 reward 间接影响, 两个信号实际是混的。

## 结构

    ┌─────────────────────────────────────────────────────────┐
    │ 通路 P (world model / 预测)                              │
    │   x̂_{t+1} = W_p · φ_p(x_t, u_t)                         │
    │   δ^pred  = x_{t+1} − x̂_{t+1}                            │
    │   W_p ← W_p + η_p · δ^pred ⊗ φ_p      ← **只由预测误差更新** │
    │   产出: 「训练候选 i 后能力画像会变成什么」＋ 覆盖度            │
    └─────────────────────────────────────────────────────────┘
                            │  只作为**特征**进入
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 通路 V (value / 控制)                                    │
    │   V_i = θ_v · φ_v(s, a)                                  │
    │   δ^RL = r + γ·V(s') − V(s)                              │
    │   θ_v ← θ_v + α_i · δ^RL · φ_v        ← **只由 TD 误差更新** │
    │   α_i 由 IDBD 累积 (每权重独立步长)                        │
    └─────────────────────────────────────────────────────────┘

**关键**: 两套参数 (W_p, θ_v) 完全不共享。预测通路的输出作为 φ_v 的**一个分量**,
而不是把 δ^pred 直接加进 δ^RL —— 那才是「混」。
这样任何一个通路的误差都不会直接污染另一个的更新。

## 覆盖度门控 (Q8)

Horde 那篇 Q8: *"off-policy ≠ 无数据也能预测; 目标策略必须在现有经验中得到足够的
信息支持, 否则: 没有数据覆盖 → 预测不可靠。"*

实现: 每个 (描述子区域) 维护访问计数 n_i, 门控 g_i = 1/√(1+n_i)。
预测分量按 g_i 缩放后才进入 φ_v —— **无覆盖区域的预测不参与价值判断**。
这正是我们实测证伪的规划幻觉 (`[2,2,2,2,2]` 预测 0.9582 / 实测 0.4357) 的对策。
"""
from __future__ import annotations

import numpy as np


class DualSignalProposer:
    """δ^pred 与 δ^RL 分流; 预测输出只作特征, 两套参数不共享。

    参数
    ----
    desc      : (N, D) 候选描述子
    dim_x     : 能力状态的维度 (要预测的量, 如 6 个域)
    k         : 每次提议几个候选
    eta_p     : 预测通路的步长 (只作用于 W_p)
    mu, alpha0: 价值通路的 IDBD 元步长 / 初始步长 (只作用于 θ_v)
    gamma     : 价值通路的折扣
    lam_pred  : 预测分量进入 φ_v 的权重
    """
    def __init__(self, desc, dim_x=6, k=2, eta_p=0.05,
                 mu=0.05, alpha0=0.2, gamma=1.0, lam_pred=1.0,
                 explore_w=0.5, seed=0):
        desc = np.asarray(desc, dtype=float)
        self.n, self.d = desc.shape
        mu_d, sd_d = desc.mean(0), desc.std(0)
        sd_d[sd_d < 1e-9] = 1.0
        self.z = (desc - mu_d) / sd_d
        self.dim_x = int(dim_x)
        self.k, self.gamma, self.lam_pred = int(k), float(gamma), float(lam_pred)
        self.explore_w, self.rng = float(explore_w), np.random.default_rng(seed)

        # ── 通路 P: 世界模型 (只由 δ^pred 更新) ──
        #   φ_p = [1, z(a), 1/N·Σ_t x_t]   输入含候选 + 当前能力均值
        self.fp = 2 + self.d
        self.W_p = np.zeros((self.dim_x, self.fp))
        self.eta_p = float(eta_p)
        self.pred_err = []                   # δ^pred 的历史 (诊断)
        self._x_prev = np.zeros(self.dim_x)

        # ── 通路 V: 价值 (只由 δ^RL 更新) ──
        #   φ_v = [1, z(a), pred_term·g, cov, unc]
        self.fv = 1 + self.d + self.dim_x + 2
        self.theta = np.zeros(self.fv)
        self.alpha = np.full(self.fv, float(alpha0))
        self.h = np.zeros(self.fv)
        self.mu = float(mu)

        # ── 覆盖度 (Q8 门控) ──
        self.visits = np.zeros(self.n)
        self.any_time = 0.0
        self.global_acc = 0.0
        self.trace = []

    # ───────── 特征 (两条通路各自独立) ─────────
    def phi_p(self, i):
        """预测通路的特征: 候选 + 当前能力均值。"""
        return np.concatenate([[1.0], self.z[i], [self._x_prev.mean()]])

    def predict_next(self, i):
        """x̂_{t+1} = W_p · φ_p(x_t, u_i) —— 纯预测, 不含价值。"""
        return self.W_p @ self.phi_p(i)

    def gate(self, i):
        """覆盖度门控: 无覆盖区域的预测不参与价值判断 (Q8)。"""
        return 1.0 / np.sqrt(1.0 + self.visits[i])

    def phi_v(self, i):
        """价值通路的特征: **预测输出以门控后的形式进入** (不是 δ^pred)。"""
        cov = 1.0 / (1.0 + self.visits[i])
        unc = 1.0 / np.sqrt(1.0 + self.visits[i])
        pred = self.predict_next(i) * self.gate(i) * self.lam_pred
        return np.concatenate([[1.0], self.z[i], pred, [cov, unc]])

    def value(self, i):
        return float(self.theta @ self.phi_v(i))

    def score(self, i):
        return self.value(i) + self.explore_w / np.sqrt(1.0 + self.visits[i])

    def probs(self):
        s = np.array([self.score(i) for i in range(self.n)])
        s = np.clip((s - s.max()) / 0.5, -30.0, 0.0)
        e = np.exp(s)
        tot = e.sum()
        return e / tot if tot > 0 and np.isfinite(tot) else np.full(self.n, 1.0 / self.n)

    # ───────── 动作 ─────────
    def act(self, k=None, greedy=False):
        k = k or self.k
        if greedy:
            order = np.argsort([self.score(i) for i in range(self.n)])[::-1]
            idx = list(map(int, order[:k]))
        else:
            idx = list(map(int, self.rng.choice(self.n, size=min(k, self.n),
                                                replace=False, p=self.probs())))
        for i in idx:
            self.visits[i] += 1
        return idx

    # ───────── 两个信号, 分别进各自通路 ─────────
    # ───────── 两条通路**各自独立**的更新方法 (严格分离) ─────────
    def update_prediction(self, idx, x_next):
        """只更新预测通路 (W_p), 用 δ^pred = x_{t+1} − x̂_{t+1}。

        **绝不触碰 θ_v / alpha / h。** 这是「两个信号不混」的可执行形式。
        返回本步 δ^pred 的均值。
        """
        x_next = np.asarray(x_next, dtype=float)
        ds = []
        for i in idx:
            d = x_next - self.predict_next(i)        # δ^pred
            self.W_p += self.eta_p * np.outer(d, self.phi_p(i))
            ds.append(float(np.mean(np.abs(d))))
        self.pred_err.append(float(np.mean(ds)))
        self._x_prev = x_next.copy()
        return self.pred_err[-1]

    def update_value(self, idx, r=None):
        """只更新价值通路 (θ_v / alpha / h), 用 δ^RL = r + γV(s') − V(s)。

        **绝不触碰 W_p。** 返回本步 δ^RL 的均值。
        """
        if r is None:
            r = self.any_time - self.global_acc
        self.global_acc = self.any_time
        for i in idx:
            self.visits[i] += 1                      # 价值通路自己记账覆盖
        drl = []
        for i in idx:
            phi = self.phi_v(i)
            v = float(self.theta @ phi)
            d = float(r + self.gamma * 0.0 - v)     # γ=1 时退化为 r − V (bandit)
            drl.append(d)
            self.theta += self.alpha * d * phi
            self.h = self.h * np.maximum(0.0, 1.0 - self.alpha * phi ** 2) \
                + self.alpha * phi * d
            self.alpha = np.clip(self.alpha * np.exp(
                np.clip(self.mu * d * phi * self.h, -2.0, 2.0)), 1e-6, 1.0)
        self.trace.append({"idx": list(map(int, idx)),
                           "delta_pred": round(self.pred_err[-1], 6)
                           if self.pred_err else 0.0,
                           "delta_rl": round(float(np.mean(drl)), 6),
                           "alpha_std": round(float(self.alpha.std()), 6),
                           "w_pred_norm": round(float(np.linalg.norm(self.W_p)), 5)})
        return float(np.mean(drl))

    def update(self, idx, r=None, x_next=None):
        """便捷入口: 依次调两条通路。**两条通路仍各自独立** ——
        交叉验证见 tests/test_dual_signal.py (只喂一路时另一路逐位不变)。
        """
        if x_next is not None:
            self.update_prediction(idx, x_next)
        return self.update_value(idx, r=r)

    # ───────── 覆盖度检查 (Q8): 这个反事实可不可信 ─────────
    def counterfactual(self, i, min_visits=3):
        """对被门控掉的候选给出「预测不可信」标记, 而不是硬报一个数。

        返回 (x̂_next, 可信?, 门控值)。**这是对规划幻觉的直接对策** ——
        实测 `[2,2,2,2,2]` 的预测 0.9582 / 实测 0.4357, 就是因为当时没有这道门。
        """
        g = self.gate(i)
        return self.predict_next(i), bool(self.visits[i] >= min_visits), float(g)

    def stats(self):
        al = self.alpha
        return {
            "pred_err_mean": round(float(np.mean(self.pred_err[-20:]))
                                   if self.pred_err else 0.0, 6),
            "w_pred_norm": round(float(np.linalg.norm(self.W_p)), 5),
            "theta_norm": round(float(np.linalg.norm(self.theta)), 5),
            "alpha_std": round(float(al.std()), 6),
            "alpha_ratio": round(float(al.max() / max(1e-12, al.min())), 2),
            "coverage": round(float((self.visits > 0).mean()), 4),
            # ★ 两个信号各自的量级, 用来证明它们**没有混在一起**
            "signal_gap": round(abs(float(np.mean(self.pred_err[-20:])
                                         if self.pred_err else 0.0)), 6),
        }
