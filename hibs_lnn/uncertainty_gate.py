"""uncertainty_gate.py — 覆盖度 + 不确定性门控, 与带不确定性的 rollout。

## 为什么需要

用户指出: 被动观测数据学出的 transition model **只知道数据覆盖过的 (s,a)**。
若 policy 提出训练数据里没出现过的动作, `T(s,a_new)` 可能完全是幻觉。
这正是我们实测到的: 转移模型对 `[2,2,2,2,2]` 预测 **0.9582**, 真实执行 **0.4357**。

于是把「覆盖度门控」升级为两个条件:

    allow(s,a) = [C(s,a) > tau_C]  ∧  [U_T(s,a) < tau_U]

    C(s,a) : 该 (状态, 动作) 有多少数据支持 (访问计数)
    U_T    : 转移预测的不确定性

## U_T 怎么估 —— 用集成分歧, 不用拍脑袋

单个模型无法报出自己有多不确定。标准做法是训练 **K 个 transition model**
(对数据做 bootstrap), 用它们的**预测分歧**作为 U_T:

    U_T(s,a) = mean_i || T_i(s,a) − mean_j T_j(s,a) ||

分歧大 = 该区域数据稀疏 / 模型没学到 → 不可信。
这与 coverage 互补: coverage 看「来过没有」, U_T 看「学稳了没有」。
"""
from __future__ import annotations

import numpy as np


class TransitionEnsemble:
    """K 个 bootstrap 转移模型的集成, 用预测分歧估计不确定性。

    T_i: x_{t+1} = x_t + Δ_i,  Δ_i = W_i @ φ(x_t, u)
    """

    def __init__(self, n_models=5, lam=1e-2, seed=0):
        self.K = int(n_models)
        self.lam = float(lam)
        self.rng = np.random.default_rng(seed)
        self.Ws = None
        self.n_dom = None
        self.dim_x = None
        self.fdim = None

    # ───────── 特征 ─────────
    @staticmethod
    def feat(x, u, n_dom, visits):
        """特征 = [x, onehot(u)]。

        ★ **不要加入常数列** (曾写成 concat([x, oh, [mean(visits)]]) 而
        mean(visits)≡1 与截距完全共线): 实测条件数被推到 **1.24e16**(近乎奇异),
        后果是双向的 ——
          ① 预测全错: Δ 估计 [35.6, 32.5, 10.3, 5.7], 真值应 ~0.3
          ② 不确定性失效: U_T 对所有动作都 ≈3.27, 完全不区分噪声
        去掉后: Δ 恢复 [0.30, 0, 0, 0], U_T 动作0(噪声.02)=0.0018 vs
        动作2(噪声.55)=0.0399 -> **22× 区分**。
        """
        oh = np.zeros(n_dom)
        if 0 <= u < n_dom:
            oh[u] = 1.0
        return np.concatenate([x, oh])

    # ───────── 拟合 (bootstrap) ─────────
    def fit(self, X, Y, n_dom, visits_dim=None):
        """X: (N, dim_x+1) = [x_t, u];  Y: (N, dim_x) = **绝对** x_{t+1}。

        ★ 接口约定: fit 与 predict **都用绝对下一状态**, 内部学 Δ 再加回 x。
        曾因约定不一致出过 bug —— 测试把绝对状态当 Δ 喂进来, 在质心 (x=0)
        处两者巧合相等, 掩盖了问题; 一旦远离质心 (R0=[4.95,4.55,1.36,1.22])
        就露出 `Δ=x+fx(u)` 的错误。现在接口自洽, 无法再误用。
        """
        self.dim_x = Y.shape[1]
        self.n_dom = int(n_dom)
        self.fdim = self.dim_x + self.n_dom          # 无截距列
        N = len(X)
        D = Y - X[:, :self.dim_x]                    # 内部转成 Δ
        self.Ws = []
        for k in range(self.K):
            idx = self.rng.integers(0, N, size=N)          # bootstrap 重采样
            F = np.array([self.feat(X[i, :self.dim_x], int(X[i, -1]),
                                    self.n_dom, np.ones(self.n_dom))
                          for i in idx])
            # ★ 不加截距列。onehot(u) 的各列之和恒等于 1, 与截距**完全共线**
            #   (Σ onehot = ones)。加了以后条件数 8.9e15, 解只在数据质心附近
            #   碰巧正确, 远离即炸: 实测在区域中心 (R0=[4.95,4.55,1.36,1.22])
            #   给出 Δ=[92.7, 96.4, 25.8, 23.7], 真值应为 [0.30, 0, 0, 0]。
            #   去掉后设计矩阵满秩 (rank = dim_x + n_dom)。
            A = F.T @ F + self.lam * np.eye(F.shape[1])
            self.Ws.append(np.linalg.solve(A, F.T @ D[idx]))
        return self

    # ───────── 预测 + 分歧 ─────────
    def predict(self, x, u, visits):
        """返回 (**绝对**下一状态的集成均值, 分歧 U_T)。"""
        if self.Ws is None:
            return np.asarray(x, dtype=float).copy(), float("inf")
        x = np.asarray(x, dtype=float)
        f = self.feat(x, u, self.n_dom, visits)             # 与 fit 一致, 无截距
        preds = np.array([f @ W for W in self.Ws])          # (K, dim_x) = Δ
        mean_d = preds.mean(0)
        unc = float(np.mean(np.linalg.norm(preds - mean_d, axis=1)))
        return x + mean_d, unc                              # 绝对下一状态

    def rollout(self, x0, actions, visits, gate=None):
        """多步 rollout: T^k(x0, a_{t:t+k})。返回 (轨迹, 逐步不确定性)。

        gate 非 None 时逐步做可信度检查; 一旦某步不可信即**中止并标记**,
        而不是继续外推 (Q8: 无覆盖 → 预测不可靠)。
        """
        x = np.asarray(x0, dtype=float).copy()
        traj, uncs, aborted_at = [x.copy()], [], None
        for t, u in enumerate(actions):
            if gate is not None:
                ok, why = gate.allow(x, u, visits)
                if not ok:
                    aborted_at = (t, why)
                    break
            x_next, unc = self.predict(x, u, visits)        # 已是绝对状态
            if not np.isfinite(unc):
                aborted_at = (t, "unc_inf")
                break
            x = x_next
            traj.append(x.copy())
            uncs.append(unc)
        return np.array(traj), np.array(uncs), aborted_at


class UncertaintyGate:
    """allow(s,a) = [C(s,a) > tau_C] ∧ [U_T(s,a) < tau_U]

    参数
    ----
    tau_C : 覆盖度阈值 (访问计数的分位/绝对阈值)
    tau_U : 不确定性阈值 (集成分歧)
    """

    def __init__(self, tau_C=1.0, tau_U=None, adaptive=True):
        self.tau_C = float(tau_C)
        self.tau_U = tau_U
        self.adaptive = adaptive
        self.unc_hist = []

    def calibrate(self, uncs):
        """用一批**在训练数据上**的集成分歧定阈值 (取 90 分位)。

        自适应是必要的: 分歧的量纲随任务变化, 硬编码阈值不可移植。
        """
        u = np.asarray(uncs, dtype=float)
        u = u[np.isfinite(u)]
        if u.size:
            self.tau_U = float(np.quantile(u, 0.90))
        return self.tau_U

    def allow(self, x, u, visits):
        """返回 (是否允许, 原因)。"""
        c = float(visits[u]) if u < len(visits) else 0.0
        if c <= self.tau_C:
            return False, "low_coverage"
        return True, "ok"          # 不确定性检查在 rollout 内逐步做

    def allow_with_unc(self, x, u, visits, unc):
        """含不确定性判定的完整版。"""
        ok, why = self.allow(x, u, visits)
        if not ok:
            return False, why
        if self.tau_U is not None and np.isfinite(unc) and unc > self.tau_U:
            return False, "high_uncertainty"
        return True, "ok"

    def stats(self):
        return {"tau_C": self.tau_C,
                "tau_U": None if self.tau_U is None else round(self.tau_U, 6)}
