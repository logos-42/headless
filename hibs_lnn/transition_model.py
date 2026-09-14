"""transition_model.py — OaK 第③条的第一个缺失件。

## 为什么需要它

OaK feature #3:
    feature -> subproblem -> solution -> **transition model** -> planning

我们已经有 feature / subproblem(6 域 / 60 细区间)/ solution(训练),
但**没有转移模型也无法规划** —— 这是 leo 那张映射表里真正缺的东西。

## 定义 (与 lm4 的数据结构对齐)

    能力状态 a_t    = acc_matrix[t]        (在 6 个域上的准确率画像, 6 维)
    动作     u_t    = 第 t 步训练了哪个域   (one-hot, 6 维)
    转移     T      : (a_t, u_t, ctx) -> Δa,   a_{t+1} = a_t + Δa

学 **Δa** 而不是 a_{t+1}, 因为这样 "copy 基线" 就是 Δa≡0,
模型有没有学到东西一目了然。

## 输入特征

    x = [ a_t (6) | onehot(u_t) (6) | n_visits (6) | t/n_rounds (1) ]   = 19 维

## 基线 (必须打赢才有意义)

    copy     : Δa ≡ 0                     实测 MAE 0.2986
    constant : Δa ≡ mean(Δa|u)            实测 MAE 0.2723

用 **ridge** 而不是 MLP: 样本量 ~1.8K, 参数少、可复现、无调参空间
(符合 leo 的"manual > framework"偏好)。
"""
from __future__ import annotations

import numpy as np


class TransitionModel:
    """T(a_t, u_t, ctx) -> Δa 的线性(脊)转移模型。

    参数
    ----
    lam : 岭正则强度
    """

    def __init__(self, lam=1e-2):
        self.lam = float(lam)
        self.W = None            # (fdim, n_dom)
        self.n_dom = None
        self.fitted = False

    # ───────── 特征 ─────────
    @staticmethod
    def feat(a, u, n_visits, t_frac):
        oh = np.zeros(len(a))
        if 0 <= u < len(a):
            oh[u] = 1.0
        return np.concatenate([a, oh, n_visits, [t_frac]])

    def build_xy(self, records):
        """records: [(acc_matrix, dom_seq)] -> X, Y, groups"""
        X, Y, G = [], [], []
        for gi, (M, seq) in enumerate(records):
            M = np.asarray(M, dtype=float)
            n_dom = M.shape[1]
            nvis = np.zeros(n_dom)
            for t, u in enumerate(seq):
                if t + 1 >= M.shape[0]:
                    break
                if np.isnan(M[t]).any() or np.isnan(M[t + 1]).any():
                    nvis[u] += 1
                    continue
                X.append(self.feat(M[t], int(u), nvis.copy(), t / max(1, len(seq))))
                Y.append(M[t + 1] - M[t])
                G.append(gi)
                nvis[u] += 1
        return (np.array(X), np.array(Y), np.array(G)) if X else (
            np.zeros((0, 19)), np.zeros((0, 6)), np.zeros(0))

    # ───────── 拟合 ─────────
    def fit(self, records):
        X, Y, _ = self.build_xy(records)
        if len(X) == 0:
            raise ValueError("没有可用样本")
        self.n_dom = int(Y.shape[1])
        # 只对 Δa 做岭回归 (含截距)
        X1 = np.hstack([X, np.ones((len(X), 1))])
        A = X1.T @ X1 + self.lam * np.eye(X1.shape[1])
        self.W = np.linalg.solve(A, X1.T @ Y)
        self.fitted = True
        return self

    def predict_delta(self, a, u, n_visits, t_frac):
        x = np.append(self.feat(a, u, n_visits, t_frac), 1.0)
        return x @ self.W

    # ───────── 规划: 用 T 前向搜索选动作序列 ─────────
    def plan(self, a0, horizon, beam=8, n_dom=None):
        """beam search: 在 T 的预测下最大化末端平均准确率。

        这是 OaK 的 "planning" —— 用学到的转移模型做前向搜索,
        而不是靠外生给定的 schedule。
        """
        n_dom = int(n_dom if n_dom is not None else (self.n_dom or 0))
        a0 = np.asarray(a0, dtype=float)
        # beam 元素: (score, a, nvis, seq)
        beams = [(float(np.nanmean(a0)), a0.copy(), np.zeros(n_dom), [])]
        for t in range(horizon):
            cand = []
            for _, a, nvis, seq in beams:
                for u in range(n_dom):
                    d = self.predict_delta(a, u, nvis, t / max(1, horizon))
                    a2 = a + d
                    nv2 = nvis.copy()
                    nv2[u] += 1
                    cand.append((float(np.nanmean(a2)), a2, nv2, seq + [u]))
            cand.sort(key=lambda r: -r[0])
            beams = cand[:beam]
        return beams[0][3]
