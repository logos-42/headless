"""option_manager.py — Options / 时间抽象 (OaK 的 O)。

## Option 的三要素 (用户定义)

    o = (I_o, π_o, β_o)

    I_o  initiation set    什么时候可以启动这个 option
    π_o  option policy     启动后怎么行动 (这里是**一串子动作**, 不是单个动作)
    β_o  termination       什么时候结束 —— **学习式** P(terminate | s, o)

## ★ 不从 reward 定义 option, 而从「可预测的状态转移结构」发现

用户: *"Option 不一定首先由 reward 定义, 而可以由可预测的状态转移结构产生。"*

我们已有 `TransitionEnsemble` (带分歧 U_T)。发现流程:

    1. 用过渡模型把能力状态 s 映射到预测状态 ŝ, 聚类成 region
    2. 建 region 转移图: 边 (A --u--> B) 的**可预测性** = 1/(1+U_T)
    3. 在图上找多步可靠路径 A → C (要求每一步 U_T < τ_U)
    4. 每条这样的路径 = 一个 option: π_o = 该动作序列, I_o = A 区域
    5. β_o 用真实观测学: P(到达目标区域 | s, o)  —— 学习式终止, 不是固定阈值

**为什么这比手工写 "Option1=加磁通 / Option2=稳定" 强**: 手工定义只编码了人类先验;
从转移结构发现是在编码 **"哪些行为的后果我可以可靠预测"** —— 那才是可利用的时间抽象。

## 与 planning 的衔接

高层: s_t → 选 option → **rollout transition model** → 选动作序列
这正是 OaK 的 abstraction → transition model → planning 路线。
"""
from __future__ import annotations

import numpy as np


class Option:
    """o = (I_o, π_o, β_o)。"""

    def __init__(self, oid, actions, start_center, goal_center, uncs=None):
        self.oid = int(oid)
        self.actions = list(actions)        # π_o: 动作序列 (时间抽象的核心)
        self.start_center = np.asarray(start_center, dtype=float)   # I_o 的中心
        self.goal_center = np.asarray(goal_center, dtype=float)     # 目标区域
        self.uncs = list(uncs or [])        # 逐步不确定性 (用于可信度)
        self.visits = 0
        self.success = 0
        # β_o 的线性参数: P(terminate | s, o) = sigmoid(w·[s, goal, unc])
        self.w_beta = np.zeros(3 * len(self.goal_center) + 1)

    # ── I_o: 启动条件 (到起点的距离在半径内) ──
    def initiable(self, s, radius=0.5):
        return float(np.linalg.norm(np.asarray(s) - self.start_center)) <= radius

    # ── β_o: **学习式终止概率** ──
    def _beta_feat(self, s, unc=0.0):
        s = np.asarray(s, dtype=float)
        g = self.goal_center
        return np.concatenate([s, g - s, [unc, 1.0]])

    def beta_prob(self, s, unc=0.0):
        z = float(self.w_beta @ self._beta_feat(s, unc))
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def learn_termination(self, samples, lr=0.05, epochs=200):
        """从真实观测学 β_o。samples: [(s, unc, terminated_bool), ...]

        这才是「学习式 termination」—— 而不是 `if error < threshold: stop()`。
        """
        if not samples:
            return
        F = np.array([self._beta_feat(s, u) for s, u, _ in samples])
        y = np.array([1.0 if t else 0.0 for _, _, t in samples])
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-np.clip(F @ self.w_beta, -30, 30)))
            grad = F.T @ (p - y) / len(y)
            self.w_beta -= lr * grad
        return self

    def should_terminate(self, s, unc=0.0, thresh=0.5):
        return self.beta_prob(s, unc) >= thresh

    def stats(self):
        rate = (self.success / self.visits) if self.visits else 0.0
        return {"oid": self.oid, "len": len(self.actions),
                "start": np.round(self.start_center, 3).tolist(),
                "goal": np.round(self.goal_center, 3).tolist(),
                "visits": self.visits, "success_rate": round(rate, 3),
                "mean_unc": round(float(np.mean(self.uncs)) if self.uncs else 0.0, 5)}


class OptionManager:
    """从可预测的状态转移结构中**发现** options。"""

    def __init__(self, ensemble, gate, max_len=4, min_edges=3, seed=0):
        self.T = ensemble
        self.gate = gate
        self.max_len = int(max_len)
        self.min_edges = int(min_edges)
        self.options = []
        self.regions = None          # region 中心
        self.graph = {}              # (region, action) -> (next_region, unc)
        self.rng = np.random.default_rng(seed)

    # ───────── 发现 ─────────
    def discover(self, states, actions, n_regions=4, tau_U=None):
        """输入: 观测到的 (状态, 动作) 对。输出: options 列表。

        states : (N, dim_x) 能力状态
        actions: (N,) 当时训练了哪个域
        """
        S = np.asarray(states, dtype=float)
        A = np.asarray(actions, dtype=int)
        n_dom = int(A.max()) + 1 if len(A) else 1
        visits = np.zeros(n_dom)
        for a in A:
            visits[a] += 1

        # 1) 聚类成 region (用最简单可复现的 k-means, 避免引入依赖)
        self.regions, assign = self._kmeans(S, n_regions)

        # 2) 建转移图: 对每个 (region, action) 预测落点与不确定性
        self.graph = {}
        for r in range(n_regions):
            for u in range(n_dom):
                c = self.regions[r]
                s_next, unc = self.T.predict(c, u, visits)   # 已是绝对状态
                nxt = self._nearest_region(s_next)
                self.graph[(r, u)] = (nxt, unc)

        # 3) 定 τ_U (若未给) —— 用图上所有边的分歧分位
        uncs = [self.graph[k][1] for k in self.graph]
        uncs = [u for u in uncs if np.isfinite(u)]
        if tau_U is None:
            # 用**中位数**而非高分位: 我们要的是「一半的可预测边」, 不是几乎全部。
            tau_U = float(np.median(uncs)) if uncs else float("inf")
        self.tau_U = tau_U

        # 4) 找可靠路径 (每步 unc < tau_U 且确实前进了) = options
        self.options = []
        oid = 0
        for r0 in range(n_regions):
            for L in range(2, self.max_len + 1):
                for path in self._reliable_paths(r0, L, tau_U):
                    acts = path
                    goal_r = self._walk(r0, acts)
                    if goal_r == r0:
                        continue                      # 没前进, 不是 option
                    uncs_p = [self.graph[(self._walk(r0, acts[:i]), acts[i])][1]
                              for i in range(len(acts))]
                    self.options.append(Option(
                        oid, acts, self.regions[r0], self.regions[goal_r], uncs_p))
                    oid += 1
        # 去重 (同起点同目标保留最短的)
        best = {}
        for o in self.options:
            k = (tuple(np.round(o.start_center, 3)), tuple(np.round(o.goal_center, 3)))
            if k not in best or len(o.actions) < len(best[k].actions):
                best[k] = o
        self.options = list(best.values())
        for i, o in enumerate(self.options):
            o.oid = i
        return self.options

    def _reliable_paths(self, r0, L, tau_U):
        """长度 L 的路径, 每步 unc < tau_U。用 BFS 枚举 (空间很小)。"""
        out = []
        n_dom = max(u for _, u in self.graph) + 1 if self.graph else 1
        frontier = [(r0, [])]
        for depth in range(L):
            nxt = []
            for r, seq in frontier:
                for u in range(n_dom):
                    e = self.graph.get((r, u))
                    if e is None or not np.isfinite(e[1]) or e[1] >= tau_U:
                        continue
                    if e[0] == r:
                        continue                      # 不前进的边不走
                    nxt.append((e[0], seq + [u]))
            if depth == L - 1:
                out.extend(seq for _, seq in nxt)
            frontier = nxt
            if not frontier:
                break
        return out

    def _walk(self, r0, acts):
        r = r0
        for u in acts:
            e = self.graph.get((r, u))
            if e is None:
                return r
            r = e[0]
        return r

    def _nearest_region(self, s):
        d = np.linalg.norm(self.regions - np.asarray(s), axis=1)
        return int(np.argmin(d))

    @staticmethod
    def _kmeans(X, k, iters=50, seed=0):
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
        a = np.zeros(len(X), dtype=int)          # 防 iters=0 时未绑定
        for _ in range(iters):
            d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
            a = np.argmin(d, axis=1)
            for j in range(len(C)):
                if (a == j).any():
                    C[j] = X[a == j].mean(0)
        return C, a

    # ───────── 高层选择 (planning 入口) ─────────
    def select(self, s, goal=None, exclude_unc=True):
        """选一个 option: 优先「起点匹配 + 目标最接近 goal + 成功率高」。

        goal=None 时选成功率高且不确定性低的那个。
        """
        cands = []
        for o in self.options:
            if not o.initiable(s, radius=0.75):
                continue
            if exclude_unc and o.uncs and max(o.uncs) >= getattr(self, "tau_U", np.inf):
                continue
            rate = (o.success / o.visits) if o.visits else 0.5     # 先验乐观
            score = rate - (float(np.mean(o.uncs)) if o.uncs else 0.0)
            if goal is not None:
                score -= float(np.linalg.norm(o.goal_center - np.asarray(goal)))
            cands.append((score, o))
        return max(cands, key=lambda t: t[0])[1] if cands else None

    def stats(self):
        return {"n_options": len(self.options),
                "tau_U": round(float(getattr(self, "tau_U", float("nan"))), 6),
                "n_edges": len(self.graph),
                "options": [o.stats() for o in self.options[:6]]}
