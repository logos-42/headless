"""rl_proposer.py — 真正的 RL 提议器 (取代固定公式的 RegimeProposer)。

## 为什么必须有这个文件

`RegimeProposer` 是个**启发式排序器**: V(p) 是状态的纯函数, 权重 λ 是超参,
**从不被 reward 更新, 也从不积累**。leo 连续两问戳中了它:
  「有把 policy 当做一个可以迭代和积累的参数来更新吗, 积累过程是什么?」
  「步长为主导的 policy 积累, redefine 也需要探索 policy 的积累」
答案都是"没有"。本文件把闭环补成真 RL。

## 范式: 动作 → 环境 → 奖励 → 参数更新

    环境 (Environment)  = 训练回路本身: 选一个物理区间去训练 -> 模型变化 -> 能力画像变化
    状态 s_t            = 当前能力画像 (每候选的实测 acc / 覆盖次数 / 已学标志 + 全局统计)
    动作 a_t            = 选一个候选区间 (后续可扩展到"动作=连续控制量")
    奖励 r_t            = **目标函数的增量** Δ(any-time 平均准确率)
                          ← 必须是真正要的东西, 不是"我这个区间考了多少分"
    参数更新            = θ ← θ + α_i ⊙ (r − V_θ) ∇,   α_i 由 IDBD 在线元学

    π_θ(a|s) = softmax( V_θ(s,a) / τ ),   V_θ(s,a) = θ · φ(s,a)

## 步长为主导的积累 (OaK 第②条 / Sutton IDBD)

关键: **α_i 不是超参, 而是随经验累积的状态**。

    δ_t    = r_t − V_θ(s_t, a_t)                       (TD/bandit 误差)
    w_i   += α_i · δ_t · φ_i                           (权重)
    h_i    = h_i · max(0, 1 − α_i · φ_i²) + α_i · φ_i · δ_t   (★ 累积迹)
    α_i    = α_i · exp( μ · δ_t · φ_i · h_i )          (★ 步长本身累积)

`h_i` 是"信用迹", 把历史 δ 与特征的相关性**累加**起来; `α_i` 按它调整。
**这两条是把经验持续累积进参数的通路** —— 不是 EMA 式的追踪(那会主动遗忘),
而是 Sutton 意义上的增量累积。

## ★ 实测: 步长分化存在尖锐阈值 (与 V35.2/3 同病, 两处复现)

    alpha0 = 0.01 -> alpha_std ~ 1e-6   **步长死亡** (仍是超参)
    alpha0 = 0.05 -> alpha_std ~ 1e-4   **死亡**
    alpha0 = 0.20 -> alpha_std ~ 0.3    **分化成立** (max/min ~5x)
    mu=1.0 + alpha0=0.2 -> exp 溢出, ratio 1e5 = 发散 (已加指数截断)

  根因: IDBD 的 meta-梯度量级 ∝ alpha0 (见 V35.2/3: beta_init=0.001 推不动,
  0.05 才分化)。默认取稳定窗口 mu=0.05 / alpha0=0.2。
  **注意**: 这只证明"步长分化了", 不证明"分化带来收益" —— 后者必须在
  reward 真正依赖动作的回路里测。

## 探索 policy 的积累 (redefine 存活)

    explore[c] = {n_visits, mean_return, last_seen}
    探索加成 u(c) = 1/√(1 + n_visits)   (UCB 风格, 递减但不归零)

**redefine 时不清零**: 候选池重建后, 每个新候选按**物理描述子最近的旧候选**
继承 explore / α / θ 对应分量。这样"某个物理区域值不值得探索"这条知识
**跨 pool 重建存活** —— 这正是 leo 说的"redefine 也需要探索 policy 的积累"。

## 与 RegimeProposer 的关系 (共存, 不是替换)

    RegimeProposer  手工 V + 状态更新      -> 作为**基线**
    RLProposer      可学 V_θ + IDBD 步长    -> 真 RL, 用来回答"学习有没有用"
"""
from __future__ import annotations

import numpy as np


class RLProposer:
    """动作→环境→奖励→参数更新 的提议器; 步长 (IDBD) 是主导的累积参数。

    参数
    ----
    desc : (N, D) 每个候选的物理描述子 (如 [log10 密度, log10 偶极 L, sin 磁纬])
    lam  : 兼容旧接口留空
    tau  : softmax 温度
    k    : 每次提议几个
    mu   : IDBD 元步长 (步长自身的更新率), 默认 0.01
    alpha0 : 每个权重的初始步长
    gamma  : 折扣; 1.0 = bandit (无 bootstrap), <1 = TD
    explore_w : 探索加成权重 (UCB 风格)
    seed : 随机种子
    """

    def __init__(self, desc, tau=0.5, k=2, mu=0.05, alpha0=0.2,
                 gamma=1.0, explore_w=0.5, seed=0, algo="autostep"):
        desc = np.asarray(desc, dtype=np.float64)
        self.n, self.d = desc.shape
        mu_d = desc.mean(0)
        sd_d = desc.std(0)
        sd_d[sd_d < 1e-9] = 1.0
        self.desc = desc
        self.z = (desc - mu_d) / sd_d            # 归一化描述子 (用于 redefine 继承)
        self.tau, self.k = float(tau), int(k)
        self.mu, self.gamma, self.explore_w = float(mu), float(gamma), float(explore_w)
        self.rng = np.random.default_rng(seed)

        # ── 可学参数 (policy 本身) ──
        # 特征 φ(s,a) = [1, z(a), 全局统计×z(a), 覆盖, 探索加成]
        self.fdim = 1 + self.d + self.d + 2
        self.theta = np.zeros(self.fdim)          # 价值权重 (policy 参数)
        # ★ 每个权重一个步长 + 一个信用迹 —— 这就是"步长为主导的积累"
        self.alpha = np.full(self.fdim, float(alpha0))
        self.h = np.zeros(self.fdim)              # IDBD 信用迹 (累积)
        self.baseline = 0.0
        # ★ δ 的 RMS (误差归一化)。标准 IDBD 假设 δ 是 O(1) 量级, 但
        #   "目标函数的增量" δ≈0.006 -> mu*δ*phi*h ≈ 1e-6 -> 步长几乎不动
        #   (实测 alpha_std=0.0, 与 V35.2/3 记录的 beta_std=1.6e-05 同病)。
        #   归一化后 mu 的含义与任务量级解耦。
        self.drms = 1.0
        # ★ Autostep (Mahmood/Sutton/Degris/Pilarski 2012) 的两个归一化器。
        #   动机 (论文的单位分析): IDBD 的指数项 delta*x*h 量纲是 y^2, 所以
        #   最优 mu 的量纲是 1/y^2 —— **目标方差一变, 最优 mu 就跨数量级漂移**
        #   (论文实测跨 5 个数量级)。我们的 reward = Δ(any-time) ~ 0.02, y^2 极小,
        #   所以实测必须 alpha0>=0.2 才活、mu=1.0 就炸 —— 与论文所述完全一致。
        #   Autostep 的两步归一化让 mu 免调参、且"overshoot 不可能发生"。
        self.v = np.zeros(self.fdim)        # (4) 每维 |delta*x*h| 的 running max
        self.tau_a = 1e4                   # (4) 归一化器的跟踪速率
        self.algo = algo

        # ── 环境状态 ──
        self.visits = np.zeros(self.n, dtype=np.float64)
        self.acc = np.full(self.n, np.nan)
        self.learned = np.zeros(self.n, dtype=bool)
        self.global_acc = 0.0                      # 上一轮的全域平均 (算 reward 用)
        self.any_time = 0.0

        # ── 探索记录 (redefine 存活) ──
        self.explore = np.zeros(self.n, dtype=np.float64)   # 每候选的累积收益
        self.explore_n = np.zeros(self.n, dtype=np.float64)
        self.redefine_count = 0
        self.trace = []                            # 每步记录, 供分析

    # ─────────────── 特征 / 价值 / 策略 ───────────────
    def phi(self, i):
        """φ(s,a): 描述子 + 与全局状态的调制 + 覆盖/探索加成。"""
        cov = 1.0 / (1.0 + self.visits[i])
        unc = 1.0 / np.sqrt(1.0 + self.explore_n[i])      # UCB 风格探索加成
        zz = self.z[i]
        return np.concatenate([[1.0], zz, zz * self.any_time, [cov, unc]])

    def value(self, i):
        return float(self.theta @ self.phi(i))

    def score(self, i):
        return self.value(i) + self.explore_w / np.sqrt(1.0 + self.explore_n[i])

    def probs(self):
        s = np.array([self.score(i) for i in range(self.n)])
        s = (s - s.max()) / max(1e-9, self.tau)
        e = np.exp(s)
        return e / e.sum()

    # ─────────────── 动作 ───────────────
    def act(self, k=None, greedy=False):
        """选动作 (提议 k 个候选)。greedy=True 用 argmax (评估用)。"""
        k = k or self.k
        if greedy:
            order = np.argsort([self.score(i) for i in range(self.n)])[::-1]
            idx = list(map(int, order[:k]))
        else:
            p = self.probs()
            idx = list(map(int, self.rng.choice(self.n, size=min(k, self.n),
                                                replace=False, p=p)))
        for i in idx:
            self.visits[i] += 1
        return idx

    # ─────────────── 奖励 → 参数更新 ───────────────
    def observe(self, idx, acc, any_time=None):
        """环境反馈: 记录实测 acc + 全域 any-time 准确率。

        reward 由 update() 从**目标函数的增量**算出, 不是"该区间考了多少分"。
        """
        for i in idx:
            self.acc[i] = float(acc)
            self.learned[i] = True
        if any_time is not None:
            self.any_time = float(any_time)

    def update(self, idx, r=None, gamma=None):
        """★ 参数更新: 用 reward 更新 policy。返回本步的 δ。

        r 为空时用 Δ(any-time) 作为 reward —— 这才是**真正的目标**。
        """
        gamma = self.gamma if gamma is None else gamma
        if r is None:
            r = self.any_time - self.global_acc        # 目标函数增量
        self.global_acc = self.any_time

        deltas = []
        for i in idx:
            phi = self.phi(i)
            v = float(self.theta @ phi)
            delta = float(r - v)                       # δ = r − V
            # ★ 误差归一化: 把 δ 缩放到 ~O(1), 否则 IDBD 的 meta-梯度消失
            self.drms = 0.99 * self.drms + 0.01 * delta ** 2
            delta_n = delta / max(1e-6, np.sqrt(self.drms))
            if self.algo == "autostep":
                # ── Autostep (2012 Table 1) ──
                # (4) 归一化器: |delta*x*h| 的 running max (自调节跟踪速度)
                g = np.abs(delta_n * phi * self.h)
                self.v = np.maximum(g, self.v + (1.0 / self.tau_a)
                                    * self.alpha * phi ** 2 * (g - self.v))
                # (5) 归一化后的 meta-更新: 指数项 unitless 且 |·| <= 1
                nz = self.v > 0
                self.alpha[nz] *= np.exp(
                    np.clip(self.mu * delta_n * phi[nz] * self.h[nz] / self.v[nz],
                            -2.0, 2.0))
                # (6)(7) 有效步长上界: sum(alpha*x^2) <= 1 => 过冲不可能
                M = max(float(np.sum(self.alpha * phi ** 2)), 1.0)
                self.alpha = self.alpha / M
                # 权重 + 信用迹 (与 IDBD 同)
                self.theta += self.alpha * delta_n * phi
                self.h = self.h * np.maximum(0.0, 1.0 - self.alpha * phi ** 2) \
                    + self.alpha * phi * delta_n
            else:
                # ── IDBD (Sutton 1992, 官方式 (3)) ──
                # alpha *= exp(mu * delta * x * h) —— 注意指数项**带 x**
                self.theta += self.alpha * delta_n * phi
                self.h = self.h * np.maximum(0.0, 1.0 - self.alpha * phi ** 2) \
                    + self.alpha * phi * delta_n
                _e = np.clip(self.mu * delta_n * phi * self.h, -2.0, 2.0)
                self.alpha = np.clip(self.alpha * np.exp(_e), 1e-6, 1.0)
            # (4) 探索记录累积 (redefine 存活)
            self.explore[i] += r
            self.explore_n[i] += 1.0
            deltas.append(delta)

        self.baseline += 0.05 * (r - self.baseline)     # 简单基线
        self.trace.append({
            "idx": list(map(int, idx)), "reward": round(float(r), 5),
            "delta": round(float(np.mean(deltas)) if deltas else 0.0, 5),
            "alpha_mean": round(float(self.alpha.mean()), 6),
            "alpha_std": round(float(self.alpha.std()), 6),
            "h_norm": round(float(np.linalg.norm(self.h)), 5),
        })
        return float(np.mean(deltas)) if deltas else 0.0

    # ─────────────── redefine: 探索积累跨重建存活 ───────────────
    def redefine(self, new_desc):
        """重建候选池, 但**继承**探索记录 / 步长 / 价值参数。

        leo: "redefine 也需要探索 policy 的积累" —— 若重建就清零,
        则每次 redefine 都从零开始学"哪里值得探索", 探索知识无法累积。
        继承方式: 每个新候选按**物理描述子最近**的旧候选继承。
        """
        new_desc = np.asarray(new_desc, dtype=np.float64)
        # 归一化用新池自己的统计 (与构造时一致)
        mu_d, sd_d = new_desc.mean(0), new_desc.std(0)
        sd_d[sd_d < 1e-9] = 1.0
        zn = (new_desc - mu_d) / sd_d

        keep = {"explore": [], "explore_n": [], "acc": [], "learned": []}
        for a in zn:
            j = int(np.argmin(np.linalg.norm(self.z - a, axis=1)))  # 最近旧候选
            keep["explore"].append(self.explore[j])
            keep["explore_n"].append(self.explore_n[j])
            keep["acc"].append(self.acc[j])
            keep["learned"].append(bool(self.learned[j]))

        self.n = len(new_desc)
        self.desc, self.z = new_desc, zn
        self.explore = np.array(keep["explore"])
        self.explore_n = np.array(keep["explore_n"])
        self.acc = np.array(keep["acc"])
        self.learned = np.array(keep["learned"])
        self.visits = np.maximum(1.0, self.explore_n)   # 已探过的别当全新的
        self.redefine_count += 1
        # θ / α / h 是全局参数, 天然跨 redefine 存活 —— 这正是"积累"

    # ─────────────── 诊断 ───────────────
    def stats(self):
        return {
            "algo": self.algo,
            "n_candidates": int(self.n),
            "n_learned": int(self.learned.sum()),
            "redefine_count": int(self.redefine_count),
            # ★ 步长是核心累积量: 它的分化程度 = 是否真的在学"谁该快谁该慢"
            "alpha": {
                "min": round(float(self.alpha.min()), 6),
                "max": round(float(self.alpha.max()), 6),
                "std": round(float(self.alpha.std()), 6),
                "mean": round(float(self.alpha.mean()), 6),
                "ratio_max_min": round(float(self.alpha.max()
                                             / max(1e-9, self.alpha.min())), 2),
            },
            "h_norm": round(float(np.linalg.norm(self.h)), 5),
            "theta_norm": round(float(np.linalg.norm(self.theta)), 5),
            "baseline": round(float(self.baseline), 5),
            "explore": {"nonzero": int((self.explore_n > 0).sum()),
                        "mean_n": round(float(self.explore_n.mean()), 3)},
            "any_time": round(float(self.any_time), 4),
            "delta_rms": round(float(np.sqrt(max(0.0, self.drms))), 6),
        }
