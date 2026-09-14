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
                 gamma=1.0, explore_w=0.5, seed=0, algo="cidbd",
                 auto_scope="global", n_know=0):
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
        # ★ 「丰富的内部知识」(Horde/GVF 思路): 智能体**自己的预测**进入状态。
        #   手写先验 φ = [1, z(a), z(a)*any_time, cov, unc] 只有 9 维, 且每块都是
        #   人给定的结构 (见 docs/oak_alignment.md §"人为规定最少的结构")。
        #   n_know > 0 时, 评测方每步可通过 set_knowledge() 注入一段**外部知识向量**
        #   (例如: 当前能力画像 acc_matrix 的一行 + 转移模型对每个候选的预测增量),
        #   它被线性映射进 φ。知识维度越高, 可分化步长的方向就越多。
        self.n_know = int(n_know)
        self.fdim = 1 + self.d + self.d + 2 + self.n_know
        self.theta = np.zeros(self.fdim)          # 价值权重 (policy 参数)
        # ★ 每个权重一个步长 + 一个信用迹 —— 这就是"步长为主导的积累"
        self.alpha = np.full(self.fdim, float(alpha0))
        self.beta = np.log(np.maximum(self.alpha, 1e-12))
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
        self.v = np.zeros(self.fdim)        # (4) |delta*x*h| 的 running max
        self.v_g = 0.0                      # global 作用域时的标量归一化因子
        # ★ IDBD 量纲标准化器: 指数 delta*x*h 的量纲是 y^2 (论文 §3), 故须除以其
        #   自身量级才能让 mu 与任务无关。**注意这不同于 Autostep**: Autostep 除以
        #   running **max** (把指数卡死在 mu, 杀掉积累); 这里除以 running **RMS**
        #   (慢变标量, 保留跨分量比例与累积性)。
        self.grms = 1e-8
        self.beta_lo = float(np.log(1e-6))   # log(alpha) 的下界 (限制累积总量)
        self.beta_hi = float(np.log(10.0))   # 上界
        self._warm = 0                       # idbd-acc 的 warmup 计数
        # ── Continual-IDBD (按用户给的 §9/§10 设计) ──
        #   s_i    : 逐分量 |g_i| 的 EMA 尺度估计 (既归一化又保留跨分量比例)
        #   alpha_min/max : **clip 在 alpha 上**, beta 不 clip -> beta 可继续
        #                   累积, 也就能在恢复时立刻回到可塑区间
        #   E_base / kappa : 误差基线 + 变化检测阈值 (环境突变 -> 提可塑性)
        self.s = np.ones(self.fdim) * 1e-8
        self.rho = 0.01                      # s 的 EMA 速率
        self.alpha_min = 1e-5
        self.alpha_max = 10.0
        self.E_base = None
        self.kappa = 3.0                     # 误差 > kappa*基线 = 环境变了
        self.alpha_recovery = 0.05           # 恢复时的可塑性下限
        self.n_recovery = 0
        self.flip_rate = []                  # weight-flipping rate 的历史
        self.know = np.zeros(self.n_know)   # 内部知识向量 (评测方每步注入)
        self.Wk = np.zeros((self.n_know, self.n_know))  # 知识->特征 的线性映射
        if self.n_know:
            # 初始化为确定性正交映射 (无需训练, 保留信息不塌缩)
            self.Wk = np.eye(self.n_know) * (1.0 / max(1.0, np.sqrt(self.n_know)))
        self.tau_a = 1e4                   # (4) 归一化器的跟踪速率
        self.algo = algo
        # ★ Autostep 归一化因子的作用域。
        #   论文 Table 1 用**逐分量** running max: v_i = max|delta*x_i*h_i|。
        #   实测 (tests/diag_stepsize.py): 逐分量归一化把指数**卡死在 mu**
        #   (|e|max 恰 = mu), 于是跨分量的量级信息被抹掉 -> alpha max/min
        #   只有 1.10/1.31, 而 h 自身的 max/min 高达 72/55 (特征明明可分)。
        #   论文 §3 的单位分析给出正确作用域: delta*x*h 的量纲是 y^2, 故
        #   归一化因子应是一个**带 y^2 量纲的标量**, 而非逐分量向量。
        #   'global' = 用全局 running max (保留跨分量比例, 默认);
        #   'per_component' = 论文原版 (保留以作对照)。
        self.auto_scope = auto_scope

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
    def set_knowledge(self, vec):
        """注入内部知识向量 (Horde/GVF: 让智能体自己的预测成为状态)。

        调用方每步喂入, 例如:
          [acc_matrix[t] 的一行 (6 域能力画像), 之前几步的 any-time, 遗忘信号]
        或 转移模型对每个候选的预测增量。
        形状不匹配则忽略 (保证向后兼容)。
        """
        v = np.asarray(vec, dtype=float).ravel()
        if v.size != self.n_know:
            return
        self.know = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def phi(self, i):
        """φ(s,a): 描述子 + 与全局状态的调制 + 覆盖/探索加成 + **内部知识**。"""
        cov = 1.0 / (1.0 + self.visits[i])
        unc = 1.0 / np.sqrt(1.0 + self.explore_n[i])      # UCB 风格探索加成
        zz = self.z[i]
        base = np.concatenate([[1.0], zz, zz * self.any_time, [cov, unc]])
        if self.n_know:
            # 知识以 zz 为门控投影进来: 候选特定的知识 vs 全局知识
            k = self.Wk @ self.know
            return np.concatenate([base, k * (1.0 + 0.5 * zz[0])])
        return base

    def value(self, i):
        return float(self.theta @ self.phi(i))

    def score(self, i):
        return self.value(i) + self.explore_w / np.sqrt(1.0 + self.explore_n[i])

    def probs(self):
        # ★ 必须 clip: theta 学起来后分数差会拉大, 直接 exp 会让远端候选
        # **下溢成恰好 0**, 于是 rng.choice(replace=False) 报
        # "Fewer non-zero entries in p than size" 而崩。
        # clip 到 [-30, 0] 保证每个候选都有非零概率 (exp(-30) ~ 9e-14)。
        s = np.array([self.score(i) for i in range(self.n)])
        s = np.clip((s - s.max()) / max(1e-9, self.tau), -30.0, 0.0)
        e = np.exp(s)
        tot = e.sum()
        if tot <= 0 or not np.isfinite(tot):
            return np.full(self.n, 1.0 / self.n)     # 兜底: 退化成均匀
        return e / tot

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
                g = np.abs(delta_n * phi * self.h)
                if self.auto_scope == "global":
                    # ★ 标量归一化 (单位分析推导的修正): 保留跨分量比例
                    gmax = float(g.max()) if g.size else 0.0
                    self.v_g = max(gmax, self.v_g + (1.0 / self.tau_a)
                                   * float(np.sum(self.alpha * phi ** 2))
                                   * (gmax - self.v_g))
                    denom = self.v_g if self.v_g > 0 else 1.0
                    self.alpha *= np.exp(
                        np.clip(self.mu * delta_n * phi * self.h / denom, -2.0, 2.0))
                else:
                    # 论文原版: 逐分量 running max (会把 |e| 卡死在 mu)
                    self.v = np.maximum(g, self.v + (1.0 / self.tau_a)
                                        * self.alpha * phi ** 2 * (g - self.v))
                    nz = self.v > 0
                    self.alpha[nz] *= np.exp(
                        np.clip(self.mu * delta_n * phi[nz] * self.h[nz]
                                / self.v[nz], -2.0, 2.0))
                # (6)(7) 有效步长上界: sum(alpha*x^2) <= 1 => 过冲不可能
                M = max(float(np.sum(self.alpha * phi ** 2)), 1.0)
                self.alpha = self.alpha / M
                # 权重 + 信用迹 (与 IDBD 同)
                self.theta += self.alpha * delta_n * phi
                self.h = self.h * np.maximum(0.0, 1.0 - self.alpha * phi ** 2) \
                    + self.alpha * phi * delta_n
            else:
                # ── IDBD (Sutton 1992, 官方式 (3)) —— 累积型 + 三项安全化 ──
                # 官方: alpha *= exp(mu * delta * x * h)  (指数项**带 x**)
                #
                # 三处安全化, 每处都刻意**保住累积性** (与 Autostep 的取舍相反):
                #  ① 量纲标准化: 指数除以 |delta*x*h| 的 running **RMS**。
                #     Autostep 用 running max -> 指数恒被卡在 mu, n 从 36 到 1000
                #     ratio 恒为 1.01 (实测) = 完全不累积。RMS 是慢变标量,
                #     跨分量比例与时间累积都保留。
                #  ② 累积量限幅: 限制的**不是每步指数**, 而是累积后的 log(alpha)
                #     本身 (beta in [log 1e-6, log 10])。Autostep 限制每步指数,
                #     于是总位移被 exp(mu*n) 封顶; 这里允许单步大步, 只防跑飞。
                #  ③ 地板可恢复: 官方 IDBD 会把握信噪比差的分量打到下限 (实测出现
                #     1e-5 且再也回不来)。这里对撞到地板的分量给一个微小的复活推进。
                self.theta += self.alpha * delta_n * phi
                self.h = self.h * np.maximum(0.0, 1.0 - self.alpha * phi ** 2) \
                    + self.alpha * phi * delta_n
                _raw = delta_n * phi * self.h
                if self.algo == "cidbd":
                    # ── Continual-IDBD (用户给的 §9 设计) ──
                    #   g_i = delta*x_i*h_i                        (元梯度)
                    #   s_i <- (1-rho)s_i + rho*|g_i|              (逐分量 EMA 尺度)
                    #   beta_i <- beta_i + mu * g_i/(s_i+eps)      (归一化后累积)
                    #   alpha_i = clip(exp(beta_i), a_min, a_max)  (**clip 在 alpha 上**,
                    #                                               beta 不 clip)
                    #   h_i <- h_i[1-alpha_i x_i^2]^+ + alpha_i delta x_i
                    # 与 Autostep 的关键差别: 用 EMA 而非 running max。
                    # max 会把指数**卡死在 mu**(实测 |e|max ≡ mu, n 从 36 到 1000
                    # 完全不累积); EMA 是平滑估计, 保留跨分量比例与时间累积。
                    self.s = (1.0 - self.rho) * self.s + self.rho * np.abs(_raw)
                    _e = self.mu * _raw / (self.s + 1e-12)
                    _e = np.clip(_e, -2.0, 2.0)
                    self.beta = np.log(np.maximum(self.alpha, 1e-12)) + _e
                    self.alpha = np.clip(np.exp(np.clip(self.beta, -30, 30)),
                                         self.alpha_min, self.alpha_max)
                    self.flip_rate.append(float(np.mean(
                        np.sign(_e) != np.sign(getattr(self, "_e_prev", _e)))))
                    self._e_prev = _e
                elif self.algo == "idbd-raw":
                    # ★ 官方 IDBD, **不做任何 running 归一化**。
                    #   实测 (真实回路): raw 版 alpha_std=0.2357 / ratio=3.92;
                    #   而加 RMS 归一化后掉到 0.0006 / 1.01 —— 因为 delta 已被
                    #   delta_rms 归一化过一次, 再除 grad_rms 等于**双重归一化**,
                    #   指数塌到 ~0, alpha 几乎不动。分化能力被自己的"安全化"杀死。
                    #   代价: 指数无界 -> 靠下面的软壁垒与地板兜底。
                    _e = np.clip(self.mu * _raw, -2.0, 2.0)
                elif self.algo == "idbd-acc":
                    # ★ 真累积: 归一化因子在**前 warmup 步测一次就冻结**。
                    #   任何 running 归一化 (Autostep 的 max / 上面的 RMS) 都会把
                    #   指数变成平稳量 -> n 从 36 到 1000 完全不累积 (实测 ratio
                    #   恒在 ~40x 不增长)。要在时间上累积, 归一化因子必须是常数。
                    #   代价: 该常数与任务量级耦合 -> 用 warmup 自动标定一次。
                    if self._warm < 50:
                        self._warm += 1
                        self.grms = max(self.grms, float(np.mean(_raw ** 2)))
                        _den = max(1e-8, np.sqrt(self.grms))
                    else:
                        _den = max(1e-8, np.sqrt(self.grms))   # 冻结, 不再更新
                    _e = np.clip(self.mu * _raw / _den, -2.0, 2.0)
                else:
                    self.grms = 0.99 * self.grms + 0.01 * float(np.mean(_raw ** 2))
                    _e = np.clip(self.mu * _raw / max(1e-8, np.sqrt(self.grms)),
                                 -2.0, 2.0)
                # ★ 软壁垒, 不用硬 clip。
                #   硬 clip 实测的失败: 所有分量的 log(alpha) 都顶到 beta_hi
                #   -> 挤在天花板上 -> alpha max/min 塌回 1.0 (分化全丢)。
                #   软壁垒: 接近上界时只阻尼**正向**更新、接近下界时只阻尼
                #   **负向**更新。边界因此是「吸引但非吸收」的, 跨分量差异得以保留,
                #   同时仍然防跑飞。
                _la = np.log(self.alpha) + _e
                _span = self.beta_hi - self.beta_lo
                _t = np.clip((_la - self.beta_lo) / _span, 0.0, 1.0)   # 0=下界 1=上界
                _la = np.log(self.alpha) + _e * np.where(_e > 0, 1.0 - _t, _t)
                _la = np.clip(_la, self.beta_lo, self.beta_hi)         # 极值兜底
                self.alpha = np.exp(_la)
                # ③ 地板复活: 撞到下限的分量获得一个朝向可学区间的微小推进
                _dead = self.alpha <= 1e-6 * 1.01
                if _dead.any():
                    self.alpha[_dead] = np.minimum(
                        10.0, self.alpha[_dead] * np.exp(0.05 * np.sign(_e[_dead])))
            # (4) 探索记录累积 (redefine 存活)
            self.explore[i] += r
            self.explore_n[i] += 1.0
            deltas.append(delta)

        # ★ §10 reset/recovery: 误差超阈值 = 环境变了 -> **保留 w**, 只提可塑性。
        #   这是「真正的持续学习」: 不是重训丢知识, 而是提高 plasticity 重新适应。
        if self.algo == "cidbd":
            _E = float(np.mean([d ** 2 for d in deltas])) if deltas else 0.0
            if self.E_base is None:
                self.E_base = max(_E, 1e-9)
            elif _E > self.kappa * self.E_base:
                # 只抬高 alpha (beta 同步), 权重 theta 完全不动
                _new = np.maximum(self.alpha, self.alpha_recovery)
                self.alpha = _new
                self.beta = np.log(np.maximum(_new, 1e-12))
                self.n_recovery += 1
            else:
                self.E_base = 0.999 * self.E_base + 0.001 * _E
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
            "grad_rms": round(float(np.sqrt(max(0.0, self.grms))), 6),
            "alpha_dead": int((self.alpha <= 1e-5).sum()),
            "n_recovery": int(getattr(self, "n_recovery", 0)),
            "flip_rate_mean": round(float(np.mean(self.flip_rate[-50:]))
                                    if getattr(self, "flip_rate", None) else 0.0, 4),
        }
