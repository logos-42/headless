"""value_proposer.py — 价值函数提议器 (从 LMT-twister V35.19 移植, 适配连续物理区间)。

## 来源

LMT-twister 家族已把这条线验证到生产级 (skill: lmt-twister-experiment):
  V35.19 价值函数提议器 (用户理论): 简约性 + 自洽性 + 覆盖增量
  V35.20 价值函数表示迁移最优 (0.0869 > 外部课程 0.0742)
  V35.21 三 seed 确认 (0.0850±0.002), S5 内直接训练 scratch +2.05pp
  V35.22 价值函数 = 稳定器 (方差缩 3 倍)
  LM1   生产级: 提议器每步自主生成数据

原式 (V35.19):
    V(p) = λ_sim·sim(p) + λ_con·con(p) + λ_cov/(1+freq(p))
    sim(p) = 1 - max_{q∈已学} struct_sim(p,q)       # 简约性: 不可归约度
    con(p) = 1 - 矛盾度                             # 自洽性
    cov(p) = 1/(1+freq(p))                          # 覆盖增量

## 本移植

lm1 的"结构 p"是离散排列; lm4/lm5 的"结构"是**连续的物理区间**,
所以相似度换成描述子空间里的距离, 并新增一项**真实反馈** (V35 里由
CausalProposer.feedback_value 承担)。

    V(p) = λ_sim·sim(p) + λ_con·con(p) + λ_cov·cov(p) + λ_fb·fb(p)

    sim(p)  = 1 - max_{q∈已学} exp(-‖z_p - z_q‖ / σ)   # 物理上离已学区间多远
    con(p)  = 自洽性 = 该区间与已发现因果结构的吻合度  (由调用方注入)
    cov(p)  = 1/(1+freq(p))                            # 训练越少越值得学
    fb(p)   = 1 - acc(p)                               # 真实反馈: 越不准越值得学

## 关键坑 (V35.19 实测, 必须遵守)

1. **候选池必须远大于提议数**, 否则价值函数被抹平 (v1: 池 18 个被全覆盖 ->
   提议顺序无关 -> λ 扫描零效果)。本实现默认要求 len(pool) >> k·轮数。
2. **λ_cov 倒 U 型**, 平衡点 ≈ 1:1:1。默认 (1,1,1,1)。
3. **覆盖项必须真的有频率差异**, 否则退化成随机 (v2 教训: 高频重复 ≠ 均匀覆盖)。
"""
from __future__ import annotations

import numpy as np


class RegimeProposer:
    """按价值函数从候选区间池里采样"下一个学哪个物理区间"。

    参数
    ----
    desc : (N, D) 每个候选区间的物理描述子 (如 [log_dens, logL, absmaglat]).
           内部做 z-score, 保证各维等权。
    groups : (N,) 每个候选区间映射到哪个**评估域** (用于汇总/报告, 不参与打分)。
    lam : (λ_sim, λ_con, λ_cov, λ_fb)
    sigma : 描述子空间里"多远算不可归约"的尺度 (z-score 单位, 默认 0.5)
    tau : softmax 温度, 越小越贪婪
    k : 每次提议几个
    seed : 采样随机种子
    """

    def __init__(self, desc, groups=None, lam=(1.0, 1.0, 1.0, 1.0),
                 sigma=None, tau=0.5, k=2, seed=0):
        desc = np.asarray(desc, dtype=np.float64)
        self.n = len(desc)
        mu = desc.mean(0)
        sd = desc.std(0)
        sd[sd < 1e-9] = 1.0
        self.z = (desc - mu) / sd
        # ★ sigma 默认由数据决定, 不能用固定 0.5。
        #   低维描述子 (1-3 维) 下候选间距离量级很小, 固定 sigma=0.5 会让
        #   exp(-d/sigma) ≈ 0 对所有候选成立 -> sim(p) ≡ 1.0 -> 简约性项死亡
        #   (实测 signature: value 与 value-nofb 提议序列逐位相同)。
        #   取"最近邻距离的中位数"当尺度, 保证核在真实分布上有区分度。
        if sigma is None:
            d = np.linalg.norm(self.z[:, None, :] - self.z[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            sigma = float(np.median(d.min(axis=1)))
            sigma = max(sigma, 1e-3)
        self.sigma = float(sigma)
        self.groups = (np.asarray(groups) if groups is not None
                       else np.zeros(self.n, dtype=int))
        self.lam = tuple(float(x) for x in lam)
        self.tau = float(tau)
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        # 状态
        self._con = None                                  # 自洽性分数 (可注入)
        self.freq = np.zeros(self.n, dtype=np.float64)   # 被提议次数
        self.learned = []                                 # 已学区间索引
        self.acc = np.full(self.n, np.nan)                # 各区间的实测准确率
        self.trace = []                                   # 每轮提议记录

    # ---------------- 三项 + 反馈 ----------------
    def sim(self, i):
        """简约性: 1 - 与已学区间的最大相似度 (物理上离得越远越值得学)。"""
        if not self.learned:
            return 1.0
        d = np.linalg.norm(self.z[self.learned] - self.z[i], axis=1)
        return float(1.0 - np.exp(-d.min() / self.sigma))

    def cov(self, i):
        """覆盖增量: 越少被训练越值得学。"""
        return 1.0 / (1.0 + self.freq[i])

    def con(self, i):
        """自洽性 (V35.19 三项之一) —— 由调用方 set_consistency() 注入。

        **没注入时返回恒定 1.0, 该维对排序零贡献** —— 这是本轮踩过的坑:
        忘了注入 -> 自洽性项死亡 -> 价值函数退化成覆盖均匀化采样器。
        诊断: stats() 里的 con_std; 若为 0 说明这一项是死的。
        """
        return float(self._con[i]) if self._con is not None else 1.0

    def fb(self, i):
        """真实反馈: 越不准越值得学 (持续学习那一环)。"""
        a = self.acc[i]
        if not np.isfinite(a):
            return 1.0          # 没测过 = 最有价值
        return float(1.0 - a)

    def value(self, i):
        ls, lc, lv, lf = self.lam
        return (ls * self.sim(i) + lc * self.con(i)
                + lv * self.cov(i) + lf * self.fb(i))

    # ---------------- 外部注入 ----------------
    def set_consistency(self, con):
        """注入自洽性分数 (长度 N)。对 lm4: 每个区间与已发现因果边的吻合度。"""
        self._con = np.asarray(con, dtype=np.float64)

    def observe(self, idx, acc, learned=True):
        """记录一轮的结果: 该区间的实测准确率 (+ 标记为已学)。"""
        self.acc[idx] = float(acc)
        if learned and idx not in self.learned:
            self.learned.append(int(idx))

    # ---------------- 提议 ----------------
    def propose(self, k=None):
        """按 V 的 softmax 采样 k 个不重复候选 (采样而非 argmax: 保留探索)。"""
        k = k or self.k
        v = np.array([self.value(i) for i in range(self.n)])
        # 数值稳定 + 温度
        v = v - v.max()
        p = np.exp(v / max(1e-9, self.tau))
        p = p / p.sum()
        kk = min(k, self.n)
        idx = self.rng.choice(self.n, size=kk, replace=False, p=p)
        for i in idx:
            self.freq[i] += 1
        self.trace.append(list(map(int, idx)))
        return list(map(int, idx))

    def pick_random(self, k=None):
        """随机对照臂。"""
        k = k or self.k
        idx = self.rng.choice(self.n, size=min(k, self.n), replace=False)
        for i in idx:
            self.freq[i] += 1
        self.trace.append(list(map(int, idx)))
        return list(map(int, idx))

    # ---------------- 诊断 ----------------
    def stats(self):
        v = np.array([self.value(i) for i in range(self.n)])
        # ★ 各项的**区分度**诊断: 标准差为 0 = 该项是死的 (对排序零贡献)
        _sim = np.array([self.sim(i) for i in range(self.n)])
        _con = np.array([self.con(i) for i in range(self.n)])
        _cov = np.array([self.cov(i) for i in range(self.n)])
        _fb = np.array([self.fb(i) for i in range(self.n)])
        return {
            "n_candidates": int(self.n),
            "n_learned": len(self.learned),
            "n_proposed_total": int(self.freq.sum()),
            "freq_nonzero": int((self.freq > 0).sum()),
            "value_mean": float(v.mean()),
            "value_std": float(v.std()),
            "value_cv": float(v.std() / max(1e-9, abs(v.mean()))),
            "covered_frac": float((self.freq > 0).mean()),
            "sigma": float(self.sigma),
            # 各项区分度: 0 = 该项死了 (本轮踩过的坑)
            "term_std": {"sim": float(_sim.std()), "con": float(_con.std()),
                         "cov": float(_cov.std()), "fb": float(_fb.std())},
        }
