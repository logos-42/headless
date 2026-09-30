"""evidence_bank.py — **证据库**: 把 GVFBank 从「一组预测器的平均」改造成证据。

## 为什么要独立一个模块

旧的 `GVFBank.update()` 返回 `np.mean(errs)` —— 那是**简单平均**,
而用户对阶段五的硬约束是「**绝对不要把多个 GVF 简单平均**」。
更要紧的是:一个标量均值**丢掉了来源**。系统没法回答

    是哪条证据说"坏"? 哪条说"好"? 哪条**缺席**?

所以这个模块的输出**永远是引用清单**,不是分数。

## 三条不可协商的设计

1. **缺席必须显式**。旧实现把"没有证据"和"证据说没问题"混成同一个数(都落 0)
   ⇒ **未知永远伪装成好**。这里没有经验的证据被报为 `missing`,不参与平均。
2. **动力学证据只能降级**。预测误差低只说明"我预测得准",不说明"这件事有价值";
   而**新颖度会拉高误差**,所以允许升级就等于允许新颖度伪造"好"。
   ⇒ `role == "dynamics"` 的证据可以投 `bad`,**永远不能**投 `healthy`。
3. **每条证据用自己的尺度**。`err_ema / err_ref`,其中 `err_ref` 是该 GVF **自己的**
   历史误差中位数 —— 不引入跨 GVF 的全局常数(那又是另一种平均)。

## 与三态的关系

    GVFBank → EvidenceBank.support() → 引用清单 → RRSkillAgent.rr_epistemic(o)

**不新增势函数**(用户: `InternalKnowledge 首先是 evidence bank`)。
"""
from __future__ import annotations

from collections import deque

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# 语义分工 (用户指定): GVF_1 prediction / GVF_2 reward / GVF_3 transition / GVF_4 regime
#   `role` 决定这条证据能不能把状态**升级**成 good:
#     outcome  → 可以 (它直接谈结果有没有价值)
#     dynamics → **不可以** (它只谈"我预测得准不准")
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_SPEC = (
    ("prediction", "dynamics", 0.95, 0.90),
    ("reward", "outcome", 0.90, 0.80),
    ("transition", "dynamics", 0.85, 0.80),
    ("regime", "dynamics", 0.70, 0.50),
)


class GVF:
    """单个 General Value Function: 线性 TD(λ)。与旧 `knowledge.GVF` 同式。"""

    def __init__(self, name, dim, gamma=0.9, lam=0.8, lr=0.05, seed=0):
        self.name = name
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.lr = float(lr)
        self.w = np.zeros(dim)
        self.trace = np.zeros(dim)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def predict(self, phi):
        return float(self.w @ phi)

    def update(self, phi, cumulant, phi_next):
        v, v_next = self.w @ phi, self.w @ phi_next
        delta = float(cumulant) + self.gamma * v_next - v
        self.trace = self.gamma * self.lam * self.trace + phi
        self.w = self.w + self.lr * delta * self.trace
        self.steps += 1
        return abs(delta)


class GVFEvidence:
    """一条证据 —— **不是权重, 是"我凭什么这么说"**。

    每个字段都是可判定的, 不留"看起来还行"这种状态。
    """

    __slots__ = ("name", "role", "value", "err_ema", "err_ref", "n",
                 "last_step", "regime", "buf", "ema_w")

    def __init__(self, name, role, ema_w=0.05, buf_len=200):
        self.name = name
        self.role = role                 # 'dynamics' | 'outcome'
        self.value = 0.0
        self.err_ema = 0.0
        self.err_ref = 0.0               # 该 GVF **自己**的历史误差中位数
        self.n = 0
        self.last_step = -1
        self.regime = None
        self.buf = deque(maxlen=buf_len)
        self.ema_w = float(ema_w)

    def push(self, err, value, step, regime):
        self.n += 1
        self.last_step = int(step)
        self.regime = regime
        self.value = float(value)
        self.err_ema = (err if self.n == 1
                        else (1.0 - self.ema_w) * self.err_ema + self.ema_w * err)
        self.buf.append(float(err))
        # 参考尺度 = 自己的历史中位数(稳健, 不被单次尖峰带跑)
        self.err_ref = float(np.median(self.buf)) if self.buf else 0.0

    def usability(self, step, regime, n_min, stale_hl):
        """★ 缺失证据要显式缺席, 而不是按 0 平均进去。

        返回 (usable, why)。`why` 只在不可用时给出, 并进入引用清单。
        """
        if self.n < n_min:
            return False, "low_coverage"
        if step is not None and self.last_step >= 0:
            if (step - self.last_step) > stale_hl:
                return False, "stale"
        if regime is not None and self.regime is not None and self.regime != regime:
            return False, "wrong_regime"
        return True, None

    def z(self, eps=1e-9):
        """该 GVF **自己尺度**上的相对误差。"""
        return float(self.err_ema / max(self.err_ref, eps))

    def to_dict(self):
        return {"name": self.name, "role": self.role, "value": self.value,
                "err_ema": self.err_ema, "err_ref": self.err_ref,
                "z": self.z(), "n": self.n, "last_step": self.last_step,
                "regime": self.regime}


class EvidenceBank:
    """一组语义分工的 GVF + 它们的证据记录。

    ★ 没有 `aggregate()` 返回标量的接口 —— 那正是本模块要消灭的东西。
    """

    def __init__(self, dim, spec=None, seed=0, n_min=10, stale_hl=200,
                 z_bad=2.0, z_good=1.0):
        self.spec = tuple(spec) if spec else DEFAULT_SPEC
        self.dim = int(dim)
        self.n_min = int(n_min)
        self.stale_hl = int(stale_hl)
        self.z_bad = float(z_bad)
        self.z_good = float(z_good)
        self.gvfs = {}
        self.ev = {}
        # ★ 逐 step 历史 —— 互补性度量必须有"逐步序列"才能算 AUC,
        #   而**不能在事后从聚合量重建**。这里如实记下来。
        self.hist_z = {}
        self.hist_c = {}
        for i, (name, role, gamma, lam) in enumerate(self.spec):
            self.gvfs[name] = GVF(name, self.dim, gamma, lam, seed=seed + i)
            self.ev[name] = GVFEvidence(name, role)
            self.hist_z[name] = []
            self.hist_c[name] = []
        self.regime = None

    # ── 写入 ──────────────────────────────────────────────────────────
    def observe(self, phi, cumulants, phi_next, step=None, regime=None):
        """喂一组 cumulant(**按 spec 顺序**)。

        返回每条 GVF 自己的误差 —— **不返回均值**。
        """
        if len(cumulants) != len(self.spec):
            raise ValueError("cumulants 数 %d != spec 数 %d"
                             % (len(cumulants), len(self.spec)))
        if regime is not None:
            self.regime = regime
        out = {}
        for (name, _role, _g, _l), c in zip(self.spec, cumulants):
            g, e = self.gvfs[name], self.ev[name]
            err = g.update(phi, float(c), phi_next)
            e.push(err, g.predict(phi), -1 if step is None else step, regime)
            out[name] = err
            # ★ 逐 step 如实记录(互补性度量用) —— 记的是**当场**的 z,
            #   不是最后那个聚合值。
            self.hist_z[name].append(e.z())
            self.hist_c[name].append(float(c))
        return out

    def z_history(self):
        """★ 逐 step 的 z 序列与 cumulant 序列, 供互补性度量使用。

        **不能从 `evidence()` 的聚合量重建** —— 那正是"丢来源"的老毛病。
        """
        return {"z": {k: np.asarray(v, dtype=float) for k, v in self.hist_z.items()},
                "cumulant": {k: np.asarray(v, dtype=float) for k, v in self.hist_c.items()}}

    # ── 查询 ──────────────────────────────────────────────────────────
    def evidence(self, step=None, regime=None):
        """每条证据的完整记录(**e_1, e_2, e_3, e_4 原样保留**)。"""
        reg = self.regime if regime is None else regime
        out = {}
        for name, _r, _g, _l in self.spec:
            e = self.ev[name]
            ok, why = e.usability(step, reg, self.n_min, self.stale_hl)
            d = e.to_dict()
            d["usable"] = ok
            d["why"] = why
            out[name] = d
        return out

    def support(self, step=None, regime=None):
        """★★ 聚合 = **投票 + 引用**, 不是平均。

        返回的字典里**没有任何跨 GVF 的标量均值**。
        """
        reg = self.regime if regime is None else regime
        bad, healthy, missing, sources = [], [], [], {}
        usable_n = 0
        for name, role, _g, _l in self.spec:
            e = self.ev[name]
            ok, why = e.usability(step, reg, self.n_min, self.stale_hl)
            zd = e.z()
            src = {"src": name, "role": role, "z": zd, "n": e.n,
                   "err_ema": e.err_ema, "err_ref": e.err_ref}
            sources[name] = src
            if not ok:
                missing.append({"src": name, "role": role, "why": why, "n": e.n})
                continue
            usable_n += 1
            if zd >= self.z_bad:
                bad.append({"src": name, "role": role, "z": zd})
            elif zd <= self.z_good:
                # ★★ 动力学证据**只能降级**: 它可以投 bad, 但永不能投 healthy。
                #    理由: 预测误差低只说明"我预测得准", 不说明"这件事有价值";
                #    而**新颖度会拉高误差** ⇒ 允许升级就等于允许新颖度伪造"好"。
                if role == "outcome":
                    healthy.append({"src": name, "role": role, "z": zd})
                else:
                    healthy.append({"src": name, "role": role, "z": zd,
                                    "capped": "dynamics 不升级"})

        # 判定(不对称)
        if bad:
            verdict = "bad"
        elif any(h.get("role") == "outcome" and "capped" not in h for h in healthy):
            verdict = "good"
        else:
            verdict = "unknown"

        return {
            "verdict": verdict,
            "bad": bad,
            "healthy": healthy,          # 含被 cap 的动力学票(可追溯, 但不升级)
            "missing": missing,
            "usable_n": usable_n,
            "total_n": len(self.spec),
            "sources": sources,
        }

    def stats(self):
        return {"n_gvf": len(self.spec), "regime": self.regime,
                "n_min": self.n_min, "stale_hl": self.stale_hl,
                "z_bad": self.z_bad, "z_good": self.z_good,
                "evidence": {n: {"n": self.ev[n].n, "z": self.ev[n].z()}
                             for n, _r, _g, _l in self.spec}}


# ──────────────────────────────────────────────────────────────────────────
# 互补性度量 —— 回答"多个 GVF 能不能产生互补证据"
# ──────────────────────────────────────────────────────────────────────────


def _auc(scores, labels):
    """rank-based AUC(不依赖 sklearn)。labels: 1 = 正类(变化后)。

    并列用**平均秩**(Mann-Whitney U 的标准处理)。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def complementarity(err_series, shifted, z_bad=2.0, z_good=1.0):
    """多条 GVF 的证据是否**互补**。

    `err_series`: {name: 逐步误差数组}
    `shifted`:    逐步标签, 1 = 该步处于"变化之后"

    返回 AUC_i / AUC_union / complementarity / disagreement / corr_err。
    """
    names = list(err_series)
    zs = {}
    for n in names:
        a = np.asarray(err_series[n], dtype=float)
        # ★ 参考尺度只从"变化前"估计 —— 用全段会被变化本身抬高, 自己抹掉自己
        pre = a[np.asarray(shifted, dtype=int) == 0]
        ref = float(np.median(pre)) if pre.size and np.median(pre) > 0 else 1.0
        zs[n] = a / max(ref, 1e-9)

    auc_i = {n: _auc(zs[n], shifted) for n in names}
    # ★ 两种并集规则, **必须分开报** —— 它们回答的是不同问题:
    #   `any`       : 任一条超阈 ⇒ **检测**问题(加检测器不会降低 recall)
    #   `consensus` : 至少两条同时超阈 ⇒ **一致性**问题(合取, 天生比单条保守)
    #   ⚠ 只报 consensus 会得到"并集比单条差"的结论, 那是**聚合规则的产物**,
    #     不是"证据不互补"的证据。两个都算, 都报。
    flags = np.stack([(zs[n] >= z_bad).astype(int) for n in names])
    jit = 1e-6 * np.arange(flags.shape[1])          # 位置抖动, 让并列信号可排秩
    any_flag = (flags.sum(axis=0) >= 1).astype(float) + jit
    cons_flag = (flags.sum(axis=0) >= 2).astype(float) + jit
    auc_any = _auc(any_flag, shifted)
    auc_cons = _auc(cons_flag, shifted)

    valid = [v for v in auc_i.values() if v == v]
    best_i = max(valid) if valid else float("nan")
    if valid and auc_any == auc_any:
        comp = auc_any - best_i
    else:
        comp = float("nan")

    # 分歧率: **至少一条投 bad 且至少一条投 healthy** 的步数占比
    n_bad = np.stack([zs[n] >= z_bad for n in names]).sum(axis=0)
    n_ok = np.stack([zs[n] <= z_good for n in names]).sum(axis=0)
    disagreement = float(((n_bad > 0) & (n_ok > 0)).mean()) if names else float("nan")

    corr = {}
    arr = np.stack([zs[n] for n in names])
    if arr.shape[0] >= 2:
        c = np.corrcoef(arr)
        for i, a_ in enumerate(names):
            for j, b_ in enumerate(names):
                if i < j:
                    corr["%s~%s" % (a_, b_)] = float(c[i, j])

    return {"auc_i": auc_i, "auc_any": auc_any, "auc_consensus": auc_cons,
            "best_single": best_i, "complementarity": comp,
            "disagreement": disagreement, "corr_err": corr,
            "z_bad": z_bad, "z_good": z_good}
