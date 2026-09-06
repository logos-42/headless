"""causal_proposer.py — 因果驱动的自主数据提议器（改进 S4ValueProposer）

原价值函数（S4ValueProposer）用启发式三分：简约 / 自洽 / 覆盖。
本模块用因果技术栈重写"提议价值"：

  value(p) = β1 · 生成元因果新颖性（引入未覆盖的"原因"原子）
           + β2 · 干预效应增量   （do(覆盖 p 的生成元) 对泛化的因果提升）
           + β3 · backdoor 因果效应（从观测数据发现的生成元→unseen 的 ATE）
           + β4 · 反事实价值     （若早先探索 p，未来 unseen 的预期增益）
           − λ · 覆盖惩罚（freq）

接口与 S4ValueProposer 兼容：__init__(learned_perms, k=...) · propose(n=None)。
"""
import numpy as np
from .scm import S4WorldSCM, _PERMS24
from .discovery import CausalDiscovery, ci_test


class CausalProposer:
    def __init__(self, learned_perms, k=2, seed=0, alpha=0.3,
                 beta=(1.0, 1.0, 1.0, 0.4), tau=1.0, world=None):
        self.learned = list(learned_perms)
        self.k = k
        self.tau = tau
        self.beta = beta
        self.seed = seed
        self.world = world or S4WorldSCM(seed=seed)
        self.discovery = CausalDiscovery(self.world, alpha=alpha, seed=seed)
        self.freq = {}
        self.learned_atoms = set()
        for p in self.learned:
            self.learned_atoms |= {i for i in range(6)
                                   if self.world.sig_bits(p)[i]}
        self.cands = [p for p in _PERMS24 if p not in self.learned]

    # ---- 组件 1：生成元因果新颖性 ----
    def causal_novelty(self, p):
        new_atoms = {i for i in range(6) if self.world.sig_bits(p)[i]
                     and i not in self.learned_atoms}
        return len(new_atoms) / 6.0

    # ---- 组件 2：干预效应增量 do() ----
    def do_effect(self, p):
        y0 = self.world.acc(p, self.learned_atoms, noise=0.0)
        y1 = self.world.do_intervene(p, self.learned_atoms)
        return max(0.0, y1 - y0)

    # ---- 组件 3：backdoor 因果效应（从观测数据反估计生成元→unseen）----
    def backdoor_causal(self):
        """用观测因子做条件独立检验，估计"覆盖某原子"对 acc 的贡献强度。"""
        perms = self.learned[:8]                    # 用已学样本做因果归因
        if len(perms) < 2:
            return 0.0
        rows = []
        for i, p in enumerate(perms):
            f = self.world.observe(p, rng=np.random.default_rng(i))
            rows.append([f[0], f[1], f[2], self.world.acc(p, self.learned_atoms)])
        X = np.array(rows)
        # 检验 gen_count(原子覆盖数) 是否与 acc 依赖
        pval, mi = ci_test(X, 0, 3, [], rng=self.discovery.rng)
        return float(mi) if pval < 0.5 else 0.0

    # ---- 组件 4：反事实价值（FNP：若 p 被提前覆盖，未来 unseen 预期增益）----
    def counterfactual_value(self, p):
        # 观测一个"较中位"未学 perm 的当前 acc，问若 p 的生成元早被覆盖会怎样
        if not self.cands:
            return 0.0
        probe = self.cands[len(self.cands) // 2]
        y_obs = self.world.acc(probe, self.learned_atoms, noise=0.0)
        y_cf, _ = self.world.counterfactual(probe, y_obs, self.learned_atoms,
                                            do_perm=p)
        return max(0.0, y_cf - y_obs)

    def value(self, p):
        b1, b2, b3, b4 = self.beta
        v = (b1 * self.causal_novelty(p)
             + b2 * self.do_effect(p)
             + b3 * self.backdoor_causal()
             + b4 * self.counterfactual_value(p))
        v /= (1 + self.freq.get(p, 0))              # 覆盖惩罚
        return v

    def _explain(self, p):
        if not hasattr(self, '_exp'):
            self._exp = {}
        self._exp[p] = {
            'novelty': round(self.causal_novelty(p), 3),
            'do_effect': round(self.do_effect(p), 3),
            'backdoor': round(self.backdoor_causal(), 3),
            'cf_value': round(self.counterfactual_value(p), 3),
        }
        return self._exp[p]

    def propose(self, n=None):
        """提议 n 个新结构（softmax 采样），返回 perm 列表（兼容原接口）。"""
        n = n or self.k
        if not self.cands:
            return []
        scores = np.array([self.value(p) for p in self.cands])
        probs = np.exp((scores - scores.max()) / max(self.tau, 1e-8))
        probs = probs / probs.sum()
        idxs = np.random.default_rng(self.seed).choice(
            len(self.cands), size=min(n, len(self.cands)), replace=False,
            p=probs)
        out = []
        for i in idxs:
            p = self.cands[i]
            self.freq[p] = self.freq.get(p, 0) + 1
            self._explain(p)
            out.append(p)
        return out