"""scm.py — 结构因果模型 SCM + do() 干预 + 反事实（《Causal AI》第 6-9 章）

把 headless 的 S4 元学习世界形式化为一个结构因果模型：

  生成元集合 G --> 循环结构 cyc --> 可观测特征 obs --> 泛化表现 acc

  · 每个任务的"原因" = 它由哪些对换生成元（6 原子）复合而成 (G)
  · "结果"        = 在未见过结构上的泛化表现 (acc)

SCM 三角色（Pearl）：观察 P(Y|X) / 干预 do(x) / 反事实 Y_x(u)
"""
import numpy as np
from itertools import permutations


class SCM:
    """通用结构因果模型（numpy 版，无外部依赖）。"""

    def __init__(self, variables, exogenous, equations, parents):
        self.variables = list(variables)
        self.exogenous = list(exogenous)
        self.equations = dict(equations)
        self.parents = {v: set(ps) for v, ps in parents.items()}
        self.all_nodes = set(self.variables) | set(self.exogenous)

    def _mutate(self, do):
        eq = dict(self.equations)
        for var in do:
            if var in self.variables:
                eq[var] = lambda a, _v=var, _x=do[var]: _x
        return eq

    def sample(self, do=None, rng=None, n=1):
        rng = rng if rng is not None else np.random.default_rng()
        eq = self._mutate(do or {})
        samples = []
        for _ in range(n):
            a = {}
            for v in self.exogenous:
                a[v] = rng.normal(0.0, 1.0)
            for v in self.variables:
                a[v] = float(eq[v](a))
            if do:
                for v, x in do.items():
                    a[v] = float(x)
            samples.append(a)
        return samples


# ----------------------------------------------------------------------
# S4 世界的因子（生成元签名 + 循环类型）
# ----------------------------------------------------------------------
_PERMS24 = list(permutations(range(4)))


def _cycle_type(perm):
    seen = set(); lens = []
    for i in range(4):
        if i in seen:
            continue
        c = []; j = i
        while j not in seen:
            seen.add(j); c.append(j); j = perm[j]
        if len(c) > 1:
            lens.append(len(c))
    lens.sort()
    if not lens:
        return 0          # ident
    if lens == [2]:
        return 1          # swap
    if lens == [3]:
        return 2          # 3-cycle
    if lens == [4]:
        return 3          # 4-cycle
    if lens == [2, 2]:
        return 4          # double swap
    return 0


def _generate_perms():
    """perm -> (sig_bits[6], cyc_type)。sig_bits: 6 位对换生成元掩码。"""
    TRANS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    IDENT = (0, 1, 2, 3)
    from collections import deque
    out = {}
    for perm in _PERMS24:
        q = deque([(IDENT, [])]); seen = {IDENT}; found = None
        while q and found is None:
            cur, w = q.popleft()
            if cur == perm:
                found = w; break
            if len(w) >= 6:
                continue
            for gi, g in enumerate(TRANS):
                a, b = g; p = list(cur); p[a], p[b] = p[b], p[a]
                nxt = tuple(p)
                if nxt not in seen:
                    seen.add(nxt); q.append((nxt, w + [gi]))
        sig = [0] * 6
        if found:
            for gi in found:
                sig[gi] = 1
        out[perm] = (sig, _cycle_type(perm))
    return out


class S4WorldSCM:
    """S4 元学习世界的结构因果模型。

    变量：
      G    生成元集合（6 位掩码）----- 因
      cyc  循环类型（0..4）
      obs  可观测训练特征
      acc  泛化表现（0..1）---------- 果

    因果结构：G -> cyc；G -> obs；cyc -> obs；obs,cyc -> acc
    """

    PARENTS = {
        'cyc': {'G'},
        'obs': {'G', 'cyc', 'U_obs'},
        'acc': {'obs', 'cyc', 'U_acc'},
    }

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.factor = {}
        for perm in _PERMS24:
            sig, cyc = _GEN[perm]
            self.factor[perm] = {'sig': list(sig), 'cyc': cyc}

    def sig_bits(self, perm):
        return self.factor[perm]['sig']

    def cyc(self, perm):
        return self.factor[perm]['cyc']

    def observe(self, perm, rng=None):
        rng = rng or self.rng
        sig = np.array(self.sig_bits(perm), dtype=float)
        u1 = rng.normal(0, 0.3)
        u2 = rng.normal(0, 0.25)
        return np.array([sig.sum() + u1 * 0.2,
                         (6 - sig.sum()) + u2 * 0.2,
                         self.cyc(perm) * 0.6])

    def acc(self, perm, learned_atoms, seed=None, noise=0.08):
        """泛化真值 acc（0..1）。已学生成元越多 acc 越高；四/双对换更硬。"""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        sig = np.array(self.sig_bits(perm), dtype=float)
        known = sum(1 for i in range(6) if sig[i] and i in learned_atoms)
        total = max(1.0, sig.sum())
        base = 0.35 + 0.6 * (known / total)
        cyc = self.cyc(perm)
        if cyc in (3, 4):
            base *= 0.90
        if cyc == 4:
            base *= 0.88
        return float(np.clip(base + rng.normal(0, noise), 0.02, 0.98))

    def do_intervene(self, perm, learned_atoms):
        """do()：干预——假设该 perm 的生成元已被覆盖，测量其对 acc 的因果效应。"""
        new_atoms = set(learned_atoms)
        new_atoms |= {i for i in range(6) if self.sig_bits(perm)[i]}
        return self.acc(perm, new_atoms, noise=0.0)

    def counterfactual(self, observed_perm, observed_acc, learned_atoms,
                       do_perm=None):
        """反事实（Pearl 三步：Abduction → Action → Prediction）。"""
        # 1) Abduction：由观测偏差反推外生扰动 U
        n_known = sum(1 for i in range(6)
                      if self.sig_bits(observed_perm)[i] and i in learned_atoms)
        sig_total = max(1.0, sum(self.sig_bits(observed_perm)))
        base = 0.35 + 0.6 * (n_known / sig_total)
        cyc = self.cyc(observed_perm)
        if cyc in (3, 4):
            base *= 0.90
        if cyc == 4:
            base *= 0.88
        u_acc = observed_acc - base

        # 2) Action：干预——将 do_perm 的生成元也计入已学
        new_atoms = set(learned_atoms)
        if do_perm is not None:
            new_atoms |= {i for i in range(6) if self.sig_bits(do_perm)[i]}
        n_known2 = sum(1 for i in range(6)
                       if self.sig_bits(observed_perm)[i] and i in new_atoms)
        base2 = 0.35 + 0.6 * (n_known2 / max(1.0, sig_total))
        c = self.cyc(observed_perm)
        if c in (3, 4):
            base2 *= 0.90
        if c == 4:
            base2 *= 0.88

        # 3) Prediction：同一个人（同 U），不同干预 → 新结果
        y_cf = np.clip(base2 + u_acc, 0.02, 0.98)
        return float(y_cf), {'u_acc': round(float(u_acc), 4),
                             'n_known_before': n_known,
                             'n_known_after': n_known2}
_GEN = _generate_perms()