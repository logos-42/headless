"""discovery.py — 因果发现（条件独立检验 + PC 式骨架）（第 4 章 / 因果发现章）

从观测数据（各任务的因子 + 真实 acc）出发，推断结构变量之间的依赖，
从而"发现"哪些生成元/结构因子是泛化表现的决定因素——这是提议器决策的依据。

实现：
  · ci_test     —— 标准条件互信息 I(X;Y|Z) 的频率估计 + Bootstrap 置零检验
  · pc_skeleton —— PC 算法骨架：逐级条件独立检验删边
  · CausalDiscovery —— 高层封装
"""
import numpy as np
from itertools import combinations


def _discretize(col, bins=3):
    """连续列分箱为离散标签（用于互信息估计）。"""
    if np.all(col == col[0]):
        return np.zeros(len(col), dtype=int)
    q = np.quantile(col, np.linspace(0, 1, bins + 1)[1:-1])
    return np.clip(np.digitize(col, q), 0, bins - 1)


def _cmi(dx, dy, dz):
    """标准条件互信息 I(X;Y|Z) = Σ p(xyz) log[p(xy|z)/(p(x|z)p(y|z)]。"""
    n = len(dx)
    if n < 2:
        return 0.0
    from collections import Counter
    c_xyz = Counter(); c_xz = Counter(); c_yz = Counter()
    for x, y, z in zip(dx, dy, dz):
        c_xyz[(x, y, z)] += 1
        c_xz[(x, z)] += 1
        c_yz[(y, z)] += 1
    mi = 0.0
    for (x, y, z), c in c_xyz.items():
        c_x, c_y = c_xz[(x, z)], c_yz[(y, z)]
        if c_x == 0 or c_y == 0:
            continue
        term = (c / n) * np.log2((c * n) / (c_x * c_y))
        mi += term
    return float(np.clip(mi, 0.0, None))


def ci_test(data, x, y, z, rng=None, n_boot=100, alpha=0.5):
    """X⊥Y|Z 条件独立检验（互信息 + Bootstrap）。

    返回 (p_value, mi)。p 越低越倾向"相关/依赖"；p>alpha 判独立。
    检验用标准条件互信息；null 分布用打乱 X（破坏 X⊥(Y,Z)）。
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if n < 2:
        return 1.0, 0.0
    dx = _discretize(data[:, x]); dy = _discretize(data[:, y])
    if z:
        # 条件多列 → 每列单独分箱 → 联合编码成一维整数 bin code
        dz = np.zeros(n, dtype=int)
        for i in z:
            dz = dz * 2 + _discretize(data[:, i], bins=2)
    else:
        dz = np.zeros(n, dtype=int)
    mi = _cmi(dx, dy, dz)
    rng = rng or np.random.default_rng(0)
    null = []; dxx = dx.copy()
    for _ in range(n_boot):
        dxx[:] = rng.permutation(dx)
        null.append(_cmi(dxx, dy, dz))
    null = np.array(null)
    p = float(((null >= mi).sum() + 1) / (len(null) + 1))
    return p, mi


def pc_skeleton(X_data, variables, alpha=0.3, max_order=2, rng=None):
    """PC 骨架：若 X、Y 在给定阶条件集下条件独立则删边（返回无向骨架位图）。"""
    n, p = X_data.shape
    adj = {(i, j): True for i in range(p) for j in range(i + 1, p)}
    for order in range(max_order + 1):
        for (i, j) in list(adj):
            if not adj[(i, j)]:
                continue
            nb_j = [k for k in range(p) if k not in (i, j)
                    and adj.get((min(j, k), max(j, k)), False)]
            if len(nb_j) < order:
                continue
            for cond in combinations(nb_j, order):
                pval, _ = ci_test(X_data, i, j, list(cond), rng=rng)
                if pval > alpha:
                    adj[(i, j)] = False
                    break
    return adj


class CausalDiscovery:
    """高层封装：观测因子矩阵 + acc → 依赖发现。"""

    def __init__(self, world, alpha=0.4, max_order=2, seed=0):
        self.world = world
        self.alpha = alpha
        self.max_order = max_order
        self.rng = np.random.default_rng(seed)

    def optimize_adjustment(self, perms, learned_atoms):
        """基于观测因子构建数据，做 PC 骨架发现。

        返回 (vars_, adj, X)。acc 用世界真值（模拟观测），
        learned_atoms 使"已覆盖"结构差异化可被反估计。
        """
        rows = []
        for i, p in enumerate(perms):
            feat = self.world.observe(p, rng=np.random.default_rng(i))
            a = self.world.acc(p, learned_atoms)
            rows.append([feat[0], feat[1], feat[2], a])
        X = np.array(rows)
        vars_ = ['gen_count', 'new_atoms', 'cyc', 'acc']
        adj = pc_skeleton(X, vars_, alpha=self.alpha,
                          max_order=self.max_order, rng=self.rng)
        return vars_, adj, X

    def discover_graph(self, rows):
        """将变量间邻接转为因子 DAG 骨架并返回。"""
        return pc_skeleton(np.array(rows),
                           ['gen_count', 'new_atoms', 'cyc', 'acc'],
                           alpha=self.alpha, max_order=self.max_order,
                           rng=self.rng)