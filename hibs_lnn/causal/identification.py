"""identification.py — 后门调整 / do-calculus 识别 / 因果效应估计（第 10-11 章）

目标：从观测数据推出干预分布 P(Y | do(X))。

  后门准则（Pearl）：若变量集 Z 满足
    ① Z 中无 X 的后代，且
    ② Z 阻断所有 X→Y 的后门路径（X 与某节点之间指向 X 的路径），
  则  P(Y|do(X=x)) = Σ_z P(Y|X=x,z)P(z)。

用观测数据（本世界中各 perm 的真实 acc 与生成元因子）做离散化后门调整，
估计"覆盖生成元 X 对后续 unseen 提升 Y 的因果效应"，用于提议器决策。
"""
import numpy as np
from .dag import Graph, d_separated, is_dag


def _backdoor_paths_blocked(graph: Graph, X, Y, Z):
    """Z 是否阻断所有 X 与 Y 之间指向 X 的后门路径。

    X/Y 为单节点。后门路径：连接 X 与 Y、且第一条边指向 X 的无向路径。
    判定：在互图（删除 X 的出边后的无向图）中，X 与 Y 被 Z 分隔。
    """
    # 构造剔除 X→ 出边后的图（保留指向 X 的边）
    parents = {n: set(graph.parents[n]) for n in graph.nodes}
    for ch in graph.children(X):
        parents[ch].discard(X)      # 删除 X 的出边
    g2 = Graph(parents)
    # 从 X 到 Y 的所有路径（用 d_separated 判断是否被 Z 阻断其余路径）
    # 后门准则要求所有后门路径都被 Z 阻断，等价于在该互图上 X 与 Y 被 Z 分隔
    return d_separated(g2, {X}, {Y}, set(Z))


def backdoor_adjustment_set(graph: Graph, X, Y, nodes):
    """在候选节点中找一个满足后门准则的调整集 Z（返回 None 表示不存在/直接用空集）。"""
    cand = sorted(nodes - {X, Y})
    if _backdoor_paths_blocked(graph, X, Y, set()):
        return set()
    # 简单搜索：尝试小规模调整集（组合，节点少则穷举）
    from itertools import combinations
    for k in range(len(cand) + 1):
        for combo in combinations(cand, k):
            Z = set(combo)
            if not _desc_intersect(graph, X, Z) and \
                    _backdoor_paths_blocked(graph, X, Y, Z):
                return Z
    return None


def _desc_intersect(graph, X, Z):
    from .dag import descendants
    return bool(descendants(graph, X) & set(Z)) or X in Z


def causal_effect(data, X, Y, Z=None):
    """离散化后门调整估计 E[Y | do(X=x)]（比较 do=1 vs do=0 的平均因果效应）。

    data    : list of (X_val(0/1), Y_val(连续), dict(调整变量->离散值))
    Z       : 需调整的变量名（数据各行带该变量值）
    返回    : (ATE, effect_by_Z)  平均处理效应 = E[Y|do(X=1)] - E[Y|do(X=0)]
    """
    rows = data
    znames = list(Z) if Z else []
    # 先验 P(z)
    from collections import defaultdict
    z_count = defaultdict(float); z_sum1 = defaultdict(float); z_sum0 = defaultdict(float)
    for x, y, zv in rows:
        key = tuple(zv.get(n, 0) for n in znames)
        z_count[key] += 1
        if x == 1:
            z_sum1[key] += y
        else:
            z_sum0[key] += y
    total = float(len(rows))
    ate = 0.0
    for key, cnt in z_count.items():
        pz = cnt / total
        m1 = z_sum1[key] / z_count[key] if z_count[key] else 0.0  # 简化：用同层均值
        m0 = z_sum0[key] / z_count[key] if z_count[key] else 0.0
        # 同层内若缺 X 侧，用该层观测直接近似
        ate += pz * (m1 - m0)
    return ate


def g_formula(graph: Graph, X, Y, Z):
    """g-公式的图诊断：检查调整集 Z 是否满足后门/前门条件并返回可识别性。"""
    if not is_dag(graph):
        return False, "图非 DAG，不可识别"
    Zs = set(Z)
    no_desc = not (_desc_intersect(graph, X, Zs))
    blocked = _backdoor_paths_blocked(graph, X, Y, Zs)
    if no_desc and blocked:
        return True, "backdoor 满足 → 可识别（P(Y|do(X))=Σ_z P(Y|X,z)P(z)）"
    return False, "后门不满足（未实现前门/do-calculus 自动搜索）"