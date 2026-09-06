"""dag.py — DAG 数据结构 + d-separation（《Causal AI》第 3-4 章）

headless 的 S4 元学习世界本质是离散代数结构，因果图很小，
因此用"路径枚举 + 阻塞判断"实现 d-separation——对小图可证明正确，
避免大而全的图库带来的正确性/依赖负担。

d-separation 定义（Pearl, 2009）：
  路径 p = x … y 被 Z 阻塞当且仅当 p 存在某内部节点 v 使得
    ① 链 v₁→v→v₂ 或叉 v₁←v→v₂ 且 v ∈ Z；或
    ② 对撞 v₁→v←v₂ 且 v ∉ Z 且 v 无后代 ∈ Z。
  X ⟂ Y | Z 当且仅当所有从 X 到 Y 的路径都被 Z 阻塞。
"""
from collections import defaultdict


class Graph:
    """简单有向图。parents[n] = set(n 的父节点)，即边 parent -> n。"""

    def __init__(self, parents=None):
        self.parents = defaultdict(set)
        self.nodes = set()
        if parents:
            for c, ps in parents.items():
                self.add_node(c)
                for p in ps:
                    self.add_edge(p, c)

    def add_node(self, n):
        self.nodes.add(n)
        self.parents.setdefault(n, set())

    def add_edge(self, u, v):
        """有向边 u -> v。"""
        self.add_node(u); self.add_node(v)
        self.parents[v].add(u)

    def children(self, n):
        return {c for c, ps in self.parents.items() if n in ps}

    def neighbors_undirected(self, n):
        out = set()
        out |= self.parents.get(n, set())          # 父
        out |= self.children(n)                     # 子
        return out


def is_dag(g: Graph) -> bool:
    """有向图是否无环（拓扑排序）。"""
    indeg = {n: len(g.parents[n]) for n in g.nodes}
    q = [n for n in g.nodes if indeg[n] == 0]
    seen = 0
    while q:
        n = q.pop()
        seen += 1
        for c in g.children(n):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return seen == len(g.nodes)


def ancestors(g: Graph, Z) -> set:
    """Z 及其所有祖先（含 Z 自身）。"""
    Z = set(Z)
    out = set(); stack = list(Z)
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(g.parents.get(n, ()))
    return out


def descendants(g: Graph, n) -> set:
    """n 的后代（不含自身）。"""
    out = set(); stack = list(g.children(n))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack.extend(g.children(x))
    return out


def descendants_in(g: Graph, n, Z) -> bool:
    """n 是否有后代在 Z 中（含自身当 n∈Z）。"""
    return bool(descendants(g, n) & set(Z))


def d_separated(g: Graph, X, Y, Z, _max_depth=60) -> bool:
    """X ⟂ Y | Z ?  X, Y, Z 为节点集合。"""
    X = set(X); Y = set(Y); Z = set(Z)
    if not X or not Y:
        return True
    if X & Y:
        return False            # 同一节点不可能与自身独立（除非条件化自身）
    if X & Z or Y & Z:
        # 被条件化的节点不能再作为连接端点，直接视为阻塞
        rhs = Y - Z
        lhs = X - Z
        if not lhs or not rhs:
            return True
        # 退化为对剩余端点的判断
        X = lhs; Y = rhs
        if X & Y:
            return False

    An_Z = ancestors(g, Z)
    undir = {n: g.neighbors_undirected(n) for n in g.nodes}

    def edge_into(a, b):
        """是否存在有向边 a -> b。"""
        return b in g.parents and a in g.parents[b]

    def _collider(prev, cur, nxt):
        """cur 是否为"对撞点"（两条边都指向 cur）。"""
        return edge_into(prev, cur) and edge_into(nxt, cur)

    def _blocked(prev, cur, nxt):
        """cur 作为内部节点是否阻塞路径 prev-cur-nxt。"""
        if _collider(prev, cur, nxt):
            # 对撞：cur 或其后代 ∈ Z 则激活，否则阻塞
            return not (cur in Z or descendants_in(g, cur, Z))
        # 链 / 叉：cur ∈ Z 则阻塞
        return cur in Z

    # X 中任一节点能"到达"任一 Y 且全程未被阻塞 → 不独立
    def dfs2(node):
        """枚举简单路径；阻塞判断在离开内部节点时作用于该节点。"""
        if node in Y:
            return True
        for nb in undir[node]:
            if nb in path_set:
                continue
            if len(path) >= 2:
                prev = path[-2]
                if _blocked(prev, node, nb):
                    continue            # 此路径在 node 处被阻塞
            path.append(nb); path_set.add(nb)
            if dfs2(nb):
                path.pop(); path_set.discard(nb)
                return True
            path.pop(); path_set.discard(nb)
        return False

    for x in X:
        path = [x]; path_set = {x}
        # 起点是端点，不判阻塞
        if dfs2(x):
            return False                # 存在未阻塞路径 → 不独立
    return True                         # 全部阻塞 → 独立