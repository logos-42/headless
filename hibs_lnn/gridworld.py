"""gridworld.py — 论文里的两房间 / 四房间 gridworld(arXiv 2202.03466 Fig.1 / Fig.6)。

## 为什么要有这个文件

本项目的 lm4/lm5 基准**没有可用作时间抽象的结构**(实测不对称性 `A ≡ 0`,
唯一的动力学是对称纯干扰)。在那种环境上,任何类别的 option 都不可能有收益 ——
所以「Options 有害」这个结论**测的是基准的性质**。

论文用的 gridworld 是**有明确正反例**的:同一个环境里
  · bottleneck / shortest-path option  的 planning **比 primitive 还慢**(Fig.1)
  · reward-respecting option            的 planning **明显更快**(Fig.1)
这两个已知结论就是本项目的 ground truth —— 用来验证「我的 option + planning 管线写对了没有」。

## 环境规格(逐条对齐论文)

| 项 | 设定 |
|:--|:--|
| 动作 | 4 个:上 / 右 / 下 / 左 |
| 撞墙 | 位置不变 |
| 目标 | 到达 goal 得 **+1**,episode 结束 |
| 灰色区 | **进入**灰色区的转移每步 **−1** |
| 其他 | 0 |
| γ | **0.99** |
| 两房间最优 | `v*(s0) = 0.99^17 ≈ 0.843`(绕开负奖励区) |
| 四房间动力学 | 期望方向 w.p. **2/3**,其余三个方向各 **1/9** |

**注意论文原文用的是 "transitions ending in the gray region"** ——
即**落点**在灰色区才扣分,不是路径穿过的格子。本实现按**落点**判。
"""
from __future__ import annotations

import numpy as np

# 动作编号(与论文的四方向一致)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
N_ACT = 4
ACT_NAME = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}

# ── 两房间 gridworld(Fig.1 插图) ─────────────────────────────────────────
#   '#' 墙   '.' 地板   'S' 起点   'G' 目标   '-' 灰色区(落点扣分)   '+' 走廊(bottleneck)
#
#   布局要点(必须同时满足, 否则复现不了论文):
#     ① 存在一个**单格走廊**连接两个房间 -> 它是天然 bottleneck
#     ② 从 S 到走廊的**最短路径**穿过灰色区
#     ③ 存在一条**绕开灰色区**的更长路径
# ★ 第 4、5 行结尾的两个 '#' 不是笔误, 但它**修不掉原本的问题** —— 见下方说明。
#
# 实测(新版 check_two_room): 旧几何的 col 1 是通的 -> 有一条自由竖井绕过整个
# 灰色区 -> 最短路**根本不碰灰色**(neg_on_shortest=0), 而且"绕行"与最短路等长。
# 也就是说: 论文 Fig.1 的三条性质**一条都没满足**, 而因为自检函数本身有 bug
# (hall 落在灰色格里 -> "绕开灰色区到达 hall" 构造上不可能), 它一直报的是 ✗ 却没人看。
#
# 现在按论文重设: 灰色区**正对**单格通道, 使得
#     穿过灰色 = 3 步(最短)
#     绕开灰色 = 7 步(更长)
TWO_ROOM = [
    "#########",
    "#...S...#",
    "#..---..#",
    "#.......#",
    "####+####",
    "#.......#",
    "#......G#",
    "#########",
]

# ── 四房间 gridworld(Fig.6): 4 个房间 + 3 个单格走廊 ──────────────────
#
# ★ 这里曾经是**静默坏掉**的: 旧布局行长不齐(12/13/14), 且有一行含**空格**。
#   行不齐 -> GridWorld 里 `.ljust(..., '#')` 静默补成墙, 几何被改;
#   空格   -> 只有 '#' 被跳过, 空格于是变成一格**幽灵地板**(可走但看不出来)。
#   两者都不会报错, 所以四房间那一支一直在跑一个**非预期的环境**, 复现自然失败。
#   现在加 validate_layout() 在构造时**硬报错**, 这类 bug 不可能再静默。
#
# 布局设计(必须同时满足, 否则复现不了论文 Fig.6):
#     ① 2x2 四个房间, 3 个单格走廊连成环 -> 走廊是天然 bottleneck
#     ② 从 S 到走廊的最短路**穿过灰色区**(负奖励)
#     ③ 存在一条**绕开灰色区**的更长路径
#
# 实测几何(见 check_four_room):
#     最短路 S->G  = 18 步, 穿过灰色区 2 格
#     绕行路线     = 20 步, 完全避开灰色区
FOUR_ROOM = [
    "#############",
    "#S..####....#",
    "#..--.+.....#",
    "#.....#.....#",
    "#.....#.....#",
    "###+#########",
    "#.....#.....#",
    "#.....#.....#",
    "#.....+.....#",
    "#.....#....G#",
    "#############",
]


_ALLOWED = set("#.-SG+")


def validate_layout(layout, name="layout"):
    """★ 硬校验布局。**报错而非默默容错** —— 见 FOUR_ROOM 上方的注释。

    这条检查是补上一个真实事故: 旧四房间布局行长不齐 + 混入空格,
    被 `.ljust(..., '#')` 和"只跳过 '#'"两个静默行为一起吃掉了,
    结果整个四房间分支跑的是一个没人设计过的环境。

    返回 (H, W)。不合法直接 raise ValueError。
    """
    if not layout:
        raise ValueError("%s: 布局为空" % name)
    widths = {len(r) for r in layout}
    if len(widths) != 1:
        detail = ", ".join("%d:\"%s\"" % (i, r) for i, r in enumerate(layout))
        raise ValueError(
            "%s: 行长不一致 (宽度集合 %s) —— 补齐会变成墙, 必须手工对齐。各行: %s"
            % (name, sorted(widths), detail))
    W = widths.pop()
    for i, r in enumerate(layout):
        bad = sorted(set(r) - _ALLOWED)
        if bad:
            raise ValueError(
                "%s: 第 %d 行含非法字符 %r。只允许 %s —— "
                "注意**空格不是地板**('#' 才是唯一的墙, 其它字符都会被当成可走格)。"
                % (name, i, bad, sorted(_ALLOWED)))
        if r[0] != "#" or r[-1] != "#":
            raise ValueError("%s: 第 %d 行两端必须是墙 '#'" % (name, i))
    for j in range(W):
        if layout[0][j] != "#" or layout[-1][j] != "#":
            raise ValueError("%s: 第 %d 列两端必须是墙 '#'" % (name, j))
    flat = "".join(layout)
    if flat.count("S") != 1:
        raise ValueError("%s: 必须恰好有一个 'S' (实际 %d)" % (name, flat.count("S")))
    if flat.count("G") < 1:
        raise ValueError("%s: 至少需要一个 'G'" % name)
    return len(layout), W


class GridWorld:
    """确定性 / 随机性可切的 gridworld(论文规格)。"""

    def __init__(self, layout, gamma=0.99, stochastic=False, slip=2.0 / 3.0):
        self.layout = list(layout)
        self.gamma = float(gamma)
        self.stochastic = bool(stochastic)
        self.slip = float(slip)

        # ★ 先硬校验, 再构造。旧代码这里是 `.ljust(W, '#')` 静默补齐 ——
        #   它会**悄悄把短行补成墙**, 于是布局错了几何也照跑不误。
        #   validate_layout 会把行长不齐/非法字符/边界非墙变成**构造期异常**。
        self.H, self.W = validate_layout(layout, name=getattr(self, "name", "layout"))
        self.grid = list(layout)          # 已保证等宽, 不再 ljust

        self.cells = []            # (row, col)
        self.index = {}
        self.is_neg = []           # 落点是否灰色区(扣分)
        self.is_goal = []
        self.start = None
        for r in range(self.H):
            for c in range(self.W):
                ch = self.grid[r][c]
                if ch == "#":
                    continue
                i = len(self.cells)
                self.cells.append((r, c))
                self.index[(r, c)] = i
                self.is_neg.append(ch == "-")
                self.is_goal.append(ch == "G")
                if ch == "S":
                    self.start = i
        self.n_states = len(self.cells)
        self.is_neg = np.array(self.is_neg, dtype=bool)
        self.is_goal = np.array(self.is_goal, dtype=bool)
        assert self.start is not None, "布局里必须有 'S'"
        assert self.is_goal.any(), "布局里必须有 'G'"

        self._build_transitions()

    # ── 动力学 ──────────────────────────────────────────────────────────
    def _move(self, r, c, a):
        if a == UP:
            r -= 1
        elif a == DOWN:
            r += 1
        elif a == LEFT:
            c -= 1
        else:
            c += 1
        if (r, c) in self.index:
            return self.index[(r, c)]
        return None                      # 撞墙

    def _build_transitions(self):
        """T[s][a] = [(prob, next_state), ...];  R[s][a] = 期望即时奖励。"""
        self.T: list = [[[] for _ in range(N_ACT)] for _ in range(self.n_states)]
        self.R = np.zeros((self.n_states, N_ACT))
        for s, (r, c) in enumerate(self.cells):
            # ★ goal 是**吸收态**(论文: "A reward of +1 is received on reaching the
            #   goal state, **which ends the episode**")。
            #   不做这一步的话, agent 会反复落回 goal 反复拿 +1 -> 价值爆炸
            #   (实测: 子任务里 hall 处的"继续价值"算成 98.01, 而真值 <= 1)。
            if self.is_goal[s]:
                for a in range(N_ACT):
                    self.T[s][a] = [(1.0, int(s))]
                    self.R[s][a] = 0.0
                continue
            for a in range(N_ACT):
                if self.stochastic:
                    # 期望方向 2/3, 其余三个方向各 1/9(论文规格)
                    others = [x for x in range(N_ACT) if x != a]
                    cand = [(a, self.slip)] + [(x, (1.0 - self.slip) / 3.0) for x in others]
                else:
                    cand = [(a, 1.0)]
                # 合并相同落点
                acc = {}
                for act, p in cand:
                    ns = self._move(r, c, act)
                    if ns is None:
                        ns = s          # 撞墙 -> 留在原地
                    acc[ns] = acc.get(ns, 0.0) + p
                outs = sorted(acc.items())
                self.T[s][a] = [(float(p), int(ns)) for ns, p in outs]
                # 落点在灰色区 -> -1;落点是 goal -> +1
                rew = 0.0
                for p, ns in self.T[s][a]:
                    if self.is_goal[ns]:
                        rew += p * 1.0
                    elif self.is_neg[ns]:
                        rew += p * -1.0
                self.R[s][a] = rew

    def reset(self):
        return self.start

    def step(self, s, a):
        outs = self.T[s][a]
        if len(outs) == 1:
            ns = outs[0][1]
        else:
            p = np.array([x[0] for x in outs])
            p = p / p.sum()
            ns = outs[int(np.random.choice(len(outs), p=p))][1]
        done = bool(self.is_goal[ns])
        return ns, float(self.R[s][a]), done


# ── 布局性质自检(必须过, 否则复现不了论文) ─────────────────────────────

def find_bottlenecks(env):
    """介数中心性高(且度小)的格子 = 论文说的 bottleneck。

    实现方式:对确定性图做全对最短路径(无权 BFS),数每格出现在多少条最短路里。
    """
    import collections
    n = env.n_states
    # 确定性后继(取 T 里概率最大的那个, 用于结构性分析)
    succ = [[max(env.T[s][a], key=lambda x: x[0])[1] for a in range(N_ACT)] for s in range(n)]
    between = np.zeros(n)
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = collections.deque([src])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for a in range(N_ACT):
                v = succ[u][a]
                if v != u and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for tgt in range(n):
            if tgt == src or dist[tgt] < 0:
                continue
            # 回溯一条最短路(标记经过的格)
            u = tgt
            guard = 0
            while u != src and guard < 4 * n:
                between[u] += 1
                nxt = None
                for a in range(N_ACT):
                    v = succ[u][a]
                    if v != u and dist[v] == dist[u] - 1:
                        nxt = v
                        break
                if nxt is None:
                    break
                u = nxt
                guard += 1
    return between


def check_two_room(env, verbose=True):
    """验证两房间布局确实具备论文要求的三条性质。"""
    import collections
    n = env.n_states
    succ = [[max(env.T[s][a], key=lambda x: x[0])[1] for a in range(N_ACT)] for s in range(n)]

    def bfs(src, avoid_neg=False):
        dist = {src: 0}
        parent = {src: None}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            if env.is_goal[u]:
                continue
            for a in range(N_ACT):
                v = succ[u][a]
                if v == u or v in dist:
                    continue
                if avoid_neg and env.is_neg[v]:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
        return dist, parent

    bt = find_bottlenecks(env)
    # bottleneck = 非起非终、**且不是灰色格**、介数最高
    # ★ 必须排除灰色格: 否则 hall 常常正好落在灰色区里, 而下面又要求
    #   "存在绕开灰色区到达 hall 的路径" —— 那是**构造上不可能**的,
    #   于是这条自检永远报 ✗, 成了没人看的红灯(实测: 原来的 hall=(3,6) 就是 '-' 格)。
    cand = [(bt[s], s) for s in range(n)
            if s != env.start and not env.is_goal[s] and not env.is_neg[s]]
    cand.sort(reverse=True)
    hall = cand[0][1]

    d_free, _ = bfs(env.start)
    d_avoid, _ = bfs(env.start, avoid_neg=True)

    reach_hall = hall in d_free
    # 最短路到走廊经过的灰色区格数
    _, par = bfs(env.start)
    path, u = [], hall
    while u is not None:
        path.append(u)
        u = par.get(u)
    path.reverse()
    neg_on_shortest = sum(int(env.is_neg[s]) for s in path)
    has_detour = hall in d_avoid and d_avoid.get(hall, 10 ** 9) > d_free.get(hall, 0)

    if verbose:
        r0, c0 = env.cells[env.start]
        rh, ch = env.cells[hall]
        print("  [两房间自检]")
        print("    起点 (%d,%d) -> 走廊 (%d,%d)  [介数 %.0f]" % (r0, c0, rh, ch, bt[hall]))
        print("    到达走廊: %s   最短路长度 %s" % (reach_hall, d_free.get(hall)))
        print("    最短路上的灰色格数: %d   (论文要求 > 0, 否则没有'绕路'可言)"
              % neg_on_shortest)
        print("    存在绕开灰色区的路径: %s  (长度 %s vs 最短 %s)"
              % (has_detour, d_avoid.get(hall), d_free.get(hall)))
    ok = reach_hall and neg_on_shortest > 0 and has_detour
    if verbose:
        print("    三条性质全满足: %s" % ("✓" if ok else "✗ 布局需要调整"))
    return ok, hall, dict(d_free=d_free, d_avoid=d_avoid,
                          neg_on_shortest=neg_on_shortest, between=bt)


def check_four_room(env, verbose=True):
    """验证四房间布局确实具备论文 Fig.6 要求的性质。**四房间从来没有过这个检查。**

    与 check_two_room 的三条性质对应, 但四房间要额外验:
        ① 把走廊堵上后, 恰好分裂成 **4 个连通分量**(真的是四房间, 不是一团)
        ② 走廊格是 bottleneck(介数最高)
        ③ 从 S 到走廊的**最短路穿过灰色区**
        ④ 存在**绕开灰色区**的更长路径

    返回 (ok, halls, info)。
    """
    import collections
    n = env.n_states
    succ = [[max(env.T[s][a], key=lambda x: x[0])[1] for a in range(N_ACT)] for s in range(n)]

    def bfs(src, avoid_neg=False, block=None):
        dist, q = {src: 0}, collections.deque([src])
        parent = {src: None}
        while q:
            u = q.popleft()
            if env.is_goal[u]:
                continue
            for a in range(N_ACT):
                v = succ[u][a]
                if v == u or v in dist or v == block:
                    continue
                if avoid_neg and env.is_neg[v]:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
        return dist, parent

    bt = find_bottlenecks(env)
    # ★ 走廊格 = 字符 '+' 的格子(显式标记, 不靠介数猜)
    halls = [s for s, (r, c) in enumerate(env.cells) if env.grid[r][c] == "+"]

    # ① 堵上全部走廊 -> 数连通分量
    blocked = set(halls)
    seen, comps = set(), []
    for s in range(n):
        if s in seen or s in blocked:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for a in range(N_ACT):
                v = succ[u][a]
                if v == u or v in seen or v in blocked:
                    continue
                seen.add(v)
                stack.append(v)
        comps.append(comp)

    # ③④ 对每个走廊格测最短路 vs 绕行
    rows = []
    for h in halls:
        d_free, par = bfs(env.start)
        d_av, _ = bfs(env.start, avoid_neg=True)
        path, u = [], h
        while u is not None:
            path.append(u)
            u = par.get(u)
        path.reverse()
        neg_short = sum(int(env.is_neg[s]) for s in path)
        rows.append(dict(hall=h, d_free=d_free.get(h), d_avoid=d_av.get(h),
                         neg_on_shortest=neg_short,
                         detour_longer=(h in d_av and h in d_free and d_av[h] > d_free[h])))

    p_ok = abs(bt[halls[0]] - max(bt)) < 1e-6 if halls else False
    ok = (len(comps) == 4 and halls
          and all(r["d_free"] is not None for r in rows)
          and any(r["neg_on_shortest"] > 0 for r in rows)
          and any(r["detour_longer"] for r in rows))

    if verbose:
        print("  [四房间自检]")
        print("    状态数 %d  走廊格 %d 个: %s"
              % (n, len(halls), [env.cells[h] for h in halls]))
        print("    堵上走廊后的连通分量数: %d   (论文要求 4)" % len(comps))
        if halls:
            print("    走廊是介数最高点: %s (介数 %.0f / 全局最高 %.0f)"
                  % (p_ok, bt[halls[0]], max(bt)))
        for r in rows:
            rh, ch = env.cells[r["hall"]]
            print("    走廊 (%d,%d): 最短路 %s 步(其中灰色 %d 格)  绕行 %s 步  绕行更长: %s"
                  % (rh, ch, r["d_free"], r["neg_on_shortest"], r["d_avoid"], r["detour_longer"]))
        print("    四条性质全满足: %s" % ("✓" if ok else "✗ 布局需要调整"))
    return ok, halls, dict(n_comps=len(comps), rows=rows, between=bt)


def value_iteration(env, model=None, n_sweeps=1, V0=None, options=None, order=None):
    """带 option 的价值迭代(论文 §5 的 planning)。

    model : [(p, ns, r)] 形式的 dict; 默认用真模型(论文 Fig.1 假设模型准确)
            格式:  model[s][u] = [(p, ns, r), ...];  u 索引 0..N_ACT-1 是 primitive,
                   N_ACT.. 是 option(一跳到底, 返回终止状态与累计奖励)
    n_sweeps : 扫多少遍
    返回 (V, n_lookahead) —— n_lookahead = 消耗的 look-ahead 操作数(论文的横轴)
    """
    n = env.n_states
    V = np.zeros(n) if V0 is None else V0.copy()
    n_look = 0
    if model is None:
        model = {s: {a: [(p, ns, env.R[s][a]) for p, ns in env.T[s][a]] for a in range(N_ACT)}
                 for s in range(n)}
    for _ in range(n_sweeps):
        idx = range(n) if order is None else order
        for s in idx:
            if env.is_goal[s]:
                V[s] = 0.0
                continue
            best = -np.inf
            for u, outs in model[s].items():
                val = 0.0
                for p, ns, r in outs:
                    val += p * (r + env.gamma * V[ns])
                    n_look += 1            # 每次 look-ahead 计数
                best = max(best, val)
            V[s] = best
    return V, n_look


if __name__ == "__main__":
    print("=" * 70)
    print("两房间 gridworld(确定性, 论文 Fig.1)")
    print("=" * 70)
    env = GridWorld(TWO_ROOM, gamma=0.99, stochastic=False)
    print("  状态数 %d, 动作数 %d" % (env.n_states, N_ACT))
    ok, hall, info = check_two_room(env)

    V, n = value_iteration(env, n_sweeps=200)
    print("\n  value iteration 200 遍 (primitive only): V(S) = %.4f" % V[env.start])
    print("  论文报告的最优值 v*(s0) = 0.99^17 = %.4f" % (0.99 ** 17))
    print("  look-ahead 操作数 = %d" % n)

    print("\n" + "=" * 70)
    print("四房间 gridworld(随机, slip=2/3, 论文 Fig.6)")
    print("=" * 70)
    env4 = GridWorld(FOUR_ROOM, gamma=0.99, stochastic=True)
    print("  状态数 %d, 动作数 %d" % (env4.n_states, N_ACT))
    ok4, halls4, info4 = check_four_room(env4)
    # ★ 自检不过就让进程**非零退出** —— 否则坏几何又会被静默当成"实验做完了"
    if not (ok and ok4):
        print("\n✗ 布局自检未通过, 拒绝继续(退出码 1)")
        raise SystemExit(1)
    V4, n4 = value_iteration(env4, n_sweeps=200)
    print("\n  value iteration 200 遍 (primitive only): V(S) = %.4f" % V4[env4.start])
    print("  look-ahead 操作数 = %d" % n4)
