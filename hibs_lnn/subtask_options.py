r"""subtask_options.py — **按论文顺序**实现 STOMP(arXiv 2202.03466)。

    SubTask ──> Option ──> Model ──> Planning

本文件是对本项目现有 `OptionManager` 的**修正版**。差别(逐条对齐论文):

| | 现有 `option_manager.py` | 本文件 |
|:--|:--|:--|
| 起点 | **转移图**(model-first) | **subtask**:先定 GVF 的 `(c, z)` |
| 目标 | 状态向量 `g_o` + 欧氏距离 | **stopping value `z_i(s)`**(带 stopping bonus) |
| `π_o` | 对模型贪心 argmax | 在 subtask MDP 上解出来的策略 |
| `β_o` | 逻辑回归拟合 | **stop 动作最优即停**(与论文 Fig.2 的"红色停止格"一致) |
| model | 无 | **option model**(终止分布 + 累计奖励)显式学出来 |
| 用途 | 替换 base policy 的动作 | **进 planner**,指标 = planning look-ahead 操作数 |

## 论文的两种 subtask(必须都实现, 才有正反例)

```
shortest-path (bottleneck)   c = −1                          z = 0 at subgoal
reward-respecting            c = R (与主任务相同)             z_i(s) = v(s) − w_i x_i(s) + w̄_i x_i(s)
                                                                       \____ stopping bonus ____/
```
论文 Fig.1 的结论:`shortest-path` 的 planning **比 primitive 还慢**;
`reward-respecting` **明显更快**。**这两个已知结论就是本项目的 ground truth。**
"""
from __future__ import annotations

import numpy as np

from hibs_lnn.gridworld import N_ACT, UP, RIGHT, DOWN, LEFT

STOP = N_ACT          # 停止动作的编号


# ══════════════════════════════════════════════════════════════════════
# 1. Subtask 规格:GVF 的 cumulant c 与 stopping value z
# ══════════════════════════════════════════════════════════════════════

class Subtask:
    """一个 GVF 子任务 = (c, z)。

    c(s, a)       : cumulant(论文里最短路径 subtask 是 −1, reward-respecting 是主任务奖励)
    z(s)          : stopping value(在 s 停止时拿到的终止价值)
    """

    def __init__(self, name, c_fn, z_fn):
        self.name = name
        self.c_fn = c_fn          # (s, a) -> float, 该转移的期望 cumulant
        self.z_fn = z_fn          # (s) -> float
        self.gamma = 0.99         # 由 solve(env) 覆盖
        self.policy = None        # π_o:(s) -> a 或 STOP
        self.value = None         # 该 subtask 的价值函数
        self.model = None         # option model
        self.beta = None          # β_o(s): 是否停止

    # ── 解 subtask -> 得到 option(论文 §3) ────────────────────────────
    def solve(self, env, max_sweeps=2000, tol=1e-10, resume_from=None):
        """在 subtask MDP 上做价值迭代(action 集 = 4 个 primitive + STOP)。

            目标: 最大化  E[ Σ γ^t c_t + γ^K z(S_K) ]
            Q(s, a) = c(s,a) + γ (1−β) V(s') + γ β z(s')     # β 由"停止是否最优"隐式决定
        """
        n = env.n_states
        V = np.zeros(n) if resume_from is None else resume_from.copy()
        for sweep in range(max_sweeps):
            delta = 0.0
            Vn = V.copy()
            for s in range(n):
                if env.is_goal[s]:
                    Vn[s] = 0.0        # goal 是终止状态(episode 结束)
                    delta = max(delta, abs(Vn[s] - V[s]))
                    continue
                # 停下:s 的价值就是 z(s)
                best = self.z_fn(s)
                for a in range(N_ACT):
                    c = self.c_fn(s, a)
                    if c is None:
                        continue
                    val = 0.0
                    for p, ns in env.T[s][a]:
                        # 在 ns 决定:继续(value)还是停(z)
                        val += p * (c + env.gamma * max(V[ns], self.z_fn(ns)))
                    if val > best:
                        best = val
                Vn[s] = best
                delta = max(delta, abs(Vn[s] - V[s]))
            V = Vn
            if delta < tol:
                break
        self.value = V
        # 导出 π_o 与 β_o
        pol = np.full(n, STOP, dtype=int)
        for s in range(n):
            if env.is_goal[s]:
                pol[s] = STOP
                continue
            best_a, best_v = STOP, self.z_fn(s)
            for a in range(N_ACT):
                c = self.c_fn(s, a)
                if c is None:
                    continue
                val = 0.0
                for p, ns in env.T[s][a]:
                    val += p * (c + env.gamma * max(V[ns], self.z_fn(ns)))
                if val > best_v + 1e-12:
                    best_a, best_v = a, val
            pol[s] = best_a
        self.policy = pol
        # β_o(s) = True 表示"停在 s 是最优的"
        #   判据: z(s) >= "继续走"的价值 —— 与论文 Fig.2 的红色停止格一致
        cont = np.full(n, -np.inf)
        for s in range(n):
            if env.is_goal[s]:
                continue
            best = -np.inf
            for a in range(N_ACT):
                c = self.c_fn(s, a)
                if c is None:
                    continue
                val = 0.0
                for p_, ns in env.T[s][a]:
                    val += p_ * (c + env.gamma * max(V[ns], self.z_fn(ns)))
                best = max(best, val)
            cont[s] = best
        self.beta = np.array([bool(self.z_fn(s) >= cont[s] - 1e-12) for s in range(n)],
                             dtype=bool)
        self.gamma = env.gamma
        return self


# ══════════════════════════════════════════════════════════════════════
# 2. 两种 subtask 的具体定义(论文 Fig.1 的对照)
# ══════════════════════════════════════════════════════════════════════

def shortest_path_subtask(env, subgoal_state, neg_on_neg=True):
    """论文:shortest-path / bottleneck option。`c = −1`,`z = 0` 只在子目标处。"""

    def c(s, a):
        return -1.0

    def z(s):
        return 0.0 if s == subgoal_state else -np.inf

    st = Subtask("shortest_path", c, z)
    return st


def reward_respecting_subtask(env, feature_vec, V_main, bonus_weight=1.0,
                              stop_at_goal=True):
    r"""论文式(4):reward-respecting subtask of feature attainment。

        c(s,a) = R(s,a)                       -- **与主任务相同的奖励**
        z_i(s) = [w^T x(s) - w_i*x_i(s)] + w_bar_i*x_i(s)
               = V_main(s) + (w_bar_i - w_i)*x_i(s)
                              \__ stopping bonus __/

    *** 一个必须做对的细节(本项目在 ground-truth 基准上抓到的 bug) ***
    式(4) 里 `w_i` 是 **特征 i 在主任务价值函数里的权重**, 不是 1。

    第一版把 `w_i` 硬写成 1.0 且 `w_bar_i = 1` -> stopping bonus 恒为 0
    -> `z_i(s) == V_main(s)` -> 子任务近似主任务 -> `beta_o` 处处为空
    -> option 永不停止 -> option model 退化 -> reward-respecting 与
       shortest-path 给出**完全相同**的 planning 曲线(实测 2145 / 2145)。

    这正是论文原话警告的情形:
      "The stopping values should not equal the estimated values because then the
       subtask would approximate the main task and solving it would probably add
       nothing new."

    对一热特征 x_i(s) in {0,1}, 主任务线性近似下特征 i 的权重就是
    **该特征为 1 的诸状态上的价值贡献** -> w_i = mean(V_main[feature > 0])。

    bonus_weight 就是论文的 w_bar_i:
        w_bar ~ w_i    bonus 小, option 几乎不提前停
        w_bar = 1      论文主用值
        w_bar -> 很大   退化成 shortest-path(论文 §6 实测, w_bar=100 走最短路)
    """
    on = np.asarray(feature_vec, dtype=float) > 0
    w_i = float(np.mean(np.asarray(V_main, dtype=float)[on])) if on.any() else 0.0

    def c(s, a):
        return float(env.R[s][a])

    def z(s):
        if not on[s] and not (stop_at_goal and env.is_goal[s]):
            # 特征为 0 处**不允许停**(否则"停"与"继续"同价 -> beta 退化)
            return -np.inf
        return float(V_main[s]) + (bonus_weight - w_i) * float(feature_vec[s])

    st = Subtask("reward_respecting", c, z)
    st.w_i = w_i
    st.bonus_weight = bonus_weight
    return st

# ══════════════════════════════════════════════════════════════════════
# 3. Option model(论文 §4:第三步,在 option 之后)
# ══════════════════════════════════════════════════════════════════════

def build_option_model(env, opt, max_steps=200):
    r"""由 option 的 (pi_o, beta_o) 推出**SMDP 形式的 option model**。

    返回 `M[s] = [(p, s_term, r_disc, k), ...]`,其中

        p       : 从 s 出发最终终止于 s_term 的概率
        r_disc  : 沿途中**已贴现**的累计奖励  sum_t gamma^t r_t
        k       : 平均步数

    ★★ 第二个必须做对的细节(ground-truth 基准抓到的)
    -------------------------------------------------
    第一版只返回未贴现的累计奖励, 且 planner 里写成 ``r + gamma*V(s_term)``
    —— 即只贴现**一步**, 而不是 option 实际占用的 k 步。
    -> option 被系统性高估 -> 价值迭代发散
    (实测: look-ahead 打满 300000 仍未收敛, 而 primitive 只需 1716)。

    SMDP 的正确形式是:   Q(s, o) = E[ r_disc + gamma^k * V(s_term) ]
    """
    n = env.n_states
    M = []
    for s0 in range(n):
        cur = {int(s0): (1.0, 0.0, 0.0)}     # state -> (prob, 期望贴现回报, 期望步数)
        term = {}

        def merge(d, key, p, dr, k):
            o = d.get(key)
            if o is None:
                d[key] = [p, p * dr, p * k]
            else:
                o[0] += p
                o[1] += p * dr
                o[2] += p * k

        for _step in range(max_steps):
            if not cur:
                break
            nxt = {}
            for s, (p, dr, k) in cur.items():
                if opt.beta[s] or opt.policy[s] == STOP:
                    merge(term, s, p, dr, k)
                    continue
                a = int(opt.policy[s])
                r_imm = float(opt.c_fn(s, a))
                for pt, ns in env.T[s][a]:
                    pp = p * pt
                    merge(nxt, int(ns), pp, dr + (env.gamma ** k) * r_imm, k + 1.0)
            cur = {s: (v[0], v[1] / v[0] if v[0] > 0 else 0.0,
                       v[2] / v[0] if v[0] > 0 else 0.0) for s, v in nxt.items()}

        outs = []
        for s, (psum, drw, kw) in sorted(term.items()):
            if psum <= 1e-12:
                continue
            outs.append((float(psum), int(s), float(drw / psum), float(kw / psum)))
        if not outs:
            outs = [(1.0, int(s0), 0.0, 1.0)]
        M.append(outs)
    return M



# ══════════════════════════════════════════════════════════════════════
# 4. Planning(论文 §5):指标 = look-ahead 操作数
# ══════════════════════════════════════════════════════════════════════

def planning_curve(env, option_models, v_star, s0, tol=1e-3, max_ops=400000,
                   sweeps_between=1):
    """带 option 的价值迭代;记录"每消耗 N 次 look-ahead 时 V(s0) 的误差"。

    option_models : {name: M}, M 来自 build_option_model。空 dict = 只用 primitive。
    返回 (ops_list, err_list) —— 与论文 Fig.1 的横/纵轴一致。
    """
    n = env.n_states
    V = np.zeros(n)
    ops_list, err_list = [], []
    total = 0
    # 每次都从 s0 开始扫(顺序会显著影响效率, 这里固定在自然顺序, 对三种方法一致)
    order = list(range(n))
    while total < max_ops:
        for s in order:
            if env.is_goal[s]:
                V[s] = 0.0
                continue
            best = -np.inf
            # primitive
            for a in range(N_ACT):
                val = 0.0
                for p, ns in env.T[s][a]:
                    val += p * (env.R[s][a] + env.gamma * V[ns])
                    total += 1
                if val > best:
                    best = val
            # options —— SMDP:  r_disc + gamma^k * V(s_term)
            for name, M in option_models.items():
                for ent in M[s]:
                    p, s_term, r_disc, k = ent[0], ent[1], ent[2], (ent[3] if len(ent) > 3 else 1.0)
                    val = p * (r_disc + (env.gamma ** k) * V[s_term])
                    total += 1
                    if val > best:
                        best = val
            V[s] = best
        err = abs(V[s0] - v_star)
        ops_list.append(total)
        err_list.append(err)
        if err < tol:
            break
    return ops_list, err_list


def ops_to_tolerance(env, option_models, v_star, s0, tol=0.01, max_ops=400000):
    ops, errs = planning_curve(env, option_models, v_star, s0, tol=tol, max_ops=max_ops)
    return ops[-1], errs[-1], ops, errs
