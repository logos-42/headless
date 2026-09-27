# -*- coding: utf-8 -*-
"""A′ 的三个环境:把 `novelty` 与 `dynamics change` 分开。

## 为什么需要新环境

原 `KeyDoorMDP.step()` 是**完全确定性**的(无任何随机源)。实测:所有访问过
`n>=2` 的 `(s,a)`,后继分布的最大熵**逐位为 0.000000**。于是

    转移不确定性 H(T|S,A) ≡ 0
    转移预测误差 1−p̂(s'|s,a) 的唯一非零来源是 `n<2` 分支
    稳定区域 ≈ 同样的计数

三个语义不同的量坍缩成**同一个"我见过这个 (s,a) 几次"**。
⇒ 那个环境对 A 类**不可辨识**,不是 A 类不行。

## 三个环境(A′-2)

    E0  KeyDoorMDP              确定性        —— 负对照, 保留不删
    E1  StochasticKeyDoor       平稳随机      —— 不确定性可分离, 无 shift
    E2  NonStationaryKeyDoor    非平稳(slip↑) —— ★ **可用**: 满足构型 II
        FlipKeyDoor             非平稳(翻转)  —— ❌ 失败构造, 保留为反例

★ **命名与直觉相反, 以实测为准**(探针③, 逐步紧邻等长窗口):

    E2 翻转:    N 3.62  A1 3.39  A2 2.75   <- N 跳得比 A1 还凶, 失败
    E2 slip↑:   N 0.88  A1 3.10  A2 1.25   <- N 平, A1 跳, A1/A2 分歧 ✓

我原以为"翻转(支撑集不变)"最干净、"slip↑"是反例。**测量给出相反答案**:
决定 novelty 的不是转移的支撑集, 而是最终的**去向分布**; 翻转恰好把 agent
送到它几乎没去过的伙伴目标。而均匀随机 slip↑ 是在**已熟悉的地盘内**
增加不可预测性 ⇒ 访问分布不动, `p̂` 变成错的。

## 设计要点:机制只写一份

每个环境只实现 **`_perturb(a) -> 实际执行的动作`**, `step()` 与诊断用的
`true_transition()` **共用**它。

第一版把机制在 `step()` 和 `true_transition()` 里各写了一遍, 后果是:
`true_transition` 里那条 `if rand < _slip_now(): a = randint(N)` 是**均匀随机**
版本, 对 `FlipKeyDoor` 完全不对(它只翻给伙伴动作)。于是"真值"量的是另一套
动力学, 而探针偏偏靠真值判断环境可不可辨识 —— **诊断工具自己错了,
却会给出一个看起来正常的结论**。两份实现只要漂移一次就足够。
"""
import numpy as np

from hibs_lnn.skill_mdp import KeyDoorMDP

__all__ = ["StochasticKeyDoor", "FlipKeyDoor", "NonStationaryKeyDoor",
           "N_ACTIONS", "true_transition"]

N_ACTIONS = 4                       # LEFT, RIGHT, GRAB, OPEN
_NEVER = 1 << 62


class StochasticKeyDoor(KeyDoorMDP):
    """E1 平稳随机:`P(s'|s,a)` 全程不变。

    以 `(1-slip)` 执行意图动作, 否则执行**均匀随机**动作。
    作用 = 制造"不确定性高但**没有** regime shift"。
    """

    def __init__(self, slip=0.10, seed=0, **kw):
        super().__init__(**kw)
        self.slip = float(slip)
        self.rng = np.random.RandomState(int(seed))
        self.t_total = 0            # 全局步数(跨 episode) —— 子类用它定位 shift
        self.shifted = False

    # ── 机制(唯一实现; step 与 true_transition 共用)──────────────
    def _p_now(self):
        return self.slip

    def _perturb(self, a):
        """返回**实际执行**的动作。支撑集由子类的实现决定。"""
        if self.rng.rand() < self._p_now():
            return int(self.rng.randint(N_ACTIONS))
        return int(a)

    def step(self, a):
        if not self.shifted and self.t_total >= getattr(self, "shift_at", _NEVER):
            self.shifted = True
        self.t_total += 1
        return KeyDoorMDP.step(self, self._perturb(a))


class FlipKeyDoor(StochasticKeyDoor):
    """❌ 失败构造(保留为反例):"翻转分布、不扩张支撑集"**仍然混淆 novelty**。

    ## 我当初的推理(错的)

    想让 `P(s'|s,a)` 的**支撑集恒定**、只翻转权重:

        P(s'|s,a) = (1-p)·1[s'=T(s,a)] + p·1[s'=T(s,c(a))]
        shift 前 p = 0.10           shift 后 p = 0.90

    `c(a)` 是固定的伙伴动作(LEFT<->RIGHT, GRAB<->OPEN)。推理是:
    支撑集 `{T(s,a), T(s,c(a))}` 两个 p 下都一样 ⇒ 到达的状态集合不变
    ⇒ novelty 不变; 而模型学到的 `p̂` 现在是错的 ⇒ A1 跳; 熵
    `H({p,1−p})` 两 regime 同值 ⇒ A2 不跳。看起来是完美的 构型 II。

    ## 实测(逐步、紧邻等长窗口, 探针③)

        E2 翻转:  N 0.0597 -> 0.2160 (3.62)   A1 0.1908 -> 0.6476 (3.39)
                  A2 0.1125 -> 0.3093 (2.75)
        E2c slip↑: N 0.0997 -> 0.0881 (0.88)   A1 0.1738 -> 0.5382 (3.10)
                  A2 0.1326 -> 0.1656 (1.25)

    **翻转版失败**: N 跳 3.62 倍, 比 A1 还凶。

    ## 为什么推理错了(值得记)

    > **决定 novelty 的不是转移的支撑集, 而是最终的去向分布。**

    支撑集不变只保证"这些状态**可能**被到达"; 但翻转把 agent 在 90% 的
    步子上送到**伙伴目标** —— 那是它极少去的地方 ⇒ 它真的闯进了新领地。
    支撑集是"可能性", novelty 量的是"实际去过几次", 两者不是一回事。

    ## 教训

    "不扩张支撑集"看起来是把 novelty 摘掉的自然做法, 但 agent 的行为是
    **状态依赖**的 —— 改了动力学就改了轨迹, 改了轨迹就改了访问分布。
    要抑制 novelty, 得直接让**访问分布**不变, 而不是让转移的支撑集不变。
    `NonStationaryKeyDoor`(均匀随机 slip↑)无意中做到了这一点:
    它在**已熟悉的地盘内**增加不可预测性。
    """

    COMPANION = (1, 0, 3, 2)            # LEFT<->RIGHT, GRAB<->OPEN

    def __init__(self, p_pre=0.10, p_post=0.90, shift_at=3000, seed=0, **kw):
        super().__init__(slip=p_pre, seed=seed, **kw)
        self.p_pre = float(p_pre)
        self.p_post = float(p_post)
        self.shift_at = int(shift_at)

    def _p_now(self):
        return self.p_pre if self.t_total < self.shift_at else self.p_post

    def _perturb(self, a):
        if self.rng.rand() < self._p_now():
            return int(self.COMPANION[int(a)])      # 只翻给伙伴动作
        return int(a)


class NonStationaryKeyDoor(FlipKeyDoor):
    """★ E2(**可用版**):均匀随机 slip 在 `shift_at` 处抬升。

        E1: slip 恒定 0.10            E2: 0.10 -> 0.60 @shift_at

    它最初是被当成"反例"写下来的(我原以为它会混淆 novelty, 因为
    "slip 变大 ⇒ agent 走得不一样"). 实测**正好相反**, 它是唯一同时满足
    构型 II 的构造:

        N   0.0997 -> 0.0881  (0.88)   ← 平坦: 没有更"新"
        A1  0.1738 -> 0.5382  (3.10)   ← 跳: 学到的 p̂ 现在错了
        A2  0.1326 -> 0.1656  (1.25)   ← 基本平 ⇒ **A1 与 A2 分歧**

    为什么它反而干净: 均匀随机 slip 在 **已熟悉的地盘内** 增加不可预测性
    (agent 早就把大部分 (s,a) 走过很多遍了), 所以访问分布几乎不动;
    而 `p̂` 的**形状**变了 ⇒ 预测误差上去了。

    于是这条环境上 "变" 与 "新" 可辨识, 并且 A1/A2 读数**明显分离** ——
    这是 A 类候选第一次在同一环境上表现出不同的量。
    """

    def __init__(self, slip0=0.10, slip1=0.60, shift_at=3000, seed=0, **kw):
        super().__init__(p_pre=slip0, p_post=slip1, shift_at=shift_at,
                         seed=seed, **kw)

    def _perturb(self, a):
        if self.rng.rand() < self._p_now():
            return int(self.rng.randint(N_ACTIONS))     # 均匀随机 = 不选边
        return int(a)


def true_transition(env, s, a, trials=400):
    """把 `(s,a)` 的真实后继分布**数出来**(穷举环境, 非模型)。

    只用于验证"这个 benchmark 真的可辨识"(判据 ⑨′a/⑨′b)。
    agent 侧一律只用在线计数表, **不碰这个函数**。

    ★ 设置状态必须走 `decode()`:`s = pos*4 + has_key*2 + door_open`,
      `has_key`/`door_open` 也影响转移。手写 `env.s = s` 是错的
      (那个属性不存在, 而且漏掉两个标志位)。
    ★ 扰动走 `env._perturb()` —— 与 `step()` **同一份实现**。E0 没有该方法,
      退化成恒等(确定性), 正好是我们要的负对照。
    """
    pert = getattr(env, "_perturb", None)
    cnt = {}
    for _ in range(int(trials)):
        env.pos, env.has_key, env.door_open = env.decode(s)
        env.t = 0
        aa = int(a) if pert is None else int(pert(a))
        s2, _, _ = KeyDoorMDP.step(env, aa)
        cnt[int(s2)] = cnt.get(int(s2), 0) + 1
    return cnt
