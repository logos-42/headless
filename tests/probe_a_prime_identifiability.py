#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_a_prime_identifiability.py — 证明 A′ 环境真的可辨识(判据 ⑨′a/⑨′b)。

## 要证明的两件事

    构型 I  (⑨′a): Novelty ↑  而  Dynamics error **不变**
    构型 II (⑨′b): Dynamics error ↑  而  Novelty **不变**      <- 真正关键

原 `KeyDoorMDP` 两个都做不到(熵恒 0 ⇒ 一切都退化成 novelty)。
本探针在 E0/E1/E2 上直接量:

    ① 每个 (s,a) 后继分布的熵          —— "不确定性"是否有非平凡读数
    ② 同一批 (s,a) 在 shift 前后的熵   —— "变"是否发生(**而计数不变**)
    ③ 访问计数的分布                   —— "新"是否被改变

## 为什么用真值转移而不是 agent 的在线表

⑨′b 要问的是"**这个环境**有没有能力把变与新分开", 这是环境的性质, 不是
agent 学得好不好的问题。所以用环境自身的 `true_transition` 穷举。
agent 侧仍然只用在线计数表(不碰真值) —— 这条边界必须守住。

★ 但"计数不变"必须用**agent 真的走过什么**来量, 不能凭空断言。
  所以第 ③ 项用一次真实 rollout 的在线表, 而不是穷举。
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/Users/apple/Downloads/headless")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hibs_lnn.skill_mdp import KeyDoorMDP              # noqa: E402
from hibs_lnn.stochastic_keydoor import (              # noqa: E402
    N_ACTIONS, FlipKeyDoor, NonStationaryKeyDoor, StochasticKeyDoor,
    true_transition)
from benchmark_three_state import FROZEN               # noqa: E402
from hibs_lnn.rr_agent import RRSkillAgent             # noqa: E402

NL = np.log(32)


def ent(cnt):
    n = sum(cnt.values())
    if n < 2:
        return None                      # 无证据
    p = np.array(list(cnt.values()), float)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / NL)


def env_entropies(env, states, acts):
    """环境真值: 每个 (s,a) 的后继分布归一化熵。"""
    out = {}
    for s in states:
        for a in acts:
            e = ent(true_transition(env, s, a))
            if e is not None:
                out[(s, a)] = e
    return out


def rollout_counts(env, episodes, seed=0, dyn="A1"):
    """跑真 agent, 取**在线计数表**(不是真值) —— 用来量"新不新"。"""
    ag = RRSkillAgent(env, mode="rr", seed=seed, rr_v_main="learned",
                      **dict(FROZEN, rr_three_state=True, rr_dyn=dyn))
    for _ in range(episodes):
        ag.run_episode()
    return ag._rr_T, ag


print("=" * 100)
print("① 后继分布的熵: E0 应当恒 0, E1/E2 应当非平凡")
print("=" * 100)
states = list(range(32))
acts = list(range(N_ACTIONS))
print(f"  {'环境':>26}{'非平凡 (s,a) 数':>18}{'平均熵':>10}{'最大熵':>10}")
res = {}
for nm, env in (("E0 确定性", KeyDoorMDP()),
                ("E1 平稳随机 slip=0.10", StochasticKeyDoor(slip=0.10, seed=1)),
                ("E2 非平稳 slip0=0.10", NonStationaryKeyDoor(seed=1))):
    e = env_entropies(env, states, acts)
    v = np.array(list(e.values()), float)
    res[nm] = (e, env)
    print(f"  {nm:>26}{len(v):>18}{v.mean():>10.4f}{v.max():>10.4f}")

print()
print("=" * 100)
print("② ★ 构型 II (⑨′b): 同一批 (s,a), shift 前后熵变化 —— 而计数不变")
print("=" * 100)
env2 = NonStationaryKeyDoor(slip0=0.10, slip1=0.60, shift_at=10 ** 9, seed=1)
before = env_entropies(env2, states, acts)
env2b = NonStationaryKeyDoor(slip0=0.10, slip1=0.60, shift_at=0, seed=1)
env2b.t_total = 10 ** 9                       # 强制进入 shift 后的分支
after = env_entropies(env2b, states, acts)
common = sorted(set(before) & set(after))
d = np.array([after[k] - before[k] for k in common], float)
print(f"  共同 (s,a) 数 = {len(common)}")
print(f"  shift 前 平均熵 = {np.mean([before[k] for k in common]):.4f}")
print(f"  shift 后 平均熵 = {np.mean([after[k] for k in common]):.4f}")
print(f"  变化: 均值 {d.mean():+.4f}   上升的比例 {np.mean(d > 0):.3f}   "
      f"最小 {d.min():+.4f}   最大 {d.max():+.4f}")
print(f"  ⇒ 熵{'确实上升' if d.mean() > 0 else '未上升'} "
      f"({'⑨′b 的环境前提成立' if d.mean() > 0 else '不成立'})")

print()
class _RecRRSkillAgent(RRSkillAgent):
    """把每一步的 dyn 读数**按步号**记下来(探针专用, 不改生产代码)。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dlog = []

    def _rr_dval(self, s, a, s2):
        v = super()._rr_dval(s, a, s2)
        self.dlog.append((int(getattr(self.mdp, "t_total", len(self.dlog))),
                          float(v)))
        return v


def window_means(channel, env, episodes=80, shift_at=400, seed=7, win=100):
    """跑一次 rollout, 返回 shift **紧邻前后**两段等长窗口里该通道的均值。

    ★ 窗口必须**紧邻 shift 且等长**。第一版取 "前 400 步" vs "400 步之后",
      测的是**热身期 vs 稳态**(agent 还在学习, 两边都在降), 不是
      "变前 vs 变后"。那个设计下所有通道的前后比值都 < 1, 没有任何分辨力。
    ★ 也不用"新增 (s,a) 率": 任何对 P(s'|s,a) 的改动都会改变 agent 的去向
      ⇒ 也就改变了它访问哪些 (s,a)。"集合增长率"必然把"变"读成"新" ——
      翻转版 0.957、slip↑ 版 0.933, 两个都高, 指标本身没有分辨力。
      正确的量是**每一步的当场读数**, 用更新前的计数算。
    """
    ag = _RecRRSkillAgent(env, mode="rr", seed=seed, rr_v_main="learned",
                          **dict(FROZEN, rr_three_state=True, rr_dyn=channel))
    for _ in range(episodes):
        ag.run_episode()
    lo, hi = shift_at - win, shift_at
    pre = [v for t, v in ag.dlog if lo <= t < hi]
    post = [v for t, v in ag.dlog if hi <= t < hi + win]
    f = lambda w: (float(np.mean(w)) if w else float("nan"))       # noqa: E731
    return f(pre), f(post), len(pre), len(post)


print("=" * 100)
print("③ ★ 构型 II (⑨′b) 真正的判据: shift 前后**紧邻等长**窗口, N/A1/A2 并排")
print("      N = 新颖度控制量, A1 = 转移预测误差, A2 = 转移不确定性。")
print("      预期(翻转环境): N 平 / A2 平 / **A1 跳**  <-- A1 与 A2 必须分歧")
print("=" * 100)
SH, WIN = 400, 100
print(f"  {'环境':>18} {'通道':>4} {f'[{SH-WIN},{SH})':>13} {f'[{SH},{SH+WIN})':>13} "
      f"{'变化':>9} {'比值':>7}")
for nm, mk in (("E1 平稳随机", lambda: StochasticKeyDoor(slip=0.10, seed=7)),
               ("E2 翻转(干净)", lambda: FlipKeyDoor(
                   p_pre=0.10, p_post=0.90, shift_at=SH, seed=7)),
               ("E2c slip↑(反例)", lambda: NonStationaryKeyDoor(
                   slip0=0.10, slip1=0.60, shift_at=SH, seed=7))):
    for ch in ("N", "A1", "A2"):
        a1, a2, n1, n2 = window_means(ch, mk(), 80, SH, 7, WIN)
        ratio = (a2 / a1) if a1 and a1 > 1e-12 else float("inf")
        print(f"  {nm:>18} {ch:>4} {a1:>13.4f} {a2:>13.4f} "
              f"{a2 - a1:>+9.4f} {ratio:>7.2f}   (n={n1}/{n2})")
print("""
  判读: 可辨识环境要求
          N  : 比值 ≈ 1.0     (没有变得更"新")
          A2 : 比值 ≈ 1.0     (熵对 {p,1-p} 与 {1-p,p} 同值 —— 翻转不改变它)
          A1 : 比值 >> 1.0    (学到的 p̂ 现在错了)
        若 N 或 A2 也跳, 说明该 shift 构造仍混淆 novelty/不确定性。
        若 A1 与 A2 读数相同, 说明它们其实是同一个量 —— A 类仍需证伪。""")

print()
print("=" * 100)
print("④ 构型 I (⑨′a): E1 的'不确定性'能否在不制造新状态的前提下上升")
print("=" * 100)
for slip in (0.0, 0.1, 0.3, 0.6):
    e = env_entropies(StochasticKeyDoor(slip=slip, seed=3), states, acts)
    v = np.array(list(e.values()), float)
    print(f"  E1 slip={slip:.1f}: 平均熵 = {v.mean():.4f}   非平凡 (s,a) = {len(v)}")
print("""
  解读: slip 从 0.0 升到 0.6 时平均熵单调上升 —— 这叫**不确定性**。
  但它**同时**改变了 agent 能到达的状态集合(所以也带了 novelty)。
  ⇒ 单看 E1 分不开两者, 必须靠 E2 的 shift 段(P 形状变、状态集合不变)。
""")
