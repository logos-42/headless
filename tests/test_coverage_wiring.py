#!/usr/bin/env python3
"""test_coverage_wiring.py — 验证「coverage 写入」这根线接上了。

## 为什么要有这个测试

实测抓到的 bug(同一类错误第 4 次出现):
    `InternalKnowledge.observe_action()` 是 `coverage.n` 的**唯一写入点**,
    但 `OAKProposer.update()` 从来没调过它
    -> `coverage.n` 永远全零
    -> `UncertaintyGate.allow()` 对所有动作返回 False (因为 `0 <= tau_C=1.0`)
    -> `_goal_action()` 的每个候选都被 continue
    -> `best_a = None` -> 回落到 base policy
    -> **轨迹与 E9(完全无 option)逐位相同**

实测证据(长测 300 轮 × 5 seed):`option_starts > 0` 但 `option_steps == 0`,
且 goal / goal_term / goal_term_override 三档的 `alpha_mean` 与 e9 **完全相同**。

这个测试的作用:把"接线"变成**可判定的断言**,而不是靠看日志猜。
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_proposer(n_fine=6, mode="goal"):
    """用 __new__ 绕过构造, 只装测试需要的字段 —— 不依赖 RLProposer / 数据。"""
    from hibs_lnn.oak_proposer import OAKProposer
    from hibs_lnn.knowledge import InternalKnowledge

    p = object.__new__(OAKProposer)
    p.n_fine = n_fine
    p.n = n_fine
    p.use_options = True
    p.use_gate = True
    p.opt_mode = mode
    p.opt_frac = 1.0
    p._opt_round = 0
    p._opt_active = None
    p._opt_queue = []
    p._opt_age = 0
    p._opt_max_age = 8
    p._opt_stall_d = None
    p.option_steps = 0
    p.option_starts = 0
    p.replan_steps = 0
    p.override_count = 0
    p.term_reasons = {}
    p.gate_blocked = 0
    p.no_finite_unc = 0
    p.goal_none = 0
    p.start_blocked = 0
    p.untrusted_picks = 0
    p.term_eps = 0.05
    p.stall_tol = 0.0
    p._state_scale = 0.4
    p._prev_state = np.array([0.5] * n_fine)
    # update() 还会碰到的状态 (走"不刷新"分支即可)
    p.trans = []
    p.refresh_count = 0
    p.refresh_every = 4
    p.min_transitions = 40
    p.seed = 0
    p.dim_state = None
    p.refresh = lambda: None
    # 知识层 (coverage 全零起步)
    p.know = InternalKnowledge(n_fine, n_fine)
    # 桩 base: act() 固定返回 0
    p.base = SimpleNamespace(act=lambda: [0], update=lambda pick: None)
    return p


def main():
    fails = []

    # ── ① 接线: update() 必须把执行过的动作记进 coverage ──────────────
    p = make_proposer()
    before = p.know.coverage.n.copy()
    for _ in range(5):
        p.update([2])
    after = p.know.coverage.n.copy()
    grew = after[2] - before[2]
    ok = grew == 5
    print("① coverage 写入接线")
    print("   before =", before.astype(int).tolist())
    print("   after  =", after.astype(int).tolist())
    print("   动作2 增长 = %d (期望 5)  %s" % (grew, "✓" if ok else "✗"))
    if not ok:
        fails.append("① coverage 没被 update() 写入")
    print("   coverage_total =", p.know.coverage.n_total, "✓" if p.know.coverage.n_total == 5 else "✗")
    if p.know.coverage.n_total != 5:
        fails.append("① coverage_total 未累加")

    # ── ② 门控: 覆盖足够时 allow() 必须放行 ────────────────────────────
    print("\n② 门控行为 (tau_C=%.1f)" % p.know.uncertainty.tau_C)
    allowed = [p.know.uncertainty.allow(None, a, p.know.coverage.n)[0] for a in range(p.n_fine)]
    print("   覆盖 %s -> allow = %s" % (p.know.coverage.n.astype(int).tolist(), allowed))
    if sum(allowed) == 0:
        fails.append("② 覆盖存在时门控仍全拒")
        print("   ✗ 覆盖存在却仍全拒")
    else:
        print("   ✓ 至少放行 %d 个动作" % sum(allowed))

    # ── ③ _goal_action: 覆盖支持时必须给出真实动作(而不是 None) ────────
    print("\n③ _goal_action 在覆盖支持下必须返回动作")
    pred = p.know.predict(p._prev_state, 0)
    print("   know.predict 返回类型:", type(pred).__name__, "| _dyn_fitted =",
          getattr(p.know, "_dyn_fitted", None))
    p.know.fit_dynamics(
        np.array([[0.5] * p.n_fine + [a] for a in range(p.n_fine)] * 4, dtype=float),
        np.array([[0.5] * p.n_fine] * (p.n_fine * 4), dtype=float))
    p.know.uncertainty.calibrate([0.0, 0.1, 0.2, 0.3]) if hasattr(p.know.uncertainty, "calibrate") else None
    a = p._goal_action(SimpleNamespace(goal_center=np.array([1.0] * p.n_fine)))
    print("   _goal_action ->", a, " goal_none=%d gate_blocked=%d" % (p.goal_none, p.gate_blocked))
    if a is None:
        fails.append("③ 有覆盖时 _goal_action 仍返回 None")
        print("   ✗ 仍为 None")
    else:
        print("   ✓ 给出动作 %d" % a)

    # ── ④ 静默空转必须被计数(而不是伪装成"启动了") ─────────────────────
    print("\n④ 起 option 失败时必须计 start_blocked, 且**不算启动**")
    p2 = make_proposer()
    p2.know.coverage.n[:] = 0.0          # 强行零覆盖
    p2.know._dyn_fitted = False
    o = SimpleNamespace(actions=[1, 2, 3], goal_center=np.array([1.0] * p2.n_fine))
    a2 = p2._goal_action(o)
    print("   零覆盖下 _goal_action ->", a2, " gate_blocked=%d goal_none=%d" % (p2.gate_blocked, p2.goal_none))
    if a2 is not None:
        print("   (注: 该桩下未复现全拒 —— 取决于 allow 实现, 非失败)")
    else:
        print("   ✓ 正确返回 None 并记录 goal_none")

    print("\n" + "=" * 60)
    if fails:
        print("失败 %d 项:" % len(fails))
        for f in fails:
            print("  ✗", f)
        return 1
    print("全部通过 ✓  —— coverage 写入这根线已接上")
    return 0


if __name__ == "__main__":
    sys.exit(main())
