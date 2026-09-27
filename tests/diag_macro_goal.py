"""diag_macro_goal.py — 诊断 macro 模式: β_o 的触发原因分布 + 目标距离分布。

要回答的问题: macro 成功率只有 ~24%, 是「到不了目标」还是「阈值太严」?

★ 做法: **子类挂钩 _macro_step, 走真实的 run_episode 路径**。
  第一版我手写复刻了 run_episode 的循环, 结果忘了 `q.update` 与接收 `done`
  -> 150 回合只访问了 3 个唯一状态 -> 发现不出任何 option -> 零样本 (假数据)。
  **诊断必须观测真实路径, 不能复刻。** 复刻出来的零样本会被误读成「机制坏了」。

事前判据:
  · 若每次执行的 min_distance 已接近 eps (略大于)  -> 阈值问题
  · 若 min_distance 普遍远离 eps (≥2-3 倍)          -> 真的到不了
"""
import sys
sys.path.insert(0, "/Users/apple/Downloads/headless")

import numpy as np

from hibs_lnn.skill_mdp import KeyDoorMDP
from hibs_lnn.skill_agent import SkillAgent, DEFAULT_REGIMES

EXEC_LOG = []          # 每次 option 执行: {"min_d", "reason", "steps"}


class DiagAgent(SkillAgent):
    """只加观测, 不改行为。"""

    def _macro_step(self, v, active, steps):
        a, new_active, new_steps = super()._macro_step(v, active, steps)
        if new_active is not None:
            try:
                d = float(new_active.goal_distance(v))
            except Exception:
                d = float("nan")
            if getattr(self, "_cur", None) is None:
                # ★ 这一步是**执行的第一步** —— 记下起点距离
                self._cur = {"start_d": d, "dists": []}
            self._cur["dists"].append(d)
        if active is not None and new_active is None:   # 本步终止了这次执行
            eps = self.term_eps
            if eps is None:
                eps = float(getattr(self.om, "init_radius", 0.75))
            cur = self._cur or {"start_d": float("nan"), "dists": [float("nan")]}
            EXEC_LOG.append({"start_d": cur["start_d"],
                             "min_d": min(cur["dists"]),
                             "reason": "reached" if cur["dists"][-1] < eps else "budget",
                             "steps": len(cur["dists"])})
            self._cur = None
        return a, new_active, new_steps


def diag(seed=42, n_ep=150):
    chain = DEFAULT_REGIMES * 2
    mdp = KeyDoorMDP(n_pos=8, horizon=30, **chain[0])
    ag = DiagAgent(mdp, mode="macro", seed=seed)
    eps_used = None
    for ci, reg in enumerate(chain):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        ag.maybe_discover(force=True)
        if eps_used is None:
            eps_used = float(getattr(ag.om, "init_radius", 0.75))
        for _ in range(n_ep):
            ag.run_episode()
    return eps_used, ag


eps, ag = diag()
bk = ag.option_bookkeeping()
all_d = np.array([r["min_d"] for r in EXEC_LOG if np.isfinite(r["min_d"])])
n_r = sum(1 for r in EXEC_LOG if r["reason"] == "reached")
n_b = sum(1 for r in EXEC_LOG if r["reason"] == "budget")

print("=== macro 的 β_o 触发诊断 (eps = om.init_radius = %.3f) ===" % eps)
print("  台账(权威): %d/%d 成功/执行 (%.1f%%)"
      % (bk["total_success"], bk["total_visits"],
         100.0 * bk["total_success"] / max(1, bk["total_visits"])))
print("  观测到的执行次数: %d  (reached=%d budget=%d)" % (len(EXEC_LOG), n_r, n_b))
print()
if len(all_d):
    starts = np.array([r["start_d"] for r in EXEC_LOG if np.isfinite(r["start_d"])])
    print("  每次执行的**最小** goal_distance 分布 (执行粒度):")
    for q in (5, 25, 50, 75, 95):
        print("    p%-3d = %.3f" % (q, np.percentile(all_d, q)))
    print("    min = %.3f  max = %.3f" % (all_d.min(), all_d.max()))
    print()
    print("  ★ 起点距离 vs 执行中最小距离 (判别「快到了」还是「根本没动」):")
    print("    起点距离   中位 %.3f   p25 %.3f   p75 %.3f" % (
        np.median(starts), np.percentile(starts, 25), np.percentile(starts, 75)))
    print("    最小距离   中位 %.3f   p25 %.3f   p75 %.3f" % (
        np.median(all_d), np.percentile(all_d, 25), np.percentile(all_d, 75)))
    improved = float((all_d < starts - 1e-9).mean())
    same = float((np.abs(all_d - starts) <= 1e-9).mean())
    worse = float((all_d > starts + 1e-9).mean())
    print("    对比: 变近 %.1f%%   没变 %.1f%%   变远 %.1f%%"
          % (100 * improved, 100 * same, 100 * worse))
    print()
    print("=== 反事实: 阈值放宽到不同倍数下的达标率 ===")
    for mult in (1.0, 1.5, 2.0, 3.0):
        e = eps * mult
        print("  eps = %.2f (= %.1fx): 达标率 %.1f%%"
              % (e, mult, 100.0 * float((all_d < e).mean())))
    print()
    frac1 = float((all_d < eps).mean())
    frac2 = float((all_d < eps * 2).mean())
    med = float(np.median(all_d))
    print("=== 关键判别 ===")
    if improved < 0.2:
        print("  -> **根本没动**: 只有 %.1f%% 的执行比起点更近 —— 阈值不是主因,"
              % (100 * improved))
        print("     π_o 朝目标走的能力本身有问题")
    elif frac2 - frac1 > 0.3:
        print("  -> **阈值问题**: 放宽到 2x 后达标率 %.1f%% -> %.1f%%" % (100 * frac1, 100 * frac2))
    else:
        print("  -> 两者都不是: 变近比例 %.1f%%, 放宽 2x 达标率 %.1f%%" % (100 * improved, 100 * frac2))
    print("     中位最小距离 %.3f = eps 的 %.1f 倍" % (med, med / max(eps, 1e-9)))
else:
    print("  (没有观测到任何 option 执行)")
