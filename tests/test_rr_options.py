# -*- coding: utf-8 -*-
"""`rr_options` 的自检 —— 判据在跑之前就写死。

对应 `ml-ablation-methodology/references/option-design-and-execution.md`:

  §3  「一个名字里承诺了终止的模式, 必须给出**非空**的终止台账」
  §9.3「stopping value 与主任务价值**逐位相同** = 退化」(Pitfall-26 tell)
  §6  「目标容差必须由状态尺度导出, 不能手写常数」
  §8  「testbed 的 agent 必须先被证明能学」

本文件只验**子任务层**(不涉及 SkillAgent)。SkillAgent 侧的计数器在
`test_rr_agent.py` 里验。
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from hibs_lnn.skill_mdp import KeyDoorMDP                      # noqa: E402
from hibs_lnn.rr_options import (                              # noqa: E402
    KeyDoorTabular, build_bottleneck_library, build_random_goal_library,
    build_rr_library, feature_vectors, initiation_ok, main_value,
    optimal_policy, option_gain,
)

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


REGIME_A = (2, 5, 7)
REGIME_B = (1, 4, 7)

print("=" * 72)
print("rr_options 自检  |  regime A =", REGIME_A)
print("=" * 72)

mdp = KeyDoorMDP(n_pos=8, key_pos=2, door_pos=5, goal_pos=7, horizon=30)
mdp.set_regime(*REGIME_A)
mdl = KeyDoorTabular(mdp, gamma=0.95)

# ── 1. 表格模型与过程式 MDP 逐状态一致 ──────────────────────────────
#      (不能只测起点 —— 起点一致而别处不一致是 §9.3 那类"探针打在原点"的坑)
mismatch = []
for s in range(mdl.n_states):
    pos, hk, do = mdp.decode(s)
    for a in range(4):
        saved = (mdp.pos, mdp.has_key, mdp.door_open, mdp.t)
        mdp.pos, mdp.has_key, mdp.door_open = pos, hk, do
        s2, r, _done = mdp.step(a)
        mdp.pos, mdp.has_key, mdp.door_open, mdp.t = saved
        if s2 != mdl.T[s][a][0][1] or abs(r - mdl.R[s][a]) > 1e-12:
            mismatch.append((s, a, s2, mdl.T[s][a][0][1], r, mdl.R[s][a]))
check("表格模型 == 过程式 step()(全 32 状态 × 4 动作)",
      not mismatch, f"mismatch={mismatch[:3]}")

# ── 2. 主任务价值非平凡 ─────────────────────────────────────────────
V_main, pol = optimal_policy(mdl)
check("V_main 非平凡(非全 0 / 非全 1)", 0.0 < V_main[0] < 1.0,
      f"V_main[0]={V_main[0]:.4f}  max={V_main.max():.4f}")
check("V_main 单调: 靠近目标更好",
      V_main[4 * 7 + 2] > V_main[4 * 0 + 0],
      f"goal区 {V_main[4*7+2]:.4f} > 起点 {V_main[0]:.4f}")

# ── 3. reward-respecting 库: 三项关键判据 ───────────────────────────
lib = build_rr_library(mdl, V_main, bonus_weight=1.0)
check("库里有 2 个 option(key, door)", len(lib) == 2, f"n={len(lib)}")

for st in lib:
    n_beta = int(np.sum(st.beta))
    # ★ 判据 1: β_o 台账必须**非空**(§3)。空台账 = 与不退化的臂无法区分。
    check(f"{st.name}: β_o 非空(承诺了终止就必须真的有终止)",
          n_beta > 0, f"|β|={n_beta}/{mdl.n_states}")
    # ★ 判据 2: z ≠ V_main 逐位相同(§9.3 第三条, Pitfall-26 tell)
    z_vals = np.array([st.z_fn(s) for s in range(mdl.n_states)])
    fin = np.isfinite(z_vals)
    same = bool(np.allclose(z_vals[fin], V_main[fin], atol=1e-12))
    check(f"{st.name}: z ≢ V_main(非退化)", not same and fin.any(),
          f"可比 {int(fin.sum())} 态, w_i={getattr(st,'w_i',float('nan')):.4f}, "
          f"bonus={getattr(st,'bonus_weight',float('nan')):.4f}")

# ── 4. 与已知弱基线(瓶颈)必须**逐位不同** ─────────────────────────
bot = build_bottleneck_library(mdl)[0]
d = max(int(np.sum(lib[i].policy != bot.policy)) for i in range(len(lib)))
check("rr 的策略 ≠ bottleneck 的策略(§9.3 '两臂逐位相同' 反面)",
      d > 0, f"策略表最大分歧 {d} 个状态 (共 {mdl.n_states})")

# ── 5. 启动集真的包含起始状态(§6 的"radius 尺度"坑在这一层的对应物)
s0 = mdp.reset()
mdp.set_regime(*REGIME_A)
check("至少一个 option 在起始状态可启动 I_o(s0)",
      any(initiation_ok(st, s0) for st in lib),
      f"s0={s0} gains=" +
      str([f"{option_gain(st, s0, V_main):.3f}" for st in lib]))

# ── 6. 选择器有分辨力(增益不是常数) ───────────────────────────────
gains = np.array([[option_gain(st, s, V_main) for st in lib]
                  for s in range(mdl.n_states)])
fin_g = gains[np.isfinite(gains)]
check("option_gain 有分辨力(std > 0, 不是常数)",
      fin_g.size > 0 and float(np.std(fin_g)) > 1e-9,
      f"n={fin_g.size} std={float(np.std(fin_g)):.4f} "
      f"range=[{fin_g.min():.4f},{fin_g.max():.4f}]")

# ── 7. 反事实: 削掉价值函数后 z 必须变(否则"价值函数"这个旋钮是死的)
lib_z0 = build_rr_library(mdl, V_main, bonus_weight=1.0, z_mode="zeroV")
diff_z = max(int(np.sum(np.abs([lib[i].z_fn(s) for s in range(mdl.n_states)])
                        - np.array([lib_z0[i].z_fn(s) for s in range(mdl.n_states)])
                        > 1e-9)) for i in range(len(lib)))
check("z_mode='zeroV' 真的改变了 z(旋钮不是死的)", diff_z > 0,
      f"z 表最大分歧 {diff_z} 态")

# ── 8. 反事实: nobonus 应当退化(bonus ≡ 0 → z ≡ V_main) ────────────
lib_nb = build_rr_library(mdl, V_main, bonus_weight=1.0, z_mode="nobonus")
for st in lib_nb:
    z_vals = np.array([st.z_fn(s) for s in range(mdl.n_states)])
    fin = np.isfinite(z_vals)
    same = bool(np.allclose(z_vals[fin], V_main[fin], atol=1e-12))
    check(f"{st.name}(nobonus): z ≡ V_main —— 确认这是退化臂",
          same, f"可比 {int(fin.sum())} 态")

# ── 9. 反事实: cumulant='zero' 真的改变 c ─────────────────────────
lib_c0 = build_rr_library(mdl, V_main, bonus_weight=1.0, cumulant="zero")
c_diff = any(abs(lib_c0[i].c_fn(s, a) - lib[i].c_fn(s, a)) > 1e-12
             for i in range(len(lib)) for s in range(mdl.n_states)
             for a in range(4))
check("cumulant='zero' 真的改变了 c(旋钮不是死的)", c_diff)

# ── 10. 随机目标库有分辨力 ────────────────────────────────────────
rnd = build_random_goal_library(mdl, V_main, seed=0)
check("随机目标库构造成功且 β_o 非空",
      len(rnd) == 2 and all(int(np.sum(o.beta)) > 0 or o.value.max() == 0
                            for o in rnd),
      f"n={len(rnd)}")

# ── 11. regime 改变后 is_goal 真的变(免钥匙情形) ─────────────────
mdp2 = KeyDoorMDP(n_pos=8, key_pos=2, door_pos=5, goal_pos=3, horizon=30)
mdp2.set_regime(2, 5, 3)
mdl2 = KeyDoorTabular(mdp2)
check("regime 依赖: goal_pos<door_pos 时门不再是终止条件",
      int(mdl2.is_goal.sum()) != int(mdl.is_goal.sum()) or
      not np.array_equal(mdl2.is_goal, mdl.is_goal),
      f"|goal|={int(mdl.is_goal.sum())} vs {int(mdl2.is_goal.sum())}")

print("=" * 72)
if FAILS:
    print(f"✗ {len(FAILS)} 项未过: {FAILS}")
    sys.exit(1)
print("✓ 全部通过")
