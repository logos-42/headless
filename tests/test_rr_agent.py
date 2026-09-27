# -*- coding: utf-8 -*-
"""`RRSkillAgent` 的机制自检 —— 判据在跑之前写死。

对应 `option-design-and-execution.md` §3 的**模式计数器表**:

    模式                   必填计数器特征
    ─────────────────────  ────────────────────────────────
    open-loop(固定序列)   recompute-count **0**
    closed-loop(闭环)     recompute-count **≈ 执行步数**
    + 终止判据             终止原因台账**非空**
    + 证据中断             非零中断计数

再加三条本项目特定的判据:

  A. **对照干净**: `RRSkillAgent(mode="primitive")` 必须与
     `SkillAgent(mode="primitive")` **逐位相同**(子类只加了 rr 分支,
     其余路径逐字委托)。
  B. **lock-in 检**: 平均执行时长不得接近整回合长度(§3 第二条)。
  C. **testbed 能学**(§8): rr 臂自己必须能解出任务, 否则后面所有臂
     的比较都是在测一个冻结的 agent。
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from hibs_lnn.skill_mdp import KeyDoorMDP                    # noqa: E402
from hibs_lnn.skill_agent import SkillAgent                  # noqa: E402
from hibs_lnn.rr_agent import RRSkillAgent                   # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


REGIMES = [(2, 5, 7), (1, 4, 7), (3, 6, 7)]
N_EP = 60          # 短跑: 只验机制是否活, 不验效果


def make(mode, seed, cls=SkillAgent, **kw):
    mdp = KeyDoorMDP(n_pos=8, key_pos=2, door_pos=5, goal_pos=7, horizon=30)
    mdp.set_regime(*REGIMES[0])
    return cls(mdp, mode=mode, seed=seed, **kw), mdp


def run_chain(ag, mdp, n_per=10):
    succ = []
    for reg in REGIMES:
        mdp.set_regime(*reg)
        for _ in range(n_per):
            ok, _n = ag.run_episode()
            succ.append(int(ok))
    return succ


print("=" * 72)
print("RRSkillAgent 机制自检")
print("=" * 72)

# ── A. 对照干净: primitive 路径逐位委托 ────────────────────────────
for seed in (0, 1):
    a1, m1 = make("primitive", seed, cls=SkillAgent)
    a2, m2 = make("primitive", seed, cls=RRSkillAgent)
    s1 = run_chain(a1, m1)
    s2 = run_chain(a2, m2)
    q_same = bool(np.array_equal(a1.q.Q, a2.q.Q))
    check(f"seed={seed}: RRSkillAgent(mode=primitive) == SkillAgent(mode=primitive)",
          s1 == s2 and q_same,
          f"success {'逐位相同' if s1==s2 else 'DIFF'}  Q表 {'逐位相同' if q_same else 'DIFF'}")

# ── B/C. rr 臂: 机制是否活 ────────────────────────────────────────
ag, mdp = make("rr", 0, cls=RRSkillAgent, opt_prob=1.0)
succ = run_chain(ag, mdp)
bk = ag.rr_bookkeeping()
print()
print("rr 台账:", {k: bk[k] for k in
                   ("recomputes", "selections", "exec_steps", "terms",
                    "n_options", "mean_duration", "max_duration",
                    "lockin_ratio", "max_cap", "rebuilds", "beta_nonempty")})

check("① 启动过 option(selections > 0)", bk["selections"] > 0,
      f"selections={bk['selections']}")
check("② 闭环: recompute-count ≈ 执行步数(> 0 且 ≥ 80% 执行步)",
      bk["recomputes"] > 0 and bk["recomputes"] >= 0.8 * max(1, bk["exec_steps"]),
      f"recomputes={bk['recomputes']} exec_steps={bk['exec_steps']}")
check("③ 终止台账**非空**(承诺了终止就必须真的终止)",
      len(bk["terms"]) > 0, f"terms={bk['terms']}")
check("③b 台账里有**β_o 触发**(不是超时兜底撑起来的)",
      (bk["terms"].get("beta_at_feature", 0) + bk["terms"].get("beta_elsewhere", 0)) > 0,
      f"beta_at_feature={bk['terms'].get('beta_at_feature',0)} "
      f"beta_elsewhere={bk['terms'].get('beta_elsewhere',0)} "
      f"expired={bk['terms'].get('expired',0)}")
check("④ lock-in 检: 平均时长 < 50% 整回合",
      bk["lockin_ratio"] < 0.5,
      f"mean={bk['mean_duration']:.2f} / horizon={bk['horizon']} "
      f"ratio={bk['lockin_ratio']:.3f} cap={bk['max_cap']}")
check("⑤ 库里有 option 且 β 非空", bk["n_options"] > 0 and bk["beta_nonempty"] > 0,
      f"n={bk['n_options']} |β|={bk['beta_nonempty']}")
check("⑥ testbed 能学: rr 臂至少解出过半回合", sum(succ) > len(succ) * 0.5,
      f"success={sum(succ)}/{len(succ)}")

# ── D. 消融旋钮在 agent 层也真的生效 ──────────────────────────────
#   ⑦ 的精确形式(Pitfall-26 的正确读法):
#     两个臂台账**逐位相同**时, 只有两种可能 ——
#       (a) 接线 bug(旋钮没接上), 或
#       (b) 该试验台上两者**真的行为等效**。
#     区分方法: **比较它们的输入**(z 表 / c 函数)。输入不同而台账相同
#     = (b) 真等效(可报告); 输入相同 = (a) bug(必须修)。
print()
arm_stats, arm_inputs = {}, {}
for tag, kw in [
    ("rr-exact",   dict()),
    ("rr-zeroV",   dict(rr_z_mode="zeroV")),
    ("rr-nobonus", dict(rr_z_mode="nobonus")),
    ("rr-nocum",   dict(rr_cumulant="zero")),
    ("rr-learned", dict(rr_v_main="learned")),
    ("bottleneck", dict(rr_lib="bottleneck")),
    ("random",     dict(rr_lib="random")),
]:
    a, m = make("rr", 0, cls=RRSkillAgent, opt_prob=1.0, **kw)
    run_chain(a, m)
    arm_stats[tag] = a.rr_bookkeeping()
    # 输入签名: 每个 option 的 z 表(有限值处)+ c 表。
    # ★ c 必须探**全部 4 个动作** —— 只探动作 0 会让 `exact` 与 `nocum`
    #   在 goal_pos=7 的 regime 上签名相同(因为 `R[·][LEFT] ≡ 0`, 奖励
    #   永远不会从"向左"这一步拿到)。同族教训: 探针必须打在**能显出差别
    #   的地方**(参 handrolled-module-verification "探针不能打在原点")。
    lib = a.rr_lib_obj
    zsig, csig = [], []
    for st in lib:
        z = np.array([st.z_fn(s) for s in range(a.mdp.n_states)])
        zsig.append(tuple(np.round(z[np.isfinite(z)], 6)))
        csig.append(tuple(round(st.c_fn(s, act), 6)
                          for s in range(a.mdp.n_states)
                          for act in range(4)))
    arm_inputs[tag] = (tuple(zsig), tuple(csig))
    print(f"  {tag:11s} terms={arm_stats[tag]['terms']} "
          f"mean_dur={arm_stats[tag]['mean_duration']:.2f} "
          f"recomp={arm_stats[tag]['recomputes']}")

sig = {t: (str(s["terms"]), round(s["mean_duration"], 3), s["recomputes"])
       for t, s in arm_stats.items()}
uniq = len(set(sig.values()))
check("⑦a 三个结构上最不同的臂(bottleneck / random / rr-exact)台账互不相同",
      len({sig[k] for k in ("bottleneck", "random", "rr-exact")}) == 3,
      f"{uniq}/7 个不同签名")

# ⑦c: 台账相同的臂 —— 逐组给出**可检查的**等效原因, 而不是放松判据。
groups = {}
for t, s in sig.items():
    groups.setdefault(s, []).append(t)
for g in [g for g in groups.values() if len(g) > 1]:
    ins = {arm_inputs[t] for t in g}
    if len(ins) == len(g):
        check(f"⑦c {g} 台账相同但**输入不同** ⇒ 真等效(非共享分支)",
              True, f"{len(ins)}/{len(g)} 种输入签名")
    else:
        # 输入也相同 -> 必须给出一个**代数上可检查**的原因, 否则就是接线 bug。
        # 已知情形: `zeroV`(V_main ≡ 0)与 `learned`(V_main = agent 的 Q)。
        #   zeroV   : w_i=0, z = 0 + (w̄−0)·1 = w̄        (特征处)
        #   learned : w_i=c, z = c + (w̄−c)·1 = w̄        (特征处)
        # ⇒ **只要 V_learned 是常数, 两者 z 代数恒等**。检查 V_learned 是否常数。
        explained = False
        if "rr-learned" in g:
            a_l, _ = make("rr", 0, cls=RRSkillAgent, opt_prob=1.0,
                          rr_v_main="learned")
            a_l._rr_ensure()
            vl = np.asarray(a_l._rr_v_main_arr, dtype=float)
            explained = bool(np.std(vl) < 1e-9)
            detail = f"V_learned std={np.std(vl):.3e} (常数 ⇒ 代数恒等)"
        else:
            detail = "无可解释原因"
        check(f"⑦c {g} 台账与输入都相同 ⇒ 必须给出可检查的等效原因, 否则是接线 bug",
              explained, detail)

check("⑦b rr-exact 与 bottleneck 台账不同",
      sig["rr-exact"] != sig["bottleneck"],
      f"rr={sig['rr-exact']}  bot={sig['bottleneck']}")

print("=" * 72)
if FAILS:
    print(f"✗ {len(FAILS)} 项未过: {FAILS}")
    sys.exit(1)
print("✓ 全部通过")
