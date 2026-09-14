#!/usr/bin/env python3
"""验证 OaK 三个结构件: option 发现 / 反事实 rollout / coverage+uncertainty 门控。

## 四测项

  ① **Option 发现**: 从转移结构中发现**长度 ≥2 的动作序列**(时间抽象),
     且只采用**可靠**动作(高噪声动作应被 τ_U 筛掉)。
  ①b **预测正确性**: 含**远离数据质心**的探针 —— 这是曾经漏掉的回归点。
  ② **门控挡 OOD**: 零覆盖的 (s,a) 必须被拒; 高噪声动作的 U_T 必须显著高于低噪声。
  ③ **rollout 中止**: 反事实 rollout 一进入零覆盖区就停, 不硬外推。
  ④ **学习式终止 β_o**: 终止概率从数据学出, 不是硬阈值。

## 夹具的尺度是唯一的难点 (三次踩坑记录, 都不是模块的错)

  ① 状态生成成 i.i.d. 噪声 -> 状态根本不演化 -> 不存在多步结构 -> 0 option
  ② 动作效果 (0.3) << 区域间距 (数个单位) -> `center+effect` 的最近区域永远是
     它自己 -> 图里没有"前进的边" -> 0 option
  ③ 纯随机游走**无界漂移** -> 区域中心跑到 ~12 -> 同一个病

  可行解 = **低维链式环境**: 状态沿一条线排开 (dim_x=1), 单步动作跨过相邻区域,
  多步动作序列跨过多个区域 => 时间抽象才有意义。这正是 option 的经典形态。
  真实 lm4 的对应关系: 状态 = [0,1] 的精度画像 (有界), 单次训练改变 0.1~0.3,
  与 6 个域的区域间距同量级 —— 天生满足条件。
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.option_manager import OptionManager          # noqa: E402
from hibs_lnn.uncertainty_gate import (                     # noqa: E402
    TransitionEnsemble,
    UncertaintyGate,
)

DIM_X, N_DOM = 1, 4
# 动作效果: 0 可靠推进 / 1 小幅推进 / 2 高噪声 / 3 几乎不动
EFF = {0: np.full(DIM_X, 0.60), 1: np.full(DIM_X, 0.25),
       2: np.full(DIM_X, 0.20), 3: np.full(DIM_X, 0.03)}
NOI = {0: 0.02, 1: 0.02, 2: 0.60, 3: 0.02}


def build(n_ep=2500, steps=5, seed=0):
    rng = np.random.default_rng(seed)
    X, Y, S, A = [], [], [], []
    for _ in range(n_ep):
        x = rng.normal(0, 0.05, DIM_X)
        for _ in range(steps):
            u = int(rng.integers(0, N_DOM))
            y = x + EFF[u] + rng.normal(0, NOI[u], DIM_X)
            X.append(np.append(x, u)); Y.append(y)
            S.append(x.copy()); A.append(u)
            x = y
    return np.array(X), np.array(Y), np.array(S), np.array(A)


X, Y, STATES, ACTS = build()
print("训练样本 %d, 动作噪声: %s" % (len(X), NOI))

visits = np.bincount(ACTS, minlength=N_DOM).astype(float)
ens = TransitionEnsemble(n_models=5, seed=0).fit(X, Y, N_DOM, None)
gate = UncertaintyGate(tau_C=1.0)
_step = max(1, len(X) // 100)
gate.calibrate([ens.predict(X[i, :DIM_X], int(X[i, -1]), visits)[1]
                for i in range(0, len(X), _step)])
print("门控标定: τ_C=%.2f  τ_U=%.6f  (训练分歧 90 分位)" % (gate.tau_C, gate.tau_U))

N_REG = 5
om = OptionManager(ens, gate, max_len=5, seed=0)
opts = om.discover(STATES, ACTS, n_regions=N_REG)
fails = []

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("① Option 发现 (应从转移结构发现长度 ≥2 的动作序列)")
print("=" * 92)
multi = [o for o in opts if len(o.actions) >= 2]
print("  发现 %d 个 option, 转移图 %d 条边" % (len(opts), len(om.graph)))
for o in opts[:8]:
    print("    actions=%-12s 长度=%d" % (str(o.actions), len(o.actions)))
print("  -> 长度 ≥2 的 option: %d 个  %s"
      % (len(multi), "✓ 形成了时间抽象" if multi else "**只有单步 ✗**"))
if not multi:
    fails.append("option 发现未产生多步序列")

# 只应采用可靠动作: 高噪声的动作 2 不应出现在 option 里
used = {a for o in opts for a in o.actions}
print("  用到的动作集合 = %s  (动作2 是高噪声=%.2f)" % (sorted(used), NOI[2]))
if 2 in used:
    print("  **高噪声动作被采纳 ✗**")
    fails.append("高噪声动作混入 option")
else:
    print("  ✓ 高噪声动作被 τ_U 筛掉 —— 可靠路径过滤起作用")

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("①b 预测正确性 (含**远离质心**—— 这是曾经漏掉的回归点)")
print("=" * 92)
probe = om.regions[0] if om.regions is not None else np.zeros(DIM_X)
print("  探针 = 区域中心 R0 = %s (远离数据质心)" % np.round(probe, 3))
ok_cnt = 0
for u in range(N_DOM):
    s_next, unc = ens.predict(probe, u, visits)
    got, want = s_next - probe, EFF[u]
    err = float(np.max(np.abs(got - want)))
    good = err < 0.12                      # 噪声 std=0.02~0.60, 留余量
    ok_cnt += good
    print("    a%d: Δ=%s  真值=%s  最大误差=%.4f  %s"
          % (u, np.round(got, 3), np.round(want, 3), err, "✓" if good else "✗"))
print("  -> %d/%d 个动作在远离质心处预测正确  %s"
      % (ok_cnt, N_DOM, "✓" if ok_cnt >= 3 else "**仍有偏差 ✗**"))
if ok_cnt < 3:
    fails.append("远离质心的预测不正确")

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("② 门控挡 OOD (复现实测 case)")
print("=" * 92)
x_ok = STATES[0]
_, unc_lo = ens.predict(x_ok, 0, visits)
_, unc_hi = ens.predict(x_ok, 2, visits)
print("  动作0 有覆盖 (visits=%.0f):  allow=%s" % (visits[0], gate.allow(x_ok, 0, visits)))
print("  动作0 零覆盖 (复现 [2,2,...]):  allow=%s  %s"
      % (gate.allow(x_ok, 0, np.zeros(N_DOM)),
         "✓ 被挡住" if not gate.allow(x_ok, 0, np.zeros(N_DOM))[0] else "**没挡住 ✗**"))
if gate.allow(x_ok, 0, np.zeros(N_DOM))[0]:
    fails.append("零覆盖未被挡住")
print("  动作0 (低噪声) U_T=%.5f   动作2 (高噪声) U_T=%.5f   -> 高噪声被识别 %s"
      % (unc_lo, unc_hi, "✓" if unc_hi > 3 * unc_lo else "**区分不足 ✗**"))
if not unc_hi > 3 * unc_lo:
    fails.append("U_T 未能区分高低噪声")

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("③ rollout 在无覆盖处中止")
print("=" * 92)
_traj, _uncs, aborted = ens.rollout(x_ok, [0, 0, 1], visits, gate=None)
print("  不限门控 rollout [0,0,1]:  %d 步, 中止=%s" % (len(_traj), aborted))
_seq2 = [0, 0, 0]
_t2, _u2, ab2 = ens.rollout(x_ok, _seq2, np.array([1.0, 1.0, 1.0, 0.0]), gate=gate)
print("  零覆盖下 rollout %s (门控开):  %d 步, 中止=%s  %s"
      % (_seq2, len(_t2), ab2, "✓ 中止了" if ab2 else "**没中止 ✗**"))
if not ab2:
    fails.append("rollout 未在零覆盖处中止")

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("④ 学习式终止 β_o (应从数据学, 不是硬阈值)")
print("=" * 92)
if not opts:
    print("  (无可发现的 option, 跳过)")
else:
    o = multi[0] if multi else opts[0]
    goal = om.regions[int(np.argmax(om.regions[:, 0]))]
    # 样本 = (状态, 不确定性, 是否终止) —— 3 元组
    smp = [(STATES[i], 0.0, bool(abs(STATES[i, 0] - goal[0]) < 0.3))
           for i in range(0, len(STATES), 5)]
    o.learn_termination(smp, lr=0.1, epochs=300)
    print("  option actions=%s, 目标区域中心=%s" % (o.actions, np.round(goal, 3)))
    print("  β_o 权重 = %s" % np.round(o.w_beta, 4))
    print("  终止概率随「距目标」的变化:")
    ps = []
    for d in (-1.0, -0.5, -0.2, 0.0, 0.2):
        pv = o.beta_prob(goal + d)
        ps.append(pv)
        print("    距目标 %+.1f -> P(terminate)=%.4f" % (d, pv))
    mono = ps[-1] > ps[0]
    print("  -> 训练后 P(终止) 随「接近目标」上升: %s" % ("✓" if mono else "✗ 未学到"))
    if not mono:
        fails.append("学习式终止未学到「接近目标则终止」")

print()
print("=" * 92)
print("stats: %d options, tau_U = %.6f" % (len(opts), om.tau_U))
if fails:
    print("✗ 失败项: %s" % "; ".join(fails))
    sys.exit(1)
print("✓ 全部通过")
