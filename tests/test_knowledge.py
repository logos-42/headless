#!/usr/bin/env python3
"""验证 InternalKnowledge 层: 知识是**可计算对象**, 不是参数。

五测项:
  ① **Knowledge ≠ Parameter**: `knowledge_of(s,a)` 返回的是**内容**
     (预测/价值/覆盖/可信度), 不是权重; 且**零经验**时它**承认不知道**。
  ② **知识能留下来**: `to_dict()` -> 新对象 -> `growth()` 能逐项指出变化。
  ③ **增长可测**: 更多经验后, coverage/GVF/dynamics 的变化量都是**具体数字**。
  ④ **不确定性是知识的一部分**: 高噪声动作的 confidence 显著低于低噪声。
  ⑤ **可塑性是元知识**: α 在真预测特征上分化, 且与 w (知识) **分开**统计。
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.knowledge import Coverage, GVFBank, InternalKnowledge, Plasticity  # noqa: E402

DIM_X, N_DOM = 4, 4
EFF = {0: np.array([0.6, 0, 0, 0]), 1: np.array([0, 0.6, 0, 0]),
       2: np.array([0.3, 0.3, 0.3, 0.3]), 3: np.array([0.05, 0.05, 0, 0])}
NOI = {0: 0.03, 1: 0.03, 2: 0.50, 3: 0.04}
fails = []


def make_data(n_ep, steps=6, seed=0):
    rng = np.random.default_rng(seed)
    X, Y, A = [], [], []
    for _ in range(n_ep):
        x = rng.normal(0, 0.1, DIM_X)
        for _ in range(steps):
            u = int(rng.integers(0, N_DOM))
            y = x + EFF[u] + rng.normal(0, NOI[u], DIM_X)
            X.append(np.append(x, u)); Y.append(y); A.append(u)
            x = y
    return np.array(X), np.array(Y), np.array(A)


X, Y, A = make_data(300)
K = InternalKnowledge(N_DOM, DIM_X, n_models=5, seed=0)
for a in A:
    K.observe_action(int(a))
K.fit_dynamics(X, Y)
K.register_value_fn(
    lambda s, a: float(np.sum(EFF[a])) if a in EFF else float("nan"),
    policy="greedy-eff")
K.register_options(None)          # 抽象层稍后单独验

print("训练样本 %d, 覆盖 = %s" % (len(X), K.coverage.n.astype(int).tolist()))
print("τ_U = %.6f" % (K.uncertainty.tau_U or float("nan")))

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("① Knowledge ≠ Parameter —— 查询返回**内容**, 且零经验时承认不知道")
print("=" * 92)
s = np.array([0.1, -0.05, 0.02, 0.0])
for a in (0, 2, 7):
    k = K.knowledge_of(s, a)
    print("  a=%s -> 预测=%s  V=%s  覆盖=%.0f  可信=%s  新=%s  **判定=%s**"
          % (k["action"], k["predicted_next"],
             ("%.3f" % k["value"]) if k["value"] is not None else "None",
             k["coverage"], k["trustworthy"], k["novel"], k["verdict"]))
k7 = K.knowledge_of(s, 7)
if not k7["novel"] or k7["trustworthy"] or k7["known"]:
    fails.append("零覆盖动作未被识别为 novel/不可信/未知")
else:
    print("  -> 动作 7 从未见过: novel=True trustworthy=False verdict='我不知道'")
    print("     ✓ 它**承认识不知道** —— 而不是给一个编出来的数")
k0 = K.knowledge_of(s, 0)
if k0["novel"]:
    fails.append("有覆盖的动作被误判为 novel")
d = K.describe()
print("  查询接口暴露的是%s" % ("内容(预测/价值/覆盖/置信)" if "predicted_next" in k0 else "参数 ✗"))
print("  describe() 顶层键: %s" % list(d.keys()))

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("② 知识能留下来 + ③ 增长可测")
print("=" * 92)
snap = K.to_dict()                     # 冻结当前知识
print("  快照: coverage_total=%d  n_options=%d  gvf_steps=%s  β_updates=%d"
      % (snap["coverage_total"], snap["n_options"], snap["gvf_steps"], snap["plasticity_updates"]))

# 继续积累经验 (新一批数据 + GVF 观察)
X2, Y2, A2 = make_data(300, seed=1)
for a in A2:
    K.observe_action(int(a))
K.dynamics_model.fit(X2, Y2, N_DOM, None)
phi = np.ones(DIM_X)
for t in range(50):
    K.observe_gvf(phi, [0.1, 0.2, 0.3, 0.4], phi)     # 预测知识在长
    K.observe_plasticity(phi * (0.5 if t % 3 else 3.0), 0.2)
g = K.growth(snap)
print("  growth():")
for kk, vv in g.items():
    if isinstance(vv, float) and not np.isfinite(vv):
        continue
    print("     %-26s %s" % (kk, np.round(vv, 5) if isinstance(vv, (list, float)) else vv))
checks = [
    ("coverage_total_delta", g.get("coverage_total_delta", 0) > 0, "覆盖度增长"),
    ("gvf_weight_delta_mean", g.get("gvf_weight_delta_mean", 0) > 0, "预测知识变化"),
    ("dyn_weight_delta_mean", g.get("dyn_weight_delta_mean", 0) > 0, "转移知识变化"),
    ("plasticity_beta_delta", g.get("plasticity_beta_delta") is not None, "元知识变化"),
]
for name, ok, desc in checks:
    print("     %s %s" % ("✓" if ok else "✗", desc))
    if not ok:
        fails.append("growth 未反映 %s" % desc)

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("④ 不确定性是知识的一部分")
print("=" * 92)
c_lo = K.confidence(s, 0)
c_hi = K.confidence(s, 2)
print("  confidence(动作0 低噪声) = %.4f" % c_lo)
print("  confidence(动作2 高噪声) = %.4f" % c_hi)
print("  -> 高噪声动作的 confidence 更低: %s" % ("✓" if c_hi < c_lo else "✗"))
if not c_hi < c_lo:
    fails.append("confidence 未区分高低噪声")

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("⑤ 可塑性是**元知识** (与 w 分开统计)")
print("=" * 92)
#   ★ 必须**闭环**: α 的稳定性来自「δ 随 w 收敛而趋 0」。
#   早先版本把 α 的更新从它要优化的学习里孤立出来 (δ 恒 ~1), 结果 8 种
#   稳定化手段**全部**撞界 —— 因为 g = δ·φ·h, 迹稳态 h* = δ/φ, 于是
#   g ≈ δ²/φ **恒为正**, β 单向漂移, 停不下来。那不是模块的错, 是测试
#   问错了问题: 孤立地跑元更新, 就永远没有收敛信号。
#   闭环 = 用一个**稀疏 + 非平稳**的在线回归: 只有 w[0] 真的能用 x[0] 预测 y,
#   其余 5 个分量是纯噪声。看 α 能否把「该快学」和「不该学」分开。
dim, n_feat = 1, 6
pl = Plasticity(n_feat, alpha0=0.05, mu=0.02)
rng5 = np.random.default_rng(0)
w_true = np.zeros(n_feat); w_true[0] = 3.0
w_hat = np.zeros(n_feat)
rms = 1.0
hist = []
for t in range(4000):
    x = rng5.normal(0, 1.0, n_feat)
    y = float(w_true @ x)
    y_hat = float(w_hat @ x)
    delta = y - y_hat
    rms = 0.99 * rms + 0.01 * delta ** 2                 # δ 的 RMS 归一化
    dn = delta / max(np.sqrt(rms), 1e-8)
    a = pl.alpha
    w_hat += a * dn * x                                  # ★ 用 α 真的去学
    pl.update(x, dn)                                     # α 的元更新
    if t % 500 == 0:
        hist.append((t, float(pl.alpha[0]), float(np.mean(pl.alpha[1:])), float(np.linalg.norm(delta))))

st = pl.stats()
print("  闭环 4000 步 (只 w[0] 可预测, 其余 5 维纯噪声):")
print("    步     α[0]      α[1:]均值    |δ|")
for t, a0, ar, dl in hist:
    print("   %5d  %8.4f  %10.4f  %8.4f" % (t, a0, ar, dl))
print("  α: mean=%.4f std=%.4f ratio_max_min=%.2f" % (st["mean"], st["std"], st["ratio_max_min"]))
print("  β 撞界的分量数 = %d / %d" % (int((np.abs(pl.beta) >= 19.99).sum()), n_feat))
sep = pl.alpha[0] > np.mean(pl.alpha[1:])
bounded = bool(np.isfinite(pl.alpha).all() and (np.abs(pl.beta) < 19.99).all())
print("  -> α[0] (真特征) 高于噪声分量: %s" % ("✓" if sep else "✗"))
print("  -> β 全程有界 (无撞界):      %s" % ("✓" if bounded else "✗"))
if not sep:
    fails.append("α 未在真特征上分化")
if not bounded:
    fails.append("β 撞界 (元知识发散)")
# 元知识与知识分开: Plasticity 里没有 w
print("  元知识对象字段 = %s  (无 w, 与知识分离)"
      % [a for a in vars(pl) if not a.startswith("_")])

# ──────────────────────────────────────────────────────────────────────
print()
print("=" * 92)
print("端到端: agent 对世界的一句话自述")
print("=" * 92)
for a in range(N_DOM):
    k = K.knowledge_of(s, a)
    print("  动作%d: 我预计 -> %s   (覆盖 %.0f 次, 置信 %.2f, V=%.2f)"
          % (a, k["predicted_next"], k["coverage"], k["confidence"], k["value"]))

print()
if fails:
    print("✗ 失败项: %s" % "; ".join(fails))
    sys.exit(1)
print("✓ 全部通过")
