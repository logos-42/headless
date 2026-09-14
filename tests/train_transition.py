#!/usr/bin/env python3
"""从现有 acc_matrix 学 transition model, 并检验 planning 是否有效。

用法:
    python tests/train_transition.py [ROOT]

回答两个问题:
  Q1 转移可学吗?  MAE 必须 < copy 基线 (0.2986)。按 seed 留出。
  Q2 规划有用吗?  用 T 做 beam search 选动作序列, 预测末端准确率
                  是否高于 fixed / random 序列?
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hibs_lnn.transition_model import TransitionModel  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/work/liuyuanjie/headless"

# ── 载入: 只取动作可精确恢复的 stream=fixed ──
recs = []
for f in sorted(glob.glob(os.path.join(ROOT, "results/**/lm4_wave_results.json"),
                          recursive=True)):
    tag = os.path.basename(os.path.dirname(f))
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    for arm, v in d.items():
        if not isinstance(v, dict) or "acc_matrix" not in v:
            continue
        if v.get("stream") != "fixed":
            continue
        M = np.asarray(v["acc_matrix"], dtype=float)
        n_dom = M.shape[1]
        n_rounds = M.shape[0]
        # stream=fixed -> dom_seq = range(n_dom) 循环续接 (run_lm4_wave.py:654)
        dom_seq = [i % n_dom for i in range(n_rounds)]
        recs.append((tag, arm, M, dom_seq))

print("stream=fixed 记录数:", len(recs))
if not recs:
    sys.exit("没有可用记录")

# ── 按 seed 留出 (tag 里的 _sN) ──
def seed_of(tag):
    for part in tag.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:])
    return None

seeds = sorted({s for s in (seed_of(t) for t, _, _, _ in recs) if s is not None})
print("发现的 seed:", seeds)
if len(seeds) >= 2:
    test_seed, train_seeds = seeds[-1], seeds[:-1]
    tr = [(M, q) for t, a, M, q in recs if seed_of(t) in train_seeds]
    te = [(M, q) for t, a, M, q in recs if seed_of(t) == test_seed]
else:
    tr = [(M, q) for _, _, M, q in recs]
    te = tr
    test_seed = None
print("训练记录 %d / 测试记录 %d (留出 seed=%s)" % (len(tr), len(te), test_seed))
print()

tm = TransitionModel(lam=1e-2).fit(tr)

# ── Q1: 转移可学吗 ──
Xte, Yte, _ = tm.build_xy(te)
if len(Xte) == 0:
    sys.exit("测试集为空")
pred = np.array([tm.predict_delta(Xte[i, :tm.n_dom], int(np.argmax(Xte[i, tm.n_dom:2*tm.n_dom])),
                                 Xte[i, 2*tm.n_dom:3*tm.n_dom], Xte[i, -1])
                 for i in range(len(Xte))])
mae_model = float(np.mean(np.abs(pred - Yte)))
mae_copy = float(np.mean(np.abs(Yte)))                     # Δa ≡ 0
mae_const = float(np.mean(np.abs(Yte - Yte.mean(0, keepdims=True))))
print("=== Q1 转移可学吗 (测试集 MAE, 越小越好) ===")
print("  copy 基线  Δa≡0            %.4f" % mae_copy)
print("  constant   Δa≡mean         %.4f" % mae_const)
print("  transition model (ridge)   %.4f   %s"
      % (mae_model, "<- 打赢 copy ✓" if mae_model < mae_copy else "<- **没打赢 copy ✗**"))
print()

# ── Q2: 规划有用吗 ──
print("=== Q2 规划: 用 T 前向搜索 vs 固定/随机序列 (预测末端准确率) ===")
M0 = te[0][0]
a0 = M0[0]
horizon = M0.shape[0] - 1
plans = {"fixed": [i % tm.n_dom for i in range(horizon)],
         "planned(T)": tm.plan(a0, horizon, beam=8)}
rng = np.random.RandomState(0)
plans["random(mean of 20)"] = None
for name, seq in plans.items():
    if seq is None:
        vals = []
        for _ in range(20):
            q = list(rng.randint(0, tm.n_dom, size=horizon))
            a, nv = a0.copy(), np.zeros(tm.n_dom)
            for t, u in enumerate(q):
                a = a + tm.predict_delta(a, u, nv, t / horizon); nv[u] += 1
            vals.append(float(np.nanmean(a)))
        print("  %-18s %.4f  (seq 随机)" % (name, np.mean(vals)))
    else:
        a, nv = a0.copy(), np.zeros(tm.n_dom)
        for t, u in enumerate(seq):
            a = a + tm.predict_delta(a, u, nv, t / horizon); nv[u] += 1
        print("  %-18s %.4f  seq=%s" % (name, float(np.nanmean(a)), seq))
