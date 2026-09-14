#!/usr/bin/env python3
"""Regime-shift 消融: 固定 alpha / IDBD / Continual-IDBD 谁在环境突变后适应更快。

按用户给的 §13 设计做 3x4 消融, 但**用我们自己的真实 regime shift**:
  阶段 A: 描述子第 0 维决定收益
  阶段 B: **第 1 维**决定收益  (环境突变, 最优映射整体改变)
  阶段 C: 回到 A            (测 recovery / 新旧环境反复切换)

记录 §13 要求的全部指标:
  MSE / max(alpha) / min(alpha) / median(alpha) / adaptation time / recovery time /
  forgetting / weight-flipping rate F = #{t: sign(dw_t) != sign(dw_{t-1})} / T

关键判据 (§13):
  "针对非平稳动力学设计可恢复塑性, 并证明它比固定 alpha / IDBD / TIDBD
   在 regime shift 下**更快适应且更少发散**"
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/apple/Downloads/headless")
from hibs_lnn.rl_proposer import RLProposer  # noqa: E402

ND, D = 60, 3
rs = np.random.RandomState(0)
desc = rs.randn(ND, D)
PHASE = 120                      # 每个阶段长度


def target_dim(t):
    """环境映射: A=[0], B=[1], C=[0]"""
    return 1 if PHASE <= t < 2 * PHASE else 0


def run(algo, mu=0.05, alpha0=0.2, seed=1, n_know=0):
    p = RLProposer(desc, tau=0.5, k=3, mu=mu, alpha0=alpha0, algo=algo,
                   seed=seed, n_know=n_know)
    T = 3 * PHASE
    err, al_hist, w_hist, recovered = [], [], [], None
    first_b_ok = None
    for t in range(T):
        d = target_dim(t)
        idx = p.act()
        r = 0.5 + 0.3 * np.tanh(desc[idx, d].mean())
        w_before = p.theta.copy()
        for i in idx:
            p.acc[i] = r
        p.any_time = r
        p.update(idx, r=(r - p.global_acc if t else r))
        p.global_acc = r
        err.append(abs(r - float(np.mean([p.value(i) for i in idx]))))
        al_hist.append(p.alpha.copy())
        w_hist.append(p.theta.copy())
        # 阶段 B 内第一次误差降到 0.15 以下 = 适应
        if PHASE <= t < 2 * PHASE and first_b_ok is None and err[-1] < 0.15:
            first_b_ok = t - PHASE
        # 阶段 C 内第一次恢复 = recovery time
        if t >= 2 * PHASE and recovered is None and err[-1] < 0.15:
            recovered = t - 2 * PHASE
    A = np.array(err)
    AL = np.array(al_hist)
    W = np.array(w_hist)
    # weight-flipping rate (§13 的定义)
    dw = np.diff(W, axis=0)
    flip = float(np.mean(np.sign(dw[1:]) != np.sign(dw[:-1]))) if len(dw) > 1 else 0.0
    return {
        "mse_A": float(A[:PHASE].mean()),
        "mse_B": float(A[PHASE:2 * PHASE].mean()),
        "mse_C": float(A[2 * PHASE:].mean()),
        "adapt_B": first_b_ok,
        "recover_C": recovered,
        "forget": float(A[2 * PHASE:2 * PHASE + 10].mean() - A[PHASE - 10:PHASE].mean()),
        "a_max": float(AL[-1].max()), "a_min": float(AL[-1].min()),
        "a_med": float(np.median(AL[-1])),
        "a_std": float(AL[-1].std()),
        "flip": flip,
        "recov_n": int(p.stats().get("n_recovery", 0)),
    }


# (显示名, 实际 algo, 参数)
ALGOS = [("fixed-alpha", "cidbd", dict(mu=0.0)),          # 固定 alpha 基线
         ("idbd-raw", "idbd-raw", dict(mu=0.05)),
         ("idbd", "idbd", dict(mu=0.05)),
         ("autostep", "autostep", dict(mu=0.05)),
         ("cidbd", "cidbd", dict(mu=0.05)),
         ("cidbd+k14", "cidbd", dict(mu=0.05, n_know=14))]

print("=" * 112)
print("Regime-shift 消融 (A=d0 -> B=d1 -> C=d0, 每段 %d 步; 3 seed 平均)" % PHASE)
print("=" * 112)
seeds = [1, 7, 42]
print("  %-12s %8s %8s %8s %9s %9s %8s %8s %8s %7s %6s"
      % ("algo", "MSE_A", "MSE_B", "MSE_C", "adapt_B", "recov_C", "forget",
         "a_max", "a_med", "flip", "recov"))
results = {}
for name, algo, kw in ALGOS:
    rl = [run(algo, seed=s, **kw) for s in seeds]
    agg = {k: float(np.mean([r[k] for r in rl if r[k] is not None]))
           for k in rl[0] if k != "adapt_B" and k != "recover_C"}
    ab = [r["adapt_B"] for r in rl if r["adapt_B"] is not None]
    rc = [r["recover_C"] for r in rl if r["recover_C"] is not None]
    results[name] = (agg, ab, rc)
    print("  %-12s %8.4f %8.4f %8.4f %9s %9s %8.4f %8.3f %8.4f %7.3f %6.0f"
          % (name, agg["mse_A"], agg["mse_B"], agg["mse_C"],
             ("%.0f" % np.mean(ab)) if ab else "未适应",
             ("%.0f" % np.mean(rc)) if rc else "未恢复",
             agg["forget"], agg["a_max"], agg["a_med"], agg["flip"],
             agg["recov_n"]))

print()
print("=== §13 判据: 环境突变后「更快适应 且 更少发散」 ===")
base = results["fixed-alpha"][0]
idbd = results["idbd-raw"][0]
cid = results["cidbd"][0]
print("  %-22s %10s %10s %8s" % ("对比", "MSE_B", "adapt_B", "forget"))
for nm, agg, ab, rc in [("fixed-alpha", *results["fixed-alpha"]),
                        ("IDBD (raw)", *results["idbd-raw"]),
                        ("Continual-IDBD", *results["cidbd"]),
                        ("CIDBD+内部知识14", *results["cidbd+k14"])]:
    print("  %-22s %10.4f %10s %8.4f"
          % (nm, agg["mse_B"], ("%.0f" % np.mean(ab)) if ab else "未适应", agg["forget"]))
print()
print("  CIDBD vs IDBD:  MSE_B %+.4f   forget %+.4f   a_med %+.4f"
      % (cid["mse_B"] - idbd["mse_B"], cid["forget"] - idbd["forget"],
         cid["a_med"] - idbd["a_med"]))
