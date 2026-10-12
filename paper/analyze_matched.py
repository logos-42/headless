#!/usr/bin/env python3
"""决定性重跑判定 —— 严格按 PREREGISTRATION.md 规则执行。

主指标: 收敛段 (round>=2000) paired warm 代价均值差 (fixed - fractal)
判定:   paired bootstrap 95% CI (n_boot=10000, seed 固定) 相对等价界 ±1 步
look 序列: n=5000 首次判定 -> 若跨边界补到 n=10000 -> 仍跨则"未决", 停止
          (数据虽跑满 200k 轮, 判定只用预注册窗口, 绝不在全量里挑窗口)
次指标: 越界率 / 树拓扑 / cold 基线 (~32 步验证 warm 有效)
"""
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

DATA = Path("/Users/apple/Downloads/headless/paper/oml_daemon_fractal_matched_20261012.jsonl")
BOOL = {True, False}
EPS = 1e-9


def load_rows():
    warm, cold = [], []
    with DATA.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert r["round"] >= 0
            entry = {
                "round": r["round"],
                "mu0": r["mu0"],
                "plain": r["plain_steps"],
                "fractal": r["fractal_steps"],
                "plain_over": r["plain_over"],
                "fractal_over": r["fractal_over"],
                "nodes": r.get("fractal_nodes"),
            }
            (cold if r["is_cold"] else warm).append(entry)
    return warm, cold


def paired_ci(diffs, n_boot=10000, seed=20261012, alpha=0.05):
    """paired bootstrap CI for mean diff (fixed - fractal)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = d[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


def decide(n, diffs, label):
    lo, hi = paired_ci(diffs[:n])
    mean = float(np.mean(diffs[:n]))
    delta = 1.0
    verdict = None
    if hi < -delta:
        verdict = "FIXED_BETTER"
    elif lo > delta:
        verdict = "FRACTAL_BETTER"
    elif lo > -delta and hi < delta:
        verdict = "EQUIVALENT"
    else:
        verdict = "INCONCLUSIVE"
    print(f"[{label}] n={n}  mean_diff(fixed-fractal)={mean:+.4f}  95% CI=({lo:+.4f}, {hi:+.4f})  -> {verdict}")
    return verdict


def main():
    warm, cold = load_rows()
    print(f"总轮: warm={len(warm)}  cold={len(cold)}\n")

    # 收敛段 (round >= 2000, 与审稿第1轮收敛窗口一致)
    conv = [e for e in warm if e["round"] >= 2000]
    conv.sort(key=lambda e: e["round"])
    diffs = [e["plain"] - e["fractal"] for e in conv]  # positive = fractal better
    print(f"收敛段 warm 样本: {len(conv)} (round>=2000)")

    # --- 判定 (预注册 look 序列) ---
    print("\n=== 主判定 ===")
    d5000 = diffs[:5000]
    v1 = decide(5000, diffs, "look#1")
    result = v1
    if v1 == "INCONCLUSIVE":
        v2 = decide(10000, diffs, "look#2 (上限)")
        result = v2 if v2 != "INCONCLUSIVE" else "UNDECIDED"
    print(f"\n最终判定: {result}")

    # --- 次指标 ---
    print("\n=== 次指标 ===")
    # 1. 越界率
    n_w = len(warm)
    po = sum(e["plain_over"] for e in warm) / n_w
    fo = sum(e["fractal_over"] for e in warm) / n_w
    print(f"越界率 (期望<=10%): fixed={po*100:.2f}%  fractal={fo*100:.2f}%")

    # 2. 生长拓扑
    nodes = sorted({e["nodes"] for e in warm if e["nodes"] is not None})
    print(f"fractal 树节点分布: {nodes}  (cap=7, 到达cap={max(nodes) == 7 if nodes else False})")

    # 3. cold 基线 (~32 步)
    cp = statistics.mean(e["plain"] for e in cold)
    cf = statistics.mean(e["fractal"] for e in cold)
    print(f"cold 基线步数: fixed={cp:.2f}  fractal={cf:.2f}  (期望 ~32)")

    # 4. warm 长期适应 (对比前1000 vs 收敛段)
    early = warm[:1000]
    ep = statistics.mean(e["plain"] for e in early)
    ef = statistics.mean(e["fractal"] for e in early)
    convp = statistics.mean(e["plain"] for e in conv)
    convf = statistics.mean(e["fractal"] for e in conv)
    print(f"warm 均值: 前1000轮 fixed={ep:.2f} fractal={ef:.2f} | 收敛段 fixed={convp:.2f} fractal={convf:.2f}")

    # 5. 配对差分布摘要 (n=10000 窗口内)
    print(f"\n配对差 (fixed-fractal) 收敛段: mean={statistics.mean(diffs):+.3f}  sd={statistics.stdev(diffs):.3f}  "
          f"p25={statistics.quantiles(diffs, n=4)[0]:+.1f}  p50={statistics.median(diffs):+.1f}  p75={statistics.quantiles(diffs, n=4)[2]:+.1f}")

    # 6. 时间平稳性: 分 4 段看均值 (防止均值漂移掩盖结论)
    print("\n=== 时间平稳性检查 (收敛段 4 段均值) ===")
    q = len(conv) // 4
    for k in range(4):
        seg = diffs[k * q:(k + 1) * q]
        segr = conv[k * q:(k + 1) * q]
        print(f"段{k+1} rounds[{segr[0]['round']}..{segr[-1]['round']}] n={len(seg)} mean_diff={statistics.mean(seg):+.3f}")


if __name__ == "__main__":
    main()