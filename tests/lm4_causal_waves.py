#!/usr/bin/env python3
"""LM4 因果层 — 对电磁波结构做因果发现 + 干预预测 + 不变性检验。

变量集(从真实 RBSP-A 特征构建, 10 个):
    log_dens   等离子体密度 (状态量)
    logB       磁场强度
    mag_{lo,mid,hi}   磁强计 13 频段聚合成低/中/高
    ele_{lo,mid,hi}   电场 13 频段聚合成低/中/高
    logL       L-shell
    absmaglat  磁纬绝对值

三步:
  1. **发现**: PC 骨架 (条件互信息 CI 检验) → 哪些变量在给定其余变量后仍直接相关
  2. **干预**: 对方向已定向的边, 用后门调整估计 do(X) 的因果效应
     (真实空间数据无法真做干预 —— 用后门调整公式做观测等价估计)
  3. **不变性检验**(与持续学习的连接点):
     因果边应在**子总体间不变**(如低 L vs 高 L 区); 伪相关会漂移。
     这正是"学到的结构能否跨域保留"的可测量形式。

用法:
  python3 tests/lm4_causal_waves.py --data data/wave --sample 20000
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from run_lm4_wave import build_feature_matrix
from hibs_lnn.causal.discovery import ci_test, pc_skeleton
from hibs_lnn.causal.identification import causal_effect, backdoor_adjustment_set
from hibs_lnn.causal.dag import Graph


def build_variables(data_dir, bands=13, use_lshell=True, sample=20000, seed=0):
    """从真实特征矩阵构建因果变量集。"""
    t0 = time.time()
    times, dens, F, ok = build_feature_matrix(data_dir, bands=bands,
                                              use_wfr=True,
                                              use_lshell=use_lshell)
    print(f"[data] 特征 {F.shape}, {time.time()-t0:.0f}s", flush=True)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    dens = np.asarray(dens, dtype=float)
    good = np.isfinite(dens) & (dens > 0)
    F, dens = F[good], dens[good]
    n = len(F)

    # 13 段 → 低/中/高 三段均值
    mag = F[:, 4:4 + bands]
    ele = F[:, 4 + bands:4 + 2 * bands]
    k = bands // 3
    agg_mag = [mag[:, :k].mean(1), mag[:, k:2 * k].mean(1), mag[:, -k:].mean(1)]
    agg_ele = [ele[:, :k].mean(1), ele[:, k:2 * k].mean(1), ele[:, -k:].mean(1)]

    cols, names = [], []
    cols.append(np.log10(np.clip(dens, 1e-4, None))); names.append("log_dens")
    cols.append(F[:, 0]); names.append("logB")
    for nm, c in zip(["mag_lo", "mag_mid", "mag_hi"], agg_mag):
        cols.append(c); names.append(nm)
    for nm, c in zip(["ele_lo", "ele_mid", "ele_hi"], agg_ele):
        cols.append(c); names.append(nm)
    if use_lshell and F.shape[1] >= 4 + 2 * bands + 2:
        cols.append(F[:, 4 + 2 * bands]); names.append("logL")
        cols.append(np.abs(F[:, 4 + 2 * bands + 1])); names.append("absmaglat")

    X = np.stack(cols, axis=1).astype(float)
    rng = np.random.default_rng(seed)
    if sample and len(X) > sample:
        idx = rng.choice(len(X), sample, replace=False)
        X = X[idx]
    print(f"[vars] {names}", flush=True)
    print(f"[vars] X {X.shape}, 有效点 {n}", flush=True)
    return X, names


def discover(X, names, alpha=0.3, max_order=2, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    t_c = time.time()
    adj = pc_skeleton(X, names, alpha=alpha, max_order=max_order, rng=rng)
    print(f"  [pc] 骨架完成 {time.time()-t_c:.0f}s", flush=True)
    edges = [(names[i], names[j]) for (i, j), v in adj.items() if v]
    print(f"\n=== PC 骨架 {tag} (alpha={alpha}, order<={max_order}, {time.time()-t0:.0f}s) ===")
    print(f"  变量 {len(names)} 个, 边 {len(edges)} 条 (完全图 {len(names)*(len(names)-1)//2} 条)")
    for a, b in edges:
        print(f"    {a} — {b}")
    return adj, edges


def invariance(X, names, adj, n_splits=2, split_on="logL", alpha=0.3,
               max_order=2, seed=0):
    """不变性检验: 因果边应在子总体间稳定, 伪相关会漂移。

    按 split_on 变量分位数切成 n_splits 段, 在每段上重跑发现, 比较边集。
    """
    j = names.index(split_on)
    qs = np.quantile(X[:, j], np.linspace(0, 1, n_splits + 1))
    subsets = []
    for s in range(n_splits):
        m = (X[:, j] >= qs[s]) & (X[:, j] <= qs[s + 1])
        if m.sum() < 500:
            continue
        subsets.append((s, X[m]))
    print(f"\n=== 不变性检验 (按 {split_on} 切 {len(subsets)} 段) ===")
    adj_by = {}
    for s, Xs in subsets:
        a, e = discover(Xs, names, alpha=alpha, max_order=max_order,
                        seed=seed + s, tag=f"[{split_on} 段{s}]")
        adj_by[s] = {(names[i], names[j_]) if i < j_ else (names[j_], names[i])
                     for (i, j_), v in a.items() if v}
    # 在全部段中都出现的边 = 稳定边
    common = set.intersection(*adj_by.values()) if adj_by else set()
    union = set.union(*adj_by.values()) if adj_by else set()
    print(f"\n  稳定边 (所有段都有, {len(common)} 条):")
    for a_, b_ in sorted(common):
        print(f"    ✓ {a_} — {b_}")
    print(f"  漂移边 (仅在部分段出现, {len(union-common)} 条):")
    for a_, b_ in sorted(union - common):
        where = [s for s in adj_by if (a_, b_) in adj_by[s]]
        print(f"    ✗ {a_} — {b_}   (仅段 {where})")
    return common, union, adj_by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "wave"))
    ap.add_argument("--bands", type=int, default=13)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--max-order", type=int, default=2)
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--split-on", default="logL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "lm4_causal"))
    args = ap.parse_args()

    X, names = build_variables(args.data, bands=args.bands,
                               sample=args.sample, seed=args.seed)
    adj, edges = discover(X, names, alpha=args.alpha,
                          max_order=args.max_order, seed=args.seed, tag="[全体]")
    common, union, adj_by = invariance(
        X, names, adj, n_splits=args.splits, split_on=args.split_on,
        alpha=args.alpha, max_order=args.max_order, seed=args.seed)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    rep = {"vars": names, "edges": [list(e) for e in edges],
           "stable": [list(e) for e in sorted(common)],
           "drifting": [list(e) for e in sorted(union - common)],
           "config": vars(args)}
    Path(args.out, "causal_waves.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\n报告: {args.out}/causal_waves.json")


if __name__ == "__main__":
    main()
