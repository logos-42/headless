#!/usr/bin/env python3
"""天花板归因: 是数据限制还是模型限制?

用非序列强基线 (窗口聚合特征 + MLP) 探 joint 天花板:
  - 若聚合基线 ≈ SSM 的 joint (0.70) → 天花板由【数据/标签】决定, 扫 SSM 容量无意义
  - 若聚合基线明显更高         → SSM 容量是瓶颈, 值得扫
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_lm4_wave import build_feature_matrix, build_windows, assign_domains

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/wave")
ap.add_argument("--bands", type=int, default=13)
ap.add_argument("--paths", default="wfr,lshell", help="逗号分隔: wfr / lshell")
ap.add_argument("--domains", type=int, default=6)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

use_wfr = "wfr" in args.paths
use_ls = "lshell" in args.paths

t0 = time.time()
times, dens, F, ok = build_feature_matrix(args.data, bands=args.bands,
                                          use_wfr=use_wfr, use_lshell=use_ls)
X, y_log = build_windows(times, dens, F, ok)
y_dom, qs = assign_domains(y_log, args.domains)
print(f"[data] {X.shape} 窗口, {time.time()-t0:.0f}s")

# 窗口聚合: mean/std/last/first/last-first(斜率)
agg = np.concatenate([X.mean(1), X.std(1), X[:, -1, :], X[:, 0, :],
                      X[:, -1, :] - X[:, 0, :]], axis=1).astype(np.float32)
print(f"[agg] {agg.shape[1]} 维聚合特征")

torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
idx = rng.permutation(len(agg))
n_te = len(idx) // 5
te, tr = idx[:n_te], idx[n_te:]
mu, sd = agg[tr].mean(0), agg[tr].std(0) + 1e-6
ftr = torch.tensor((agg[tr] - mu) / sd)
fte = torch.tensor((agg[te] - mu) / sd)
ytr = torch.tensor(y_dom[tr]); yte = torch.tensor(y_dom[te])


def run_mlp(hidden, layers, steps, tag):
    torch.manual_seed(args.seed)
    mods = [nn.Linear(ftr.shape[1], hidden), nn.ReLU()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods += [nn.Linear(hidden, args.domains)]
    m = nn.Sequential(*mods)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = len(ftr)
    for _ in range(steps):
        b = torch.randint(0, n, (128,))
        loss = nn.functional.cross_entropy(m(ftr[b]), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (m(fte).argmax(-1) == yte).float().mean().item()
    print(f"  {tag:<28} acc {acc:.4f}")
    return acc


print(f"\n=== 聚合基线 (非序列, 随机基线 {1/args.domains:.4f}) ===")
best = 0.0
for hidden, layers in [(128, 2), (512, 3), (2048, 3)]:
    best = max(best, run_mlp(hidden, layers, args.steps, f"MLP h{hidden} x{layers}"))

print(f"\n=== 对照 (SSM, 序列输入) ===")
print(f"  WaveSSM d128 l2 (joint)       acc 0.7213  [已知]")
print(f"\n判定: 聚合基线 {best:.4f} vs SSM 0.7213")
print(f"  → {'数据/标签是瓶颈 (扫 SSM 容量意义有限)' if best >= 0.70 else 'SSM 容量可能是瓶颈 (值得扫容量)'}")
