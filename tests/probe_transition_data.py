#!/usr/bin/env python3
"""摸清现有 155 份 acc_matrix 能不能支撑 transition model 的训练。

OaK feature #3 需要: feature → subproblem → solution → **transition model** → planning
我们把 lm4 的"能力状态"定义为 acc_matrix 的一行 (在 6 个域上的准确率画像),
那么转移模型就是:

    T_θ( a_t , u_t ) -> a_{t+1}

    a_t   = acc_matrix[t]      (能力画像, 6 维)
    u_t   = 第 t 步训练了哪个域 (动作)
    a_{t+1}= acc_matrix[t+1]

本脚本检查: (1) 有哪些 arm 存了 acc_matrix; (2) 动作 u_t 能否恢复;
(3) 转移的"可预测性"下界 (用 copy 基线 a_{t+1}=a_t 衡量)。
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/work/liuyuanjie/headless"
fs = sorted(glob.glob(os.path.join(ROOT, "results/**/lm4_wave_results.json"),
                      recursive=True))

recs = []
for f in fs:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    tag = os.path.basename(os.path.dirname(f))
    for arm, v in d.items():
        if isinstance(v, dict) and "acc_matrix" in v:
            recs.append((tag, arm, v))

print("记录数:", len(recs))
print()

# 按 arm 名统计
from collections import Counter
c = Counter(a for _, a, _ in recs)
print("arm 分布 (前 12):")
for k, n in c.most_common(12):
    print("   %-22s %3d" % (k, n))
print()

# 检查一个记录的完整字段
tag, arm, v = recs[0]
print("样例 %s / %s 的字段:" % (tag, arm))
for k in sorted(v.keys()):
    val = v[k]
    if isinstance(val, list):
        print("   %-26s list[%d]" % (k, len(val)))
    else:
        print("   %-26s %s" % (k, type(val).__name__))
print()

# 动作能否恢复? 找 schedule / stream / domains 之类
has_sched = sum(1 for _, _, v in recs if any(
    k in v for k in ("schedule", "stream", "actions", "domains")))
print("含 schedule/stream/actions/domains 的记录: %d / %d" % (has_sched, len(recs)))
print()

# 转移的可预测性下界: copy 基线
errs_copy, errs_mean = [], []
for tag, arm, v in recs[:60]:
    M = np.array(v["acc_matrix"], dtype=float)
    if M.ndim != 2 or M.shape[0] < 3:
        continue
    # a_{t+1} vs a_t (copy)  vs 全局均值 (常数基线)
    ok = ~np.isnan(M)
    d_copy = np.abs(M[1:] - M[:-1])
    errs_copy.append(np.nanmean(d_copy))
    mu = np.nanmean(M, axis=0, keepdims=True)
    errs_mean.append(np.nanmean(np.abs(M[1:] - mu)))

print("=== 转移可预测性下界 (越小越好) ===")
print("  copy 基线  |a_{t+1} - a_t|        = %.4f" % np.mean(errs_copy))
print("  常数基线   |a_{t+1} - mean(a)|    = %.4f" % np.mean(errs_mean))
print("  => 转移模型的 MAE 必须 < %.4f 才算学到东西" % np.mean(errs_copy))
