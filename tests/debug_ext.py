#!/usr/bin/env python3
"""诊断外部测试集: 每个域的样本数 / 标签纯度 / 常数预测基线。"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/work/liuyuanjie/headless")
sys.path.insert(0, "/work/liuyuanjie/headless/tests")
from run_lm4_wave import (build_feature_matrix, build_windows, assign_domains)

# 训练年边界
t1, d1, F1, ok1 = build_feature_matrix("/work/liuyuanjie/headless/data/wave",
                                       bands=13, use_wfr=True, use_lshell=True)
X1, yl1 = build_windows(t1, d1, F1, ok1)
y1, qs = assign_domains(yl1, 6)
print("训练年 2015 边界:", [round(float(q), 2) for q in qs])
print("训练年 2015 每域样本:", np.bincount(y1, minlength=6).tolist())

# 测试年 (沿用边界)
t2, d2, F2, ok2 = build_feature_matrix("/work/liuyuanjie/headless/data/wave2016",
                                       bands=13, use_wfr=True, use_lshell=True)
X2, yl2 = build_windows(t2, d2, F2, ok2)
y2, _ = assign_domains(yl2, 6, qs=qs)
print("\n测试年 2016 每域样本 (沿用 2015 边界):", np.bincount(y2, minlength=6).tolist())
print("测试年 log10(density) 范围:", round(float(yl2.min()), 2), "~", round(float(yl2.max()), 2))
print("训练年 log10(density) 范围:", round(float(yl1.min()), 2), "~", round(float(yl1.max()), 2))

# 关键检查: 每个 dd 的子集是否只含 dd 标签 (当然), 以及"常数预测"能拿多少
print("\n每个域子集的样本数 (若某域样本极少, 其准确率噪声极大):")
for dd in range(6):
    n = int((y2 == dd).sum())
    print(f"  D{dd}: {n} 条")
