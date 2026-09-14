#!/usr/bin/env python3
"""检验 external_test 的构造: 未训练模型在 6 个域上应各得 ~1/6, 不应全 1.0。"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/work/liuyuanjie/headless")
sys.path.insert(0, "/work/liuyuanjie/headless/tests")
from run_lm4_wave import (build_feature_matrix, build_windows, assign_domains,
                          evaluate, StatMLP)

t1, d1, F1, ok1 = build_feature_matrix("/work/liuyuanjie/headless/data/wave",
                                       bands=13, use_wfr=True, use_lshell=True)
X1, yl1 = build_windows(t1, d1, F1, ok1)
y1, qs = assign_domains(yl1, 6)

t2, d2, F2, ok2 = build_feature_matrix("/work/liuyuanjie/headless/data/wave2016",
                                       bands=13, use_wfr=True, use_lshell=True)
X2, yl2 = build_windows(t2, d2, F2, ok2)
y2, _ = assign_domains(yl2, 6, qs=qs)
print("2016 每域样本:", np.bincount(y2, minlength=6).tolist())

dev = torch.device("cpu")
ext = {}
for dd in range(6):
    idx = np.where(y2 == dd)[0]
    _X = torch.from_numpy(X2[idx].astype(np.float32))
    _y = torch.from_numpy(y2[idx])
    ext[dd] = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(_X, _y), batch_size=512)

torch.manual_seed(0)
m = StatMLP(n_feat=X1.shape[-1], n_classes=6, hidden=64, n_layers=2,
            head_type="linear", norm="fixed")
m.eval()
print("\n未训练模型在各域外部测试集上的准确率 (应各 ~0.167):")
for dd in range(6):
    a = evaluate(m, ext[dd], dev)
    print(f"  D{dd}: {a:.4f}   (该域样本 {len(ext[dd].dataset)})")

# 再看一个常数预测器 (全判 5) 的表现
class Const(torch.nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x, agg=None):
        return torch.zeros(x.shape[0], 6).scatter_(1, torch.full((x.shape[0],1), self.c), 10.0)

cm = Const(5)
print("\n常数预测器 (全判 D5) 的表现 (D5 应 1.0, 其余 0.0):")
for dd in range(6):
    print(f"  D{dd}: {evaluate(cm, ext[dd], dev):.4f}")
