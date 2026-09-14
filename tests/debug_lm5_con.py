#!/usr/bin/env python3
"""调试 lm5 的 wave 自洽性为什么恒为 1.0。"""
import sys
import numpy as np

sys.path.insert(0, "/work/liuyuanjie/headless")
sys.path.insert(0, "/work/liuyuanjie/headless/tests")
from run_lm4_wave import build_feature_matrix, build_windows  # noqa: E402

times, dens, Fm, ok = build_feature_matrix("/work/liuyuanjie/headless/data/wave",
                                           bands=13, use_wfr=True, use_lshell=False)
X, y_log = build_windows(times, dens, Fm, ok)
print("wave_X:", X.shape, X.dtype)
Xw = np.asarray(X, dtype=np.float64)
if Xw.ndim == 3:
    Xw = Xw.mean(axis=1)
nf = Xw.shape[1]
print("均值后:", Xw.shape, "nf =", nf)
for i, j in [(0, 1), (0, nf - 2), (1, nf - 2)]:
    a, b = Xw[:, i], Xw[:, j]
    print("  (%d,%d): std_a=%.4g std_b=%.4g  |c|=%s" %
          (i, j, a.std(), b.std(),
           round(abs(np.corrcoef(a, b)[0, 1]), 4) if a.std() > 1e-9 and b.std() > 1e-9 else "NaN"))
