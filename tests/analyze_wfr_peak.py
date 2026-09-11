#!/usr/bin/env python3
"""WFR 峰值的物理性质: 峰值频率随 |B| 移动(共振) 还是固定(波种群)?"""
import numpy as np

z = np.load("/tmp/lm4_feat_cache.npz")
dens, F = z["dens"], z["F"]
ok = np.isfinite(dens) & (dens > 0) & np.isfinite(F).all(axis=1)
dens, F = dens[ok], F[ok]
y = np.log10(dens)
qs = np.quantile(y, np.linspace(0, 1, 7)); qs[0] -= 1e-6; qs[-1] += 1e-6
dom = np.digitize(y, qs[1:-1])

BANDS = 13
FREQ = np.array([6.4, 17.1, 28.6, 50.2, 90.7, 161.0, 285.8, 508.4, 903.4, 1610.0, 2862.0, 5086.0, 9036.0])
bpow = F[:, 4:4 + BANDS]
logMag = F[:, 0]

# 每个样本的谱峰频段 (扣除低频仪器底: 只看 >=90.7Hz 的 4~13 段)
sub = bpow[:, 4:]
peak = sub.argmax(axis=1) + 4
peak_f = FREQ[peak]

print("=== 1. 谱峰频段分布 (按密度域) ===")
print(f"{'域':<5}{'样本':>9}  " + "".join(f"{f:>7g}" for f in FREQ[4:]))
for d in range(6):
    m = dom == d
    cnt = np.bincount(peak[m], minlength=BANDS)[4:]
    pct = cnt / max(cnt.sum(), 1) * 100
    print(f"D{d:<4}{m.sum():>9}  " + "".join(f"{p:>7.1f}" for p in pct))

print("\n=== 2. 峰值频率 vs |B| ===")
lf = np.log10(peak_f)
r = float(np.corrcoef(lf, logMag)[0, 1])
print(f"  corr(log10 峰值频率, log10|B|) = {r:+.3f}")
print(f"  → {'随磁场移动 → 共振特征(下杂波/离子回旋)' if abs(r) > 0.3 else '基本固定 → 波种群(嘶声带)'}")
print(f"  各域中位峰值频率: " + ", ".join(
    f"D{d}={np.median(peak_f[dom==d]):.0f}Hz" for d in range(6)))

print("\n=== 3. 下杂波频率参考 (flh ≈ 2.07 * B[nT]) ===")
for B in [50, 100, 200, 400, 800]:
    print(f"  |B|={B:>4} nT → flh ≈ {2.07*B:>7.0f} Hz")
print("  (与谱峰 161~508 Hz 对比)")
