#!/usr/bin/env python3
"""关键检验: WFR 波谱携带的是 |B| 之外的独立信息吗?"""
import numpy as np

z = np.load("/tmp/lm4_feat_cache.npz")
dens, F = z["dens"], z["F"]
ok = np.isfinite(dens) & (dens > 0) & np.isfinite(F).all(axis=1)
dens, F = dens[ok], F[ok]
y = np.log10(dens)
qs = np.quantile(y, np.linspace(0, 1, 7)); qs[0] -= 1e-6; qs[-1] += 1e-6
dom = np.digitize(y, qs[1:-1])

BANDS = 13
FREQ = [6.4, 17.1, 28.6, 50.2, 90.7, 161.0, 285.8, 508.4, 903.4, 1610.0, 2862.0, 5086.0, 9036.0]
logMag = F[:, 0]
b285 = F[:, 4 + 6]     # B@285.8Hz (最强波频段)
b161 = F[:, 4 + 5]
rms = F[:, 1]


def auc(scores, labels):
    o = np.argsort(scores, kind="mergesort")
    r = np.empty(len(scores)); r[o] = np.arange(1, len(scores) + 1)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


m = (dom == 2) | (dom == 3)
lab = (dom[m] == 3).astype(int)
d0 = (dom == 0); d5 = (dom == 5)
m05 = d0 | d5
lab05 = dom[m05] == 5
lab05 = lab05.astype(int)

print("=== 1. 相关性 (与 logMag 有多冗余?) ===")
for name, v in [("log rms", rms), ("B@161Hz", b161), ("B@285.8Hz", b285)]:
    r = float(np.corrcoef(v, logMag)[0, 1])
    print(f"  corr({name:<10}, logMag) = {r:+.3f}")

print("\n=== 2. 单特征 AUC ===")
print(f"{'特征':<22}{'D2vsD3':>10}{'D0vsD5':>10}")
for name, v in [("logMag", logMag), ("log rms", rms), ("B@285.8Hz", b285), ("B@161Hz", b161)]:
    print(f"  {name:<20}{auc(v[m], lab):>10.4f}{auc(v[m05], lab05):>10.4f}")

def lda_auc(X, lab):
    """多维线性判别 (类均值方向) 的 AUC"""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    w = Xn[lab == 1].mean(0) - Xn[lab == 0].mean(0)
    return auc(Xn @ w, lab)


print("\n=== 3. 残差化: 去掉 |B| 的线性贡献后, 波谱还剩多少区分力? ===")
print(f"  (a) MAG 4 维 (线性判别) D2vsD3 AUC = {lda_auc(F[m][:, :4], lab):.4f}")
print(f"      WFR 磁 13 维          D2vsD3 AUC = {lda_auc(F[m][:, 4:4+BANDS], lab):.4f}")

# 用最小二乘把 b285 的 |B| 成分去掉
A = np.vstack([logMag, np.ones_like(logMag)]).T
coef, *_ = np.linalg.lstsq(A, b285, rcond=None)
resid = b285 - A @ coef
print(f"  (b) B@285.8Hz 对 |B| 回归的 R² = {1 - resid.var()/b285.var():.3f}")
print(f"      → 残差(resid) 的 D2vsD3 AUC = {auc(resid[m], lab):.4f}   (若≈0.5 则纯属 |B| 的影子)")
print(f"      → 原始 B@285.8Hz 的 AUC      = {auc(b285[m], lab):.4f}")

# 反向: logMag 去掉 b285 后还剩多少
A2 = np.vstack([b285, np.ones_like(b285)]).T
c2, *_ = np.linalg.lstsq(A2, logMag, rcond=None)
resid2 = logMag - A2 @ c2
print(f"  (c) logMag 去掉 B@285.8Hz 后 D2vsD3 AUC = {auc(resid2[m], lab):.4f}  (原始 {auc(logMag[m], lab):.4f})")

print("\n=== 4. 频段\"有效性\"排名 (D2vsD3 信息量, 按 |AUC-0.5|) ===")
vals = []
for i, f in enumerate(FREQ):
    vals.append((f"B@{f:g}Hz", auc(F[m][:, 4 + i], lab)))
    vals.append((f"E@{f:g}Hz", auc(F[m][:, 4 + BANDS + i], lab)))
vals.sort(key=lambda t: -abs(t[1] - 0.5))
print("  最有信息:")
for n_, a in vals[:6]:
    print(f"    {n_:<14} AUC={a:.4f}")
print("  最没用(接近 0.5):")
for n_, a in vals[-6:]:
    print(f"    {n_:<14} AUC={a:.4f}")
