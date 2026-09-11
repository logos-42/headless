#!/usr/bin/env python3
"""LM4 数据分析: 为什么 WFR 能救活 D2/D3? 各特征/频段携带多少密度信息?"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_lm4_wave import build_feature_matrix

CACHE = "/tmp/lm4_feat_cache.npz"
BANDS = 13
FREQ = [6.4, 17.1, 28.6, 50.2, 90.7, 161.0, 285.8, 508.4, 903.4, 1610.0, 2862.0, 5086.0, 9036.0]
MAG_NAMES = ["logMag", "log rms", "delta", "lambda"]

if os.path.exists(CACHE):
    z = np.load(CACHE)
    dens, F = z["dens"], z["F"]
    print(f"[cache] 载入 {F.shape}")
else:
    t, dens, F, ok = build_feature_matrix("data/wave", bands=BANDS, use_wfr=True)
    dens, F = dens[ok], F[ok]
    np.savez_compressed(CACHE, dens=dens, F=F)
    print(f"[built] {F.shape}")

valid = np.isfinite(dens) & (dens > 0) & np.isfinite(F).all(axis=1)
dens, F = dens[valid], F[valid]
y = np.log10(dens)
print(f"样本 {len(y)}, 特征 {F.shape[1]} 维")

# 6 个分位数域
qs = np.quantile(y, np.linspace(0, 1, 7)); qs[0] -= 1e-6; qs[-1] += 1e-6
dom = np.digitize(y, qs[1:-1])

names = MAG_NAMES + [f"B@{f:g}Hz" for f in FREQ] + [f"E@{f:g}Hz" for f in FREQ]
assert len(names) == F.shape[1], (len(names), F.shape)

# ---------- A. 单特征判别力: ANOVA F 统计量 ----------
print("\n=== A. 单特征判别力 (跨 6 个密度域, F 越大越有信息) ===")
Fs, MI = [], []
for j in range(F.shape[1]):
    v = F[:, j]
    gm = v.mean()
    ssb = sum(len(v[dom == d]) * (v[dom == d].mean() - gm) ** 2
              for d in range(6) if (dom == d).sum() > 0)
    ssw = sum(((v[dom == d] - v[dom == d].mean()) ** 2).sum()
              for d in range(6) if (dom == d).sum() > 0)
    dfb, dfw = 5, len(v) - 6
    Fs.append((ssb / dfb) / (ssw / dfw + 1e-30))
    # MI (20 bins)
    hb = np.histogram(v, bins=20)[0].astype(float); hb /= hb.sum() + 1e-30
    hd = np.bincount(dom, minlength=6).astype(float); hd /= hd.sum()
    H_d = -(hd[hd > 0] * np.log(hd[hd > 0])).sum()
    H_dv = 0.0
    for d in range(6):
        m = dom == d
        if m.sum() == 0: continue
        h2 = np.histogram(v[m], bins=20)[0].astype(float); h2 /= h2.sum() + 1e-30
        H_dv += hd[d] * (-(h2[h2 > 0] * np.log(h2[h2 > 0])).sum())
    MI.append(H_d - H_dv)

order = np.argsort(Fs)[::-1]
print(f"{'feature':<14}{'F':>12}{'MI':>8}")
for j in order[:14]:
    print(f"{names[j]:<14}{Fs[j]:>12.1f}{MI[j]:>8.4f}")
print("  ...")
print(f"{'最弱5个:':<14}" + ", ".join(f"{names[j]}({Fs[j]:.0f})" for j in order[-5:]))

# ---------- B. 各域的平均磁场波谱 (揭示物理机制) ----------
print("\n=== B. 各密度域的平均磁功率谱 (log10, 每格=该域该频段均值) ===")
bpow = F[:, 4:4 + BANDS]
print(f"{'freq(Hz)':>10}" + "".join(f"{'D'+str(d):>9}" for d in range(6)) + f"{'D5-D0':>9}")
for i, fr in enumerate(FREQ):
    row = [bpow[dom == d, i].mean() for d in range(6)]
    print(f"{fr:>10.1f}" + "".join(f"{v:>9.2f}" for v in row) + f"{row[5]-row[0]:>9.2f}")

# ---------- C. D2 vs D3 二分类可解性 ----------
def auc_score(scores, labels):
    """Mann-Whitney AUC. labels: 1=正类。"""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("\n=== C. D2 vs D3 二分类 (最难的相邻域) ===")
m23 = (dom == 2) | (dom == 3)
y23 = (dom[m23] == 3).astype(int)
print(f"  样本数: D2={int((dom==2).sum())}, D3={int((dom==3).sum())}")

for tag, cols in [("MAG 4 维", [0, 1, 2, 3]),
                  ("WFR 磁 13 维", list(range(4, 4 + BANDS))),
                  ("WFR 电 13 维", list(range(4 + BANDS, 4 + 2 * BANDS))),
                  ("全部 30 维", list(range(30)))]:
    Xs = F[m23][:, cols]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    Xn = (Xs - mu) / sd
    c0, c1 = Xn[y23 == 0].mean(0), Xn[y23 == 1].mean(0)
    w = c1 - c0
    proj = Xn @ w
    a = auc_score(proj, y23)
    print(f"  {tag:<14} 线性判别 AUC = {a:.4f}  (0.5=无信息)")

# ---------- D. 单个频段对 D2/D3 的区分力 ----------
print("\n=== D. 各频段单独区分 D2/D3 的能力 (AUC, 只列 |AUC-0.5|>0.05 的) ===")
rowsd = []
for j in range(F.shape[1]):
    a = auc_score(F[m23][:, j], y23)
    rowsd.append((names[j], a))
rowsd.sort(key=lambda t: -abs(t[1] - 0.5))
for n_, a in rowsd[:12]:
    print(f"  {n_:<14} AUC={a:.4f}  ({'高密度→更强' if a > 0.5 else '高密度→更弱'})")

# ---------- E. 波谱形状 vs 密度 (整体相关) ----------
print("\n=== E. 各频段功率 vs log10(density) 的相关系数 (只列 |r|>0.15) ===")
cor = []
for j in range(4, 4 + 2 * BANDS):
    v = F[:, j]
    if np.std(v) < 1e-9:
        continue
    r = float(np.corrcoef(v, y)[0, 1])
    cor.append((names[j], r))
cor.sort(key=lambda t: -abs(t[1]))
for n_, r in cor[:12]:
    print(f"  {n_:<14} r={r:+.3f}")

