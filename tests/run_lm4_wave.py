#!/usr/bin/env python3
"""LM4 — 持续学习的电磁波模型 (Van Allen Probes RBSP-A)

数据: NASA CDAWeb EMFISIS
  - DENSITY L4: density [cm^-3]  → 物理状态标签(域)
  - MAG L3:     Magnitude/rms/lambda/delta → 独立电磁波特征
    (刻意排除 fpe/fuh/wpe_over_wce: 与 density 有解析关系 = 标签泄漏)

任务: 从电磁波/磁场特征序列推断等离子体物理状态(密度域)
持续学习: 顺序学习 6 个密度域(D1→D6), 测遗忘
对比: naive sequential vs +Replay

用法:
  python3 tests/run_lm4_wave.py --data data/wave --epochs-per-domain 300
"""
import os, sys, json, math, time, random, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from hibs_lnn.ssm_v30_3 import SSM_Layer_V30_3

WINDOW = 32
N_FEAT = 4
EPS = 1e-9
MAG_SENTINEL_LO = -1e4      # MAG L3 用 -100000 标记无效采样 (实测占 0.2%)
                            # 注意: 旧版用 np.abs() 会把哨兵伪装成 log10(1e5)=5.0 混进特征


# ============================================================
# 模型: 连续输入 SSM + 域分类头
# ============================================================
class WaveSSM(nn.Module):
    """连续特征序列 → SSM → 物理状态(密度域)分类。"""

    def __init__(self, n_feat=N_FEAT, d_model=128, d_state=8, n_layers=2,
                 n_classes=6):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        self.layers = nn.ModuleList([
            SSM_Layer_V30_3(d_model, d_state, layer_idx=i, ent_mode='none')
            for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """x: (B, L, n_feat) → logits (B, n_classes)  (取末时刻)"""
        h = self.in_proj(x)
        for layer in self.layers:
            h, _ = layer(h)
        h = self.norm(h)
        return self.head(h[:, -1])


# ============================================================
# 数据
# ============================================================
def _parse_time(s):
    """'2015-01-02T00:00:00.795946000Z' → epoch 秒"""
    from datetime import datetime, timezone
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".")
        frac = (frac + "000000")[:6]
        s = f"{head}.{frac}"
    else:
        s = s + ".000000"
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f").replace(
        tzinfo=timezone.utc).timestamp()


def _logsumexp(*arrs):
    """log10(10^a1 + 10^a2 + ...): 用于把 3 个分量功率合成总功率。
    NaN 视为无贡献; 全 NaN → NaN。"""
    stack = np.stack([np.where(np.isnan(x), -np.inf, x) for x in arrs])
    m = np.max(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sum(np.exp(stack - m), axis=0)
        out = m + np.log10(s)
    out[~np.isfinite(m)] = np.nan
    return out


def _band_reduce(x, bands):
    """频段轴降维: 65 → bands 段 (log 域均值 = 几何平均)"""
    N, B = x.shape
    if bands <= 0 or bands >= B:
        return x
    edges = np.linspace(0, B, bands + 1).astype(int)
    out = np.full((N, bands), np.nan, dtype=np.float32)
    for i in range(bands):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg = x[:, lo:hi]
        cnt = np.sum(~np.isnan(seg), axis=1)
        out[:, i] = np.where(cnt > 0, np.nansum(seg, axis=1) / np.maximum(cnt, 1), np.nan)
    return out


def _align(t_query, t_source, source_feats, tol=3.0):
    """把 source 近邻对齐到 t_query 时间轴 → (feats_sorted, idx, ok)"""
    try:
        q = np.array(t_query, dtype="datetime64[us]")
        s = np.array(t_source, dtype="datetime64[us]")
    except Exception:
        q = np.array([_parse_time(t) for t in t_query])
        s = np.array([_parse_time(t) for t in t_source])
    order = np.argsort(s)
    s = s[order]
    feats = source_feats[order]
    pos = np.clip(np.searchsorted(s, q), 0, len(s) - 1)
    left = np.clip(pos - 1, 0, len(s) - 1)
    dr = np.abs((s[pos] - q) / np.timedelta64(1, "s"))
    dl = np.abs((s[left] - q) / np.timedelta64(1, "s"))
    near = np.where(dl < dr, left, pos)
    ok = np.minimum(dl, dr) <= tol
    return feats, near, ok


def load_wfr_data(wave_dir, bands=13):
    """读 WFR 月度 npz, 逐月把 (N,6,65) 谱降成 2*bands 维。

    磁总功率 = log10(10^BuBu + 10^BvBv + 10^BwBw), 电总功率同理;
    再各自聚合到 bands 段。逐月处理并释放原始谱 (避免一次性载入 ~4GB)。"""
    files = sorted(Path(wave_dir).glob("wfr_*.npz"))
    T, F = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        psd = z["psd"].astype(np.float32)                 # (N,6,65) log10
        b = _logsumexp(psd[:, 0], psd[:, 1], psd[:, 2])
        e = _logsumexp(psd[:, 3], psd[:, 4], psd[:, 5])
        F.append(np.concatenate([_band_reduce(b, bands),
                                 _band_reduce(e, bands)], axis=1).astype(np.float32))
        T.extend(z["time"].tolist())
        del psd, b, e, z
    if not F:
        return [], np.zeros((0, 2 * bands), dtype=np.float32)
    tot = sum(len(x) for x in F)
    print(f"[wfr] {len(files)} 月, {tot} 条, {2*bands} 维特征")
    return T, np.concatenate(F, axis=0).astype(np.float32)


def build_feature_matrix(wave_dir, bands=13, use_wfr=True, tol=3.0):
    """统一特征管线: 以密度时间轴为准, 对齐 MAG + WFR。

    返回 (times, dens, feats (N,D), ok (N,))
      D = 4 (log10 Mag, log10 rms, lambda, delta) [+ 2*bands (WFR 磁/电总功率谱)]
    """
    d_dir = Path(wave_dir)
    dens_files = sorted(d_dir.glob("rbsp_a_*.npz"))
    if not dens_files:
        raise FileNotFoundError(f"无 DENSITY 数据: {d_dir}")

    # 密度: vals 列 = y,bmag,fce,fpe,wpe_over_wce,fuh,density (6=1e31 fill)
    t_all, dens_all = [], []
    for f in dens_files:
        z = np.load(f, allow_pickle=True)
        t_all.extend(z["time"].tolist())
        dens_all.append(z["vals"][:, 6])
    dens = np.concatenate(dens_all)
    dens = np.where(dens < -1e30, np.nan, dens)
    n = len(t_all)

    # MAG: feats = Magnitude, rms, delta, lambda
    mt, mf = [], []
    for f in sorted(d_dir.glob("mag_*.npz")):
        z = np.load(f, allow_pickle=True)
        mt.extend(z["time"].tolist())
        mf.append(z["feats"])
    mf = np.concatenate(mf) if mf else np.zeros((0, 4), dtype=np.float32)
    mag_f, mag_idx, mag_ok = _align(t_all, mt, mf, tol)
    print(f"[data] density={n} pts, mag={len(mt)} pts")

    m = mag_f[mag_idx].astype(np.float64)
    # 哨兵必须在变换前剔除, 否则 abs() 会把它变成 log10(1e5)=5.0 的"合法"特征
    mag_valid = np.isfinite(m).all(axis=1) & (m > MAG_SENTINEL_LO).all(axis=1)
    X = np.empty((n, 4), dtype=np.float32)
    X[:, 0] = np.log10(np.maximum(m[:, 0], EPS))
    X[:, 1] = np.log10(np.maximum(m[:, 1], EPS))
    X[:, 2] = m[:, 2]
    X[:, 3] = m[:, 3]
    ok = mag_ok & mag_valid
    n_bad = int((~mag_valid).sum())
    if n_bad:
        print(f"[data] MAG 剔除哨兵/无效采样: {n_bad} ({n_bad/n*100:.2f}%)")

    if use_wfr:
        wt, wf = load_wfr_data(wave_dir, bands)
        if len(wt):
            w_f, w_idx, w_ok = _align(t_all, wt, wf, tol)
            X = np.concatenate([X, w_f[w_idx]], axis=1).astype(np.float32)
            X[~w_ok, 4:] = 0.0          # 缺失填 0 (这些窗口会被 ok 掩码丢弃)
            ok &= w_ok

    ok &= np.isfinite(X).all(axis=1)
    X[~ok] = 0.0
    print(f"[data] 特征 {X.shape[1]} 维, 有效点 {int(ok.sum())}/{n}")
    return t_all, dens, X, ok


def build_windows(times, dens, feats, ok, window=WINDOW, stride=None):
    """构造 (特征窗口, 密度域标签)。

    stride 默认 = window (**不重叠**)。重叠窗口 + 随机切分会让测试窗口的
    近邻副本留在训练集(实测 100% 的测试窗口与最近训练窗口相距 <=8 索引,
    即时间上重叠 >=24/32 步) → 严重泄漏, 会同时抬高 acc 并虚增 replay 效果。

    feats/ok 由 build_feature_matrix 给出(已对齐到密度时间轴)。"""
    if stride is None:
        stride = window
    d_feat = feats.shape[1] if feats.ndim == 2 else 1
    n = len(times)
    if n == 0 or len(feats) == 0:
        return np.zeros((0, window, d_feat), dtype=np.float32), np.zeros((0,))
    X, y = [], []
    for i in range(0, n - window, stride):
        if not ok[i:i + window].all():
            continue
        d_end = dens[i + window - 1]
        if not np.isfinite(d_end) or d_end <= 0:
            continue
        X.append(feats[i:i + window])
        y.append(math.log10(d_end))
    if not X:
        return np.zeros((0, window, d_feat), dtype=np.float32), np.zeros((0,))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def assign_domains(y_log10, n_domains=6):
    """按 log10(density) 分位数划分域 (等样本量, 保证每域可训练)."""
    qs = np.quantile(y_log10, np.linspace(0, 1, n_domains + 1))
    qs[0] -= 1e-6
    qs[-1] += 1e-6
    lab = np.digitize(y_log10, qs[1:-1])
    return lab.astype(np.int64), qs


# ============================================================
# 持续学习实验
# ============================================================
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def run_experiment(X, y_dom, device, d_model=128, d_state=8, n_layers=2,
                   epochs_per_domain=300, batch=32, lr=1e-3, replay=False,
                   replay_ratio=0.3, seed=42, n_domains=6, n_feat=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 每域划分 train/test
    train_by_dom, test_by_dom = {}, {}
    for dd in range(n_domains):
        idx = np.where(y_dom == dd)[0]
        if len(idx) < 50:
            print(f"  [warn] 域 {dd} 样本少: {len(idx)}")
            continue
        rng = np.random.RandomState(seed + dd)
        rng.shuffle(idx)
        n_te = max(20, len(idx) // 10)
        test_by_dom[dd] = idx[:n_te]
        train_by_dom[dd] = idx[n_te:]
    domains = sorted(train_by_dom)

    def loader_for(idxs, shuffle=False):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X[idxs]), torch.from_numpy(y_dom[idxs]))
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

    test_loaders = {d: loader_for(test_by_dom[d]) for d in domains}
    n_classes = n_domains
    model = WaveSSM(n_feat=n_feat or X.shape[-1], d_model=d_model,
                    d_state=d_state, n_layers=n_layers,
                    n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    acc_matrix = []          # acc_matrix[step][domain]
    replay_by_dom = {}       # dd -> (X_dev, y_dev): 每域回放缓冲(预置 device)

    for step, dd in enumerate(domains):
        dl = loader_for(train_by_dom[dd], shuffle=True)
        model.train()
        it = 0
        while it < epochs_per_domain:
            for xb, yb in dl:
                if it >= epochs_per_domain:
                    break
                xb, yb = xb.to(device), yb.to(device)
                loss = F.cross_entropy(model(xb), yb)
                # Replay: 按域均衡混入旧域样本
                if replay and replay_by_dom:
                    k = max(1, int(batch * replay_ratio))
                    seen_doms = list(replay_by_dom)
                    per = max(1, k // len(seen_doms))
                    rxs, rys = [], []
                    for sd in seen_doms:
                        bx, by = replay_by_dom[sd]
                        m = min(per, len(bx))
                        sel = torch.randint(0, len(bx), (m,), device=device)
                        rxs.append(bx[sel]); rys.append(by[sel])
                    rx, ry = torch.cat(rxs), torch.cat(rys)
                    loss = loss + F.cross_entropy(model(rx), ry)
                opt.zero_grad(); loss.backward(); opt.step()
                it += 1
        # 记录: 已见域整体准确率
        row = [evaluate(model, test_loaders[d], device) if d in test_loaders else float('nan')
               for d in domains]
        acc_matrix.append(row)
        seen = [f"D{d}:{row[i]:.3f}" for i, d in enumerate(domains) if i <= step]
        print(f"  [{'replay' if replay else 'naive '}] step{step+1} (域{dd}) → {' '.join(seen)}", flush=True)
        # 存回放样本 (每域固定 2000, 预置 device 避免逐步搬运)
        if replay:
            idx = train_by_dom[dd]
            sel = np.random.RandomState(seed + step).choice(
                idx, size=min(2000, len(idx)), replace=False)
            replay_by_dom[dd] = (torch.from_numpy(X[sel]).to(device),
                                 torch.from_numpy(y_dom[sel]).to(device))
    return acc_matrix, domains


def summarize(acc_matrix, domains):
    """遗忘 + 平均准确率。遗忘 = 每个域历史最佳 - 最终。"""
    M = np.asarray(acc_matrix)
    finals = M[-1]
    forgets = []
    for j in range(M.shape[1]):
        col = M[:, j]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            forgets.append(col.max() - col[-1])
    return {
        "final_mean_acc": float(np.nanmean(finals)),
        "mean_forget": float(np.mean(forgets)) if forgets else float('nan'),
        "acc_matrix": M.tolist(),
        "domains": domains,
    }


def run_joint(X, y_dom, device, d_model=128, d_state=8, n_layers=2,
              steps=1500, batch=32, lr=1e-3, seed=42, n_domains=6, n_feat=None):
    """诊断: 所有域混合训练 (非持续学习上限)。若这个也学不好 → 特征/任务定义有问题。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr_idx, te_by_dom = [], {}
    for dd in range(n_domains):
        idx = np.where(y_dom == dd)[0]
        if len(idx) < 50:
            continue
        rng = np.random.RandomState(seed + dd)
        idx = idx.copy(); rng.shuffle(idx)
        n_te = max(20, len(idx) // 10)
        te_by_dom[dd] = idx[:n_te]
        tr_idx.extend(idx[n_te:])
    tr_idx = np.asarray(tr_idx)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X[tr_idx]), torch.from_numpy(y_dom[tr_idx]))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    model = WaveSSM(n_feat=n_feat or X.shape[-1], d_model=d_model,
                    d_state=d_state, n_layers=n_layers,
                    n_classes=n_domains).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    it = 0
    while it < steps:
        for xb, yb in dl:
            if it >= steps:
                break
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
    accs = {}
    for dd in sorted(te_by_dom):
        ds_te = torch.utils.data.TensorDataset(
            torch.from_numpy(X[te_by_dom[dd]]), torch.from_numpy(y_dom[te_by_dom[dd]]))
        accs[dd] = evaluate(model, torch.utils.data.DataLoader(ds_te, batch_size=64), device)
    return accs, float(np.mean(list(accs.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "wave"))
    ap.add_argument("--stride", type=int, default=None,
                    help="窗口步长 (默认=window 即不重叠, 防 train/test 泄漏)")
    ap.add_argument("--no-wfr", action="store_true",
                    help="只用 MAG 4 维特征, 不接 WFR 波谱")
    ap.add_argument("--wfr-bands", type=int, default=13,
                    help="WFR 65 频段聚合成的段数 (每段磁/电各 1 维 → 2*N 维)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs-per-domain", type=int, default=300)
    ap.add_argument("--joint-steps", type=int, default=1500,
                    help="联合训练诊断步数 (非持续学习上限)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-ratio", type=float, default=1.0,
                    help="回放样本数 / 当前 batch 数 (1.0 = 等量混入)")
    ap.add_argument("--domains", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "lm4_wave"))
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    times, dens, F_mat, ok = build_feature_matrix(
        args.data, bands=args.wfr_bands, use_wfr=not args.no_wfr)
    X, y_log = build_windows(times, dens, F_mat, ok, stride=args.stride)
    eff_stride = args.stride if args.stride else WINDOW
    d_feat = X.shape[-1] if len(X) else 0
    print(f"[data] 样本 {len(X)} 窗口 x {WINDOW} x {d_feat} 维 (stride={eff_stride}), "
          f"{time.time()-t0:.0f}s", flush=True)
    if len(X) == 0:
        print("ERROR: 无样本 (检查 MAG/WFR 数据是否下载)"); return
    y_dom, qs = assign_domains(y_log, args.domains)
    print(f"[domains] 边界 log10(density): {[round(float(q),2) for q in qs]}", flush=True)
    cnt = np.bincount(y_dom, minlength=args.domains)
    print(f"[domains] 每域样本: {cnt.tolist()}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    results = {}

    # 诊断: 联合训练上限 (判断任务本身是否可解; --joint-steps 0 跳过)
    if args.joint_steps > 0:
        print("\n=== Joint (联合训练上限诊断) ===", flush=True)
        joint_accs, joint_mean = run_joint(
            X, y_dom, device, d_model=args.d_model, d_state=args.d_state,
            n_layers=args.n_layers, steps=args.joint_steps, batch=args.batch,
            lr=args.lr, seed=args.seed, n_domains=args.domains, n_feat=d_feat)
        print("  → 联合训练平均 acc %.4f | 各域 %s" % (
            joint_mean, {k: round(v, 3) for k, v in joint_accs.items()}), flush=True)
        results["joint"] = {"final_mean_acc": joint_mean, "per_domain": joint_accs}

    for replay in (False, True):
        print(f"\n=== {'Replay' if replay else 'Naive sequential'} ===", flush=True)
        M, doms = run_experiment(X, y_dom, device, d_model=args.d_model,
                                 d_state=args.d_state, n_layers=args.n_layers,
                                 epochs_per_domain=args.epochs_per_domain,
                                 batch=args.batch, lr=args.lr, replay=replay,
                                 replay_ratio=args.replay_ratio,
                                 seed=args.seed, n_domains=args.domains,
                                 n_feat=d_feat)
        s = summarize(M, doms)
        key = "replay" if replay else "naive"
        results[key] = s
        print(f"  → 最终平均 acc {s['final_mean_acc']:.4f}, 平均遗忘 {s['mean_forget']:.4f}", flush=True)

    # 对比报告
    feat_desc = (f"MAG 4 维" if args.no_wfr
                 else f"MAG 4 维 + WFR 波谱 {2*args.wfr_bands} 维 ({args.wfr_bands} 段 x 磁/电)")
    lines = ["# LM4 电磁波持续学习 — 结果", "",
             f"- 数据: RBSP-A EMFISIS (DENSITY L4 + MAG L3{' + WFR L2' if not args.no_wfr else ''}), 窗口 {WINDOW}",
             f"- 特征: {feat_desc} (共 {d_feat} 维)",
             f"- 窗口步长: {eff_stride} ({'不重叠' if eff_stride == WINDOW else '有重叠!'})",
             f"- 样本: {X.shape[0]}, 域数: {args.domains}",
             f"- 模型: WaveSSM d_model={args.d_model} d_state={args.d_state} layers={args.n_layers}",
             f"- 每域训练步: {args.epochs_per_domain}",
             f"- replay_ratio: {args.replay_ratio}, seed: {args.seed}", "",
             "| 方法 | 最终平均 acc | 平均遗忘 |", "|:--|--:|--:|"]
    for k in ("naive", "replay"):
        if k in results:
            lines.append(f"| {k} | {results[k]['final_mean_acc']:.4f} | {results[k]['mean_forget']:.4f} |")
    lines += ["", "## 遗忘矩阵 (行=训练阶段, 列=域)", ""]
    for k in ("naive", "replay"):
        if k not in results:
            continue
        M = results[k]["acc_matrix"]
        lines.append(f"### {k}")
        lines.append("")
        hdr = "| 阶段 | " + " | ".join(f"D{d}" for d in results[k]["domains"]) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(results[k]["domains"]) + 1))
        for i, row in enumerate(M):
            cells = " | ".join("nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}" for v in row)
            lines.append(f"| {i+1} | {cells} |")
        lines.append("")
    out_md = os.path.join(args.out, "lm4_wave_report.md")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    json.dump(results, open(os.path.join(args.out, "lm4_wave_results.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n报告: {out_md}", flush=True)
    print(f"总耗时: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
