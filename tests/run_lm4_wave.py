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
from hibs_lnn.pln_head import PLNHead, inner_adapt, oml_step

WINDOW = 32
N_FEAT = 4
EPS = 1e-9
MAG_SENTINEL_LO = -1e4      # MAG L3 用 -100000 标记无效采样 (实测占 0.2%)
                            # 注意: 旧版用 np.abs() 会把哨兵伪装成 log10(1e5)=5.0 混进特征


# ============================================================
# 模型: 连续输入 SSM + 域分类头
# ============================================================
def window_agg(x):
    """(B,L,F) → (B,5F) 窗口聚合: mean/std/last/first/斜率。

    非序列基线 (MLP on 150 维聚合) 实测 joint 0.8045, 而旧 WaveSSM (只用末时刻
    隐状态) 只有 0.7213 —— 差距来自 SSM 必须从零学时间池化。"""
    return torch.cat([x.mean(1), x.std(1), x[:, -1, :], x[:, 0, :],
                      x[:, -1, :] - x[:, 0, :]], dim=-1)


class WaveSSM(nn.Module):
    """连续特征序列 → SSM → 物理状态(密度域)分类。

    pool: 时序池化
      'last' — 只用末时刻隐状态 (旧行为, 实测打不过聚合基线)
      'cat'  — [末时刻, 均值, 最大] 拼接 (默认)
    agg_path: 把原始窗口统计量直接拼进分类头 —— 给模型一条"聚合直通车",
      不必从零学池化 (与 MLP 基线对齐, 同时保留 SSM 的时序建模)
    """

    def __init__(self, n_feat=N_FEAT, d_model=128, d_state=8, n_layers=2,
                 n_classes=6, pool="cat", agg_path=False, head_type="linear",
                 pln_d=64, inner_lr=0.1, stat_input=False):
        super().__init__()
        self.pool = pool
        self.agg_path = agg_path
        self.head_type = head_type
        self.stat_input = stat_input
        # stat_input: 把窗口统计(5*n_feat)拼到**每个时刻的输入**上。
        #   动机: 探针显示 SSM 的序列编码低于简单统计基线 0.065-0.296
        #   (docs/lm4_ssm_bottleneck_verdict.md), 而 --agg-path (只拼到头上)
        #   实测更差 (0.6525 vs 0.7213) —— 说明统计量必须进**递归**才能被用上。
        n_in = n_feat + 5 * n_feat if stat_input else n_feat
        self.in_proj = nn.Linear(n_in, d_model)
        self.layers = nn.ModuleList([
            SSM_Layer_V30_3(d_model, d_state, layer_idx=i, ent_mode='none')
            for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        d_head = d_model * (3 if pool == "cat" else 1)
        if agg_path:
            d_head += 5 * n_feat
        self.d_head = d_head
        if head_type == "pln":
            # lm1 架构: 用 PLN (小 MLP + per-feature 步长) 取代线性头
            self.head = PLNHead(d_head, pln_d, n_classes, inner_lr=inner_lr)
        else:
            self.head = nn.Linear(d_head, n_classes)

    def encode(self, x, agg=None):
        """x: (B, L, n_feat) → 表示 s (B, d_head)。双循环里内/外循环都复用这个 s。"""
        if self.stat_input:
            st = window_agg(x)                       # (B, 5*n_feat)
            x = torch.cat([x, st.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        h = self.in_proj(x)
        for layer in self.layers:
            h, _ = layer(h)
        h = self.norm(h)
        if self.pool == "last":
            s = h[:, -1]
        elif self.pool == "mean":
            s = h.mean(1)
        elif self.pool == "max":
            s = h.max(1).values
        else:
            s = torch.cat([h[:, -1], h.mean(1), h.max(1).values], dim=-1)
        if self.agg_path:
            s = torch.cat([s, window_agg(x) if agg is None else agg], dim=-1)
        return s

    def forward(self, x, agg=None):
        """x: (B, L, n_feat) → logits (B, n_classes)"""
        return self.head(self.encode(x, agg))


class StatMLP(nn.Module):
    """窗口统计 + MLP backbone (替代 SSM)。

    依据: GPU 探针 (docs/lm4_ssm_bottleneck_verdict.md) 显示
      域数   MLP聚合基线   SSM joint   架构损失
       3     0.9159       0.8510      +0.065
       6     0.8065       0.7213      +0.085
      12     0.6638       0.3674      +0.296
    SSM 在**每个**粒度上都低于这个"不做序列建模"的基线, 且差距随粒度扩大。
    所以换掉它: 更快 (无递归)、更强、且同样的 CL 机制 (PLN/SwiftTD 头) 照常可用。

    接口与 WaveSSM 对齐 (encode / forward), 这样 run_experiment 不用改分支。
    """

    def __init__(self, n_feat=N_FEAT, n_classes=6, hidden=512, n_layers=3,
                 head_type="linear", pln_d=64, inner_lr=0.1, dropout=0.0,
                 stat_input=True, norm="batchnorm", **kw):
        super().__init__()
        self.n_feat = n_feat
        self.head_type = head_type
        self.norm_mode = norm
        # 聚合特征: mean/std/last/first/斜率 = 5*n_feat
        d_in = 5 * n_feat
        # ★ 必须标准化。探针 (probe_ceiling.py) 对聚合特征做了 (x-mu)/sd, 实测
        #   6 域 0.8065; 不做标准化的 StatMLP 只有 0.6558 —— 差 0.15 全在这一点。
        # ★ BN 有已知的持续学习陷阱: running stats 随域漂移 -> 旧域被错误归一化
        #   -> 灾难性遗忘。fixed 模式用冻结的全局 mu/sd 来验证这一点。
        if norm == "batchnorm":
            nml = nn.BatchNorm1d(d_in)
        elif norm == "fixed":
            nml = nn.BatchNorm1d(d_in)
        else:
            nml = nn.Identity()
        self.norm_layer = nml
        mods = [nml, nn.Linear(d_in, hidden), nn.GELU()]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.GELU()]
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*mods)
        self.d_head = hidden
        self.d_wave = hidden
        if head_type == "swifttd":
            from hibs_lnn.pln_head import SwiftTDAdapter
            self.head = SwiftTDAdapter(hidden, n_classes)
        elif head_type == "pln":
            from hibs_lnn.pln_head import PLNHead
            self.head = PLNHead(hidden, pln_d, n_classes, inner_lr=inner_lr)
        else:
            self.head = nn.Linear(hidden, n_classes)

    def set_fixed_stats(self, agg):
        """fixed 模式: 用整个训练集的 mu/sd 填 BN running stats 并冻结。"""
        if self.norm_mode != "fixed":
            return
        with torch.no_grad():
            mu = agg.mean(0)
            sd = agg.std(0).clamp(min=1e-6)
            self.norm_layer.running_mean.copy_(mu)
            self.norm_layer.running_var.copy_(sd ** 2)
            self.norm_layer.num_batches_tracked.fill_(1)
        self.norm_layer.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "norm_mode", None) == "fixed":
            self.norm_layer.eval()          # 永远 eval, 统计量不漂移
        return self

    def encode(self, x, agg=None):
        """x: (B, L, F) -> (B, hidden)。agg 参数仅为接口兼容。"""
        return self.body(window_agg(x))

    def forward(self, x, agg=None):
        return self.head(self.encode(x, agg))


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


def load_lshell_data(wave_dir, tol=3.0):
    """MAG 4SEC 的 coordinates(GEI, km) → [log10(偶极 L), sin(磁纬)], 2 维。

    注意: 用 GEI 的 z 轴当磁轴是近似(真实磁轴倾角约 11°)。
    valid = (magFill==0) & (magInvalid==0), 官方有效性标志。
    """
    files = sorted(Path(wave_dir).glob("magpos_*.npz"))
    T, F, V = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        c = z["coords"].astype(np.float64)
        fl = z["flags"]
        r = np.linalg.norm(c, axis=1)
        r_re = r / 6371.0
        sinlat = np.clip(c[:, 2] / np.maximum(r, 1e-9), -0.999, 0.999)
        L = r_re / np.maximum(1.0 - sinlat ** 2, 1e-3)
        F.append(np.stack([np.log10(np.maximum(L, 1e-3)), sinlat],
                          axis=1).astype(np.float32))
        V.append((fl[:, 0] == 0) & (fl[:, 1] == 0))
        T.extend(z["time"].tolist())
    if not F:
        return [], np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    tot = sum(len(x) for x in F)
    print(f"[lshell] {len(files)} 月, {tot} 条, 2 维特征")
    return T, np.concatenate(F, axis=0).astype(np.float32), np.concatenate(V)


def _feat_cache_path(wave_dir, bands, use_wfr, use_lshell):
    return (Path(wave_dir) / "_cache"
            / f"feat_b{bands}_wfr{int(use_wfr)}_ls{int(use_lshell)}.npz")


def build_feature_matrix(wave_dir, bands=13, use_wfr=True, use_lshell=False,
                         tol=3.0, use_cache=True):
    """统一特征管线: 以密度时间轴为准, 对齐 MAG + WFR + L-shell。

    返回 (times, dens, feats (N,D), ok (N,))
      D = 4 (log10 Mag, log10 rms, lambda, delta)
          [+ 2*bands (WFR 磁/电总功率谱)] [+ 2 (log10 偶极 L, sin 磁纬)]
    结果缓存到 data/wave/_cache/, 重跑省去 ~160s 重建。
    """
    cache = _feat_cache_path(wave_dir, bands, use_wfr, use_lshell)
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] 特征矩阵命中 {cache.name}")
        return z["times"].tolist(), z["dens"], z["F"], z["ok"]

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
            n0 = X.shape[1]
            X = np.concatenate([X, w_f[w_idx]], axis=1).astype(np.float32)
            X[~w_ok, n0:] = 0.0          # 缺失填 0 (这些窗口会被 ok 掩码丢弃)
            ok &= w_ok

    if use_lshell:
        lt, lf, lv = load_lshell_data(wave_dir)
        if len(lt):
            # 有效性标志必须跟着特征一起排序 → 拼成额外一列再拆开
            lf2 = np.concatenate([lf, lv.astype(np.float32)[:, None]], axis=1)
            l_f, l_idx, l_ok = _align(t_all, lt, lf2, tol)
            got = l_f[l_idx]
            l_feats, l_valid = got[:, :-1], got[:, -1] > 0.5
            l_ok = l_ok & l_valid
            n0 = X.shape[1]
            X = np.concatenate([X, l_feats], axis=1).astype(np.float32)
            X[~l_ok, n0:] = 0.0
            ok &= l_ok

    ok &= np.isfinite(X).all(axis=1)
    X[~ok] = 0.0
    print(f"[data] 特征 {X.shape[1]} 维, 有效点 {int(ok.sum())}/{n}")
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, times=np.array(t_all, dtype=object),
                            dens=dens, F=X, ok=ok)
        print(f"[cache] 已写入 {cache.name} ({cache.stat().st_size/1024/1024:.0f} MB)")
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


def assign_domains(y_log10, n_domains=6, qs=None):
    """按 log10(density) 分位数划分域 (等样本量, 保证每域可训练).

    qs 不为 None 时**沿用给定边界** —— 跨年/外部分布评估必须这样,
    否则测试年用自己算的分位数, 域定义与训练年不同, 标签根本对不上
    (曾因此得到 naive 跨年 1.000 的荒谬结果)。
    """
    if qs is None:
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
                   replay_ratio=0.3, seed=42, n_domains=6, n_feat=None,
                   pool="cat", agg_path=False, shuffle_domains=False,
                   cl_method="naive", pln_d=64, inner_k=2, inner_lr=0.1,
                   outer_lr=None, meta_every=1, reptile_lr=0.0,
                   consolidate_every=10, stat_input=False, backbone="ssm",
                   head_type="linear", norm="batchnorm",
                   trace_every=0, external_test=None):
    """cl_method:
      naive/replay — 单循环 (线性头), 原行为
      oml          — 快慢双循环 (lm3 `oml`): 内循环每步更新头, 外循环低频更新 RLN
      oml2         — 真 OML 双循环 (lm3 `oml2` + lm1 PLN): support 上适应克隆头
                     (per-feature 步长), query loss 反传全模型, 再合并
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # head_type 与 cl_method 解耦: 否则 "OML 失败" 分不清是 OML 的问题
    # 还是它默认带的 PLN 头(2 层 tanh MLP)的问题 —— 这是同时变两个变量的混淆。
    use_pln = head_type in ("pln", "swifttd")

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
    if shuffle_domains:
        # 随机化训练顺序: 检验"按密度递增"的课程是否重要
        random.Random(seed * 7919 + 13).shuffle(domains)

    def loader_for(idxs, shuffle=False):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X[idxs]), torch.from_numpy(y_dom[idxs]))
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)

    test_loaders = {d: loader_for(test_by_dom[d]) for d in domains}
    n_classes = n_domains
    ht = head_type
    if backbone == "mlp":
        model = StatMLP(n_feat=n_feat or X.shape[-1], n_classes=n_classes,
                        hidden=d_model, n_layers=n_layers, head_type=ht,
                        pln_d=pln_d, inner_lr=inner_lr, norm=norm).to(device)
        if norm == "fixed":
            _ti = np.concatenate([train_by_dom[d] for d in domains])
            with torch.no_grad():
                _all = torch.from_numpy(X[_ti]).to(device)
                model.set_fixed_stats(window_agg(_all))
            del _all
    else:
        model = WaveSSM(n_feat=n_feat or X.shape[-1], d_model=d_model,
                        d_state=d_state, n_layers=n_layers,
                        n_classes=n_classes, pool=pool, agg_path=agg_path,
                        head_type=ht, pln_d=pln_d, inner_lr=inner_lr,
                        stat_input=stat_input).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 双循环: 外循环 (慢, 表示+meta 参数) 与 内循环 (快, 仅头)
    opt_outer = opt
    if use_pln:
        enc_params = [p for n_, p in model.named_parameters()
                      if not n_.startswith("head.")]
        head_params = [p for n_, p in model.named_parameters()
                       if n_.startswith("head.")]
        if cl_method == "oml2":
            # 真 OML: 外循环更新全部 meta 参数 (表示 + 头初始化 + per-feature 步长)
            opt_outer = torch.optim.Adam(model.parameters(), lr=(outer_lr or lr))
            # lm3 的 oml2 是**两部分**: OML 元步 + 常规训练步(含 replay)。
            # 只做元步会让头仅靠 K 步内循环学, 合并时被覆盖 -> 什么都不剩。
            opt_norm = torch.optim.Adam(model.parameters(), lr=lr)
            opt_inner = None
        else:
            # 快慢双循环: 外循环只动表示, 内循环只动头
            opt_outer = torch.optim.Adam(enc_params, lr=(outer_lr or lr))
            opt_inner = torch.optim.Adam(head_params, lr=inner_lr)
    else:
        opt_inner = None

    acc_matrix = []          # acc_matrix[step][domain]
    cross = []               # 每个域学完后, 在**外部分布**(如 2016 年)上的准确率
    replay_by_dom = {}       # dd -> (X_dev, y_dev): 每域回放缓冲(预置 device)

    curves = {}        # dd -> [(step, acc)]  学习效率曲线
    for step, dd in enumerate(domains):
        dl = loader_for(train_by_dom[dd], shuffle=True)
        model.train()
        it = 0
        cur = []
        while it < epochs_per_domain:
            for xb, yb in dl:
                if it >= epochs_per_domain:
                    break
                xb, yb = xb.to(device), yb.to(device)

                if cl_method == "oml2":
                    # ===== 真 OML 双循环 (lm3 oml2 + lm1 PLN) =====
                    # 内循环: 克隆头在 support 上适应 K 步 (per-feature 步长)
                    # 外循环: fast head 在 query 上的 loss 反传全部 meta 参数
                    # support/query 都混入回放样本 (lm4 聚合所有有利于 CL 的机制)
                    parts_x, parts_y = [xb], [yb]
                    if replay and replay_by_dom:
                        k = max(1, int(batch * replay_ratio))
                        seen_doms = list(replay_by_dom)
                        per = max(1, k // len(seen_doms))
                        for sd in seen_doms:
                            bx, by = replay_by_dom[sd]
                            m = min(per, len(bx))
                            sel = torch.randint(0, len(bx), (m,), device=device)
                            parts_x.append(bx[sel]); parts_y.append(by[sel])
                    XB, YB = torch.cat(parts_x), torch.cat(parts_y)
                    # ★ 必须**随机**切 support/query。批次是 [当前域..., 回放(旧域)...]
                    #   拼的, 按位置切会让 support=纯新域 / query=纯旧域 ->
                    #   元目标退化成"适应新域、再在旧域上预测" -> 教模型摧毁旧域可分性。
                    _perm = torch.randperm(XB.shape[0], device=device)
                    XB, YB = XB[_perm], YB[_perm]
                    half = max(2, XB.shape[0] // 2)
                    xq, yq = XB[half:], YB[half:]
                    if xq.shape[0] < 2:
                        it += 1; continue
                    with torch.no_grad():
                        h_sup = model.encode(XB[:half])
                    h_qry = model.encode(xq)              # 保留计算图
                    # 关键: 合并必须**周期性**做 (lm3 是每 10 步), 每步合并会让头
                    # 被"只在当前批次上适应过 K 步"的版本覆盖 → 永远积累不了知识
                    lq, _ = oml_step(model, model.head, model.encode,
                                     h_sup, YB[:half], h_qry, yq, opt_outer,
                                     K=inner_k, per_feature=True,
                                     consolidate=(consolidate_every > 0
                                                  and it % consolidate_every == 0),
                                     reptile_lr=reptile_lr)
                    # ---- lm3 oml2 的第二部分: 头/全模型照常训练, 含 replay ----
                    # (这才是让 OML 站在 replay 之上而不是取代它的关键)
                    loss_n = F.cross_entropy(model(XB), YB)
                    opt_norm.zero_grad(); loss_n.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
                    opt_norm.step()
                    loss = torch.tensor(lq, device=device)
                    it += 1
                    continue

                if cl_method == "oml":
                    # ===== 快慢双循环 (lm3 oml) =====
                    # 内循环(快): 表示视为固定特征, 只更新头 (+回放)
                    parts_x, parts_y = [xb], [yb]
                    if replay and replay_by_dom:
                        k = max(1, int(batch * replay_ratio))
                        seen_doms = list(replay_by_dom)
                        per = max(1, k // len(seen_doms))
                        for sd in seen_doms:
                            bx, by = replay_by_dom[sd]
                            m = min(per, len(bx))
                            sel = torch.randint(0, len(bx), (m,), device=device)
                            parts_x.append(bx[sel]); parts_y.append(by[sel])
                    XB, YB = torch.cat(parts_x), torch.cat(parts_y)
                    with torch.no_grad():
                        hb = model.encode(XB)
                    loss = F.cross_entropy(model.head(hb), YB)
                    opt_inner.zero_grad(); loss.backward(); opt_inner.step()
                    # 外循环(慢): 低频更新表示
                    if meta_every > 0 and it % meta_every == 0:
                        opt_outer.zero_grad()
                        l2 = F.cross_entropy(model(XB), YB)
                        l2.backward(); opt_outer.step()
                    it += 1
                    continue

                # ===== 单循环 (naive / replay): 原行为, 不变 =====
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
                # ── 学习效率追踪: 每 trace_every 步评一次**当前域** ──
                if (trace_every and it % trace_every == 0
                        and dd in test_loaders):
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        a = evaluate(model, test_loaders[dd], device)
                    model.train(was_training)
                    cur.append((it, a))
        # 记录: 已见域整体准确率
        row = [evaluate(model, test_loaders[d], device) if d in test_loaders else float('nan')
               for d in domains]
        acc_matrix.append(row)
        # 跨年/外部分布: 每步评**所有**外部域, 形成外部矩阵。
        # 只报"刚训完 dd 时在外部 dd 上的准确率"是没有意义的 —— naive 在那个瞬间
        # 恰好专精于 dd, 会得到 1.0。有意义的量是**全流程训完后**的末行。
        if external_test:
            model.eval()
            with torch.no_grad():
                cross.append([evaluate(model, external_test[d], device)
                              if d in external_test else float('nan')
                              for d in range(n_domains)])
            model.train()
        else:
            cross.append(None)
        if cur:
            curves[dd] = cur
        seen = [f"D{d}:{row[i]:.3f}" for i, d in enumerate(domains) if i <= step]
        print(f"  [{'replay' if replay else 'naive '}] step{step+1} (域{dd}) → {' '.join(seen)}", flush=True)
        # 存回放样本 (每域固定 2000, 预置 device 避免逐步搬运)
        if replay:
            idx = train_by_dom[dd]
            sel = np.random.RandomState(seed + step).choice(
                idx, size=min(2000, len(idx)), replace=False)
            replay_by_dom[dd] = (torch.from_numpy(X[sel]).to(device),
                                 torch.from_numpy(y_dom[sel]).to(device))
    return acc_matrix, domains, curves, cross


def efficiency_report(curves, domains, thresholds=(0.5, 0.9)):
    """学习效率: 每个域达到其最终准确率 50%/90% 需要多少步。

    这是"更新产生效率"的直接度量 —— 与"压缩/天花板"无关。
    """
    if not curves:
        return {}
    out = {}
    for dd, cur in curves.items():
        if not cur:
            continue
        final = max(a for _, a in cur)
        d = domains[dd] if dd < len(domains) else str(dd)
        row = {}
        for th in thresholds:
            tgt = final * th
            hit = next((st for st, a in cur if a >= tgt), None)
            row[f"{int(th*100)}"] = hit
        out[f"D{d}(终{final:.3f})"] = row
    return out


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
              steps=1500, batch=32, lr=1e-3, seed=42, n_domains=6, n_feat=None,
              pool="cat", agg_path=False, stat_input=False, backbone="ssm"):
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
    if backbone == "mlp":
        model = StatMLP(n_feat=n_feat or X.shape[-1], n_classes=n_domains,
                        hidden=d_model).to(device)
    else:
        model = WaveSSM(n_feat=n_feat or X.shape[-1], d_model=d_model,
                        d_state=d_state, n_layers=n_layers,
                        n_classes=n_domains, pool=pool, agg_path=agg_path,
                        stat_input=stat_input).to(device)
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
    ap.add_argument("--lshell", action="store_true",
                    help="追加 L-shell 特征 (log10 偶极 L, sin 磁纬), 需 magpos_*.npz")
    ap.add_argument("--no-cache", action="store_true",
                    help="不使用/不写入特征矩阵缓存")
    ap.add_argument("--wfr-bands", type=int, default=13,
                    help="WFR 65 频段聚合成的段数 (每段磁/电各 1 维 → 2*N 维)")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--pool", default="last", choices=["last", "mean", "max", "cat"],
                    help="SSM 时序池化 (last 实测最好: 0.7213 vs cat 0.6681)")
    ap.add_argument("--test-data", default=None,
                    help="跨年/外部分布测试集目录 (如 data/wave2016 做 2015->2016 泛化)")
    ap.add_argument("--trace-every", type=int, default=0,
                    help="每 N 步评一次当前域, 产出学习效率曲线 (0=关闭)")
    ap.add_argument("--norm", default="batchnorm", choices=["batchnorm", "fixed", "none"],
                    help="StatMLP 归一化: batchnorm (随域漂移) / fixed (冻结全局 mu/sd) / none")
    ap.add_argument("--head", default="linear", choices=["linear", "pln", "swifttd"],
                    help="分类头: linear (原) / pln (Meta-SGD 逐参数步长) / swifttd (lm1 局部规则)")
    ap.add_argument("--backbone", default="mlp", choices=["mlp", "ssm"],
                    help="骨干: mlp = 窗口统计+MLP (探针实测更强更快); ssm = 原复值 SSM")
    ap.add_argument("--stat-input", action="store_true",
                    help="把窗口统计拼到每个时刻的输入上 (修 SSM 丢全局信息的候选方案)")
    ap.add_argument("--agg-path", action="store_true",
                    help="窗口聚合统计量直接拼进分类头 (聚合直通车)")
    ap.add_argument("--epochs-per-domain", type=int, default=300)
    ap.add_argument("--joint-steps", type=int, default=1500,
                    help="联合训练诊断步数 (非持续学习上限)")
    ap.add_argument("--joint-only", action="store_true",
                    help="只跑 joint 天花板, 跳过 naive/replay (容量扫描用)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-ratio", type=float, default=1.0,
                    help="回放样本数 / 当前 batch 数 (1.0 = 等量混入)")
    ap.add_argument("--domains", type=int, default=6)
    ap.add_argument("--shuffle-domains", action="store_true",
                    help="随机化域训练顺序 (检验按密度递增的课程是否重要)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cl-method", default="naive", choices=["naive", "replay", "oml", "oml2"],
                    help="持续学习方法: naive/replay = 单循环 (线性头); "
                         "oml = 快慢双循环 (lm3 oml); oml2 = 真 OML 双循环 (lm3 oml2 + lm1 PLN)")
    ap.add_argument("--pln-d", type=int, default=64, help="PLN 头隐层维度 (lm1 V31_PLN)")
    ap.add_argument("--inner-k", type=int, default=2, help="OML 内循环适应步数 K")
    ap.add_argument("--inner-lr", type=float, default=0.1, help="内循环基础步长 (per-feature β 初值)")
    ap.add_argument("--outer-lr", type=float, default=None, help="外循环学习率 (默认同 --lr)")
    ap.add_argument("--meta-every", type=int, default=1,
                    help="快慢双循环里外循环每 N 步更新一次 (lm3 用 10)")
    ap.add_argument("--reptile-lr", type=float, default=0.0,
                    help="OML 里 Reptile init 平均步长 (lm1 meta_step 的 init_lr; 0=关闭)")
    ap.add_argument("--consolidate-every", type=int, default=10,
                    help="OML 把适应后的头合并回本体的间隔步数 (lm3 用 10; 0=不合并)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "lm4_wave"))
    args = ap.parse_args()

    device = torch.device(args.device)
    t0 = time.time()
    times, dens, F_mat, ok = build_feature_matrix(
        args.data, bands=args.wfr_bands, use_wfr=not args.no_wfr,
        use_lshell=args.lshell, use_cache=not args.no_cache)
    X, y_log = build_windows(times, dens, F_mat, ok, stride=args.stride)
    eff_stride = args.stride if args.stride else WINDOW
    d_feat = X.shape[-1] if len(X) else 0
    print(f"[data] 样本 {len(X)} 窗口 x {WINDOW} x {d_feat} 维 (stride={eff_stride}), "
          f"{time.time()-t0:.0f}s", flush=True)
    if len(X) == 0:
        print("ERROR: 无样本 (检查 MAG/WFR 数据是否下载)"); return
    y_dom, domain_qs = assign_domains(y_log, args.domains)
    print(f"[domains] 边界 log10(density): {[round(float(q),2) for q in domain_qs]}", flush=True)
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
            lr=args.lr, seed=args.seed, n_domains=args.domains, n_feat=d_feat,
            pool=args.pool, agg_path=args.agg_path,
            stat_input=args.stat_input, backbone=args.backbone)
        print("  → 联合训练平均 acc %.4f | 各域 %s" % (
            joint_mean, {k: round(v, 3) for k, v in joint_accs.items()}), flush=True)
        results["joint"] = {"final_mean_acc": joint_mean, "per_domain": joint_accs}

    # ── 跨年/外部分布测试集 (如 2015 训练 -> 2016 测试) ──
    ext_test = None
    if args.test_data:
        t2, d2, F2, ok2 = build_feature_matrix(
            args.test_data, bands=args.wfr_bands, use_wfr=not args.no_wfr,
            use_lshell=args.lshell, use_cache=not args.no_cache)
        X2, y_log2 = build_windows(t2, d2, F2, ok2, stride=args.stride)
        y2, _ = assign_domains(y_log2, args.domains, qs=domain_qs)
        if len(X2):
            ext_test = {}
            for dd in range(args.domains):
                idx = np.where(y2 == dd)[0]
                if len(idx) < 20:
                    continue
                _X = torch.from_numpy(X2[idx].astype(np.float32)).to(device)
                _y = torch.from_numpy(y2[idx]).to(device)
                ext_test[dd] = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(_X, _y), batch_size=512)
            print(f"[cross] 外部测试集 {args.test_data}: {len(X2)} 窗口, "
                  f"{len(ext_test)} 个域有数据", flush=True)
        else:
            print(f"!! 外部测试集 {args.test_data} 无有效窗口", flush=True)

    if args.joint_only:
        runs = []
    elif args.cl_method in ("oml", "oml2"):
        # 双循环方法: 单次运行, 默认带回放 (lm4 聚合所有有利于 CL 的机制)
        runs = [(args.cl_method, True)]
    else:
        runs = [("naive", False), ("replay", True)]

    for key, replay in runs:
        print(f"\n=== {key} ===", flush=True)
        M, doms, curves, cross = run_experiment(X, y_dom, device, d_model=args.d_model,
                                 d_state=args.d_state, n_layers=args.n_layers,
                                 epochs_per_domain=args.epochs_per_domain,
                                 batch=args.batch, lr=args.lr, replay=replay,
                                 replay_ratio=args.replay_ratio,
                                 seed=args.seed, n_domains=args.domains,
                                 n_feat=d_feat, pool=args.pool,
                                 agg_path=args.agg_path,
                                 shuffle_domains=args.shuffle_domains,
                                 cl_method=(key if key in ("oml", "oml2") else "naive"),
                                 pln_d=args.pln_d, inner_k=args.inner_k,
                                 inner_lr=args.inner_lr, outer_lr=args.outer_lr,
                                 meta_every=args.meta_every,
                                 reptile_lr=args.reptile_lr,
                                 consolidate_every=args.consolidate_every,
                                 stat_input=args.stat_input,
                                 backbone=args.backbone, head_type=args.head,
                                 norm=args.norm,
                                 trace_every=args.trace_every,
                                 external_test=ext_test)
        s = summarize(M, doms)
        s["curves"] = {str(k): v for k, v in (curves or {}).items()}
        s["cross_external"] = cross                 # 外部矩阵 (每步一行)
        s["cross_external_final"] = (
            [None if (c != c) else float(c) for c in cross[-1]]
            if cross and cross[-1] is not None else None)
        s["efficiency"] = efficiency_report(curves, doms)
        results[key] = s
        print(f"  → 最终平均 acc {s['final_mean_acc']:.4f}, 平均遗忘 {s['mean_forget']:.4f}", flush=True)
        if ext_test and cross and cross[-1] is not None:
            last = cross[-1]
            ok = [c for c in last if c == c]
            if ok:
                print(f"  ── 跨年泛化 (全流程训完后) 平均 {sum(ok)/len(ok):.4f}  各域 "
                      + " ".join(f"D{i}:{c:.3f}" for i, c in enumerate(last) if c == c),
                      flush=True)
        if s["efficiency"]:
            print("  ── 学习效率 (达标步数) ──", flush=True)
            for k, v in s["efficiency"].items():
                print(f"     {k}: " + "  ".join(f"{m}%→{st}步" for m, st in v.items()), flush=True)

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
    # config 指纹: 汇总时按它过滤, 防止诊断 run(--epochs-per-domain 1 之类) 混入统计
    results["_config"] = {
        "use_wfr": not args.no_wfr, "wfr_bands": args.wfr_bands,
        "use_lshell": args.lshell, "stride": eff_stride,
        "d_model": args.d_model, "d_state": args.d_state,
        "n_layers": args.n_layers, "epochs_per_domain": args.epochs_per_domain,
        "pool": args.pool, "agg_path": args.agg_path, "stat_input": args.stat_input,
        "backbone": args.backbone, "head": args.head, "norm": args.norm,
        "domain_qs": [float(x) for x in domain_qs],
        "shuffle_domains": args.shuffle_domains,
        "cl_method": args.cl_method, "pln_d": args.pln_d,
        "inner_k": args.inner_k, "inner_lr": args.inner_lr,
        "outer_lr": args.outer_lr, "meta_every": args.meta_every,
        "reptile_lr": args.reptile_lr, "consolidate_every": args.consolidate_every,
        "joint_steps": args.joint_steps, "batch": args.batch, "lr": args.lr,
        "replay_ratio": args.replay_ratio, "domains": args.domains,
        "seed": args.seed, "n_windows": int(X.shape[0]), "n_feat": d_feat,
        # 规范化 argv 指纹: 队列据此判断"这个 tag 是否已用完全相同参数跑过"
        "argv_sig": " ".join(sorted(sys.argv[1:])),
    }
    out_md = os.path.join(args.out, "lm4_wave_report.md")
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    json.dump(results, open(os.path.join(args.out, "lm4_wave_results.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n报告: {out_md}", flush=True)
    print(f"总耗时: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
