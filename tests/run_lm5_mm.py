#!/usr/bin/env python3
"""LM5 — 文本 + 电磁波 统一持续学习模型 (multimodal continual learner).

一个**共享 SSM trunk**, 两种模态 stem, 各自的任务头:

    文本 (en/zh/code) : token ids -> Embedding  -> trunk -> text_head (下一个 token, CE)
    电磁波 (D0..D5)   : 窗口特征     -> Linear     -> trunk -> wave_head (密度域分类)

域序列(持续学习流): en -> zh -> code -> wave(D0..D5)

要回答的问题:
  1. 学完电磁波后, 文本能力丢多少?(跨模态遗忘 —— 比单模态难得多)
  2. 反过来, 学文本对电磁波是正向迁移还是干扰?
  3. CL 机制(replay / OML 双循环 / PLN per-feature 步长 / Reptile)在跨模态下是否仍有效?

用法:
  python3 tests/run_lm5_mm.py --cl-method replay --text-steps 300 --wave-epochs 300
  python3 tests/run_lm5_mm.py --cl-method oml2  --head pln
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from hibs_lnn.ssm_v30_3 import SSM_Layer_V30_3
from hibs_lnn.pln_head import (PLNHead, SwiftTDAdapter, inner_adapt,
                               oml_step, oml_step_swifttd)
from run_lm3_bpe import BPE
from run_lm4_wave import (WINDOW, build_feature_matrix, build_windows,
                          assign_domains, window_agg)
from hibs_lnn.causal import S4WorldSCM

CHUNK = 32          # 文本块长度
TEXT_DOMAINS = ("en", "zh", "code")


# ============================================================
# 模型: 共享 trunk + 模态 stem + 各自任务头
# ============================================================
class MultiModalSSM(nn.Module):
    def __init__(self, vocab, n_feat, n_wave_classes, n_causal_classes=5,
                 d_model=192, d_state=12, n_layers=2,
                 head_type="pln", pln_d=64, pool="cat", agg_path=False):
        super().__init__()
        self.pool = pool
        self.agg_path = agg_path
        self.head_type = head_type
        # ── 模态 stem (各自独立) ──
        self.text_stem = nn.Embedding(vocab, d_model)
        self.wave_stem = nn.Linear(n_feat, d_model)
        self.causal_stem = nn.Linear(CAUSAL_FEAT, d_model)
        # ── 共享 trunk (meta 参数 θ) ──
        self.layers = nn.ModuleList([
            SSM_Layer_V30_3(d_model, d_state, layer_idx=i, ent_mode='none')
            for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # ── 任务头 ──
        self.text_head = nn.Linear(d_model, vocab)          # LM
        d_wave = d_model * (3 if pool == "cat" else 1)
        if agg_path:
            d_wave += 5 * n_feat
        self.d_wave = d_wave
        if head_type == "swifttd":
            self.wave_head = SwiftTDAdapter(d_wave, n_wave_classes)
            self.causal_head = SwiftTDAdapter(d_wave, n_causal_classes)
        else:
            self.wave_head = PLNHead(d_wave, pln_d, n_wave_classes)
            self.causal_head = PLNHead(d_wave, pln_d, n_causal_classes)

    # ---- 共享 trunk ----
    def _trunk(self, h):
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h)

    # ---- 文本: (B, L) ids -> (B, L, vocab) logits ----
    def forward_text(self, ids):
        # 取末位 logits -> (B, vocab): 与统计池化骨干统一成"由上下文预测末位 token",
        # 两边才可比 (整序列 next-token 与单点预测不是同一个任务)。
        return self.text_head(self._trunk(self.text_stem(ids)))[:, -1]

    # ---- 电磁波: (B, L, n_feat) -> 表示 (B, d_wave) ----
    def _pool(self, h):
        if self.pool == "last":
            return h[:, -1]
        if self.pool == "mean":
            return h.mean(1)
        if self.pool == "max":
            return h.max(1).values
        return torch.cat([h[:, -1], h.mean(1), h.max(1).values], dim=-1)

    def encode_wave(self, x, agg=None):
        s = self._pool(self._trunk(self.wave_stem(x)))
        if self.agg_path:
            s = torch.cat([s, window_agg(x) if agg is None else agg], dim=-1)
        return s

    def forward_wave(self, x):
        return self.wave_head(self.encode_wave(x))

    # ---- 因果: (B, L, CAUSAL_FEAT) -> (B, d_wave) ----
    def encode_causal(self, x):
        return self._pool(self._trunk(self.causal_stem(x)))

    def forward_causal(self, x):
        return self.causal_head(self.encode_causal(x))

    def trunk_params(self):
        ps = [p for n, p in self.named_parameters()
              if not n.startswith(("text_head.", "wave_head.", "causal_head."))]
        return ps

    def wave_head_params(self):
        return [p for n, p in self.named_parameters()
                if n.startswith("wave_head.")]


class MultiModalMLP(nn.Module):
    """文本 + 电磁波 + 因果 的统一骨干 —— 用**统计特征 MLP** 取代 SSM 主干。

    依据 (docs/lm4_backbone_final_verdict.md): 在电磁波数据上 StatMLP 在
      天花板 / 保留率 / seed 稳定性 / 速度 四个轴上全面优于复值 SSM。
    跨模态沿用同一结论: 各模态先投影到公共 d_model, 再走**共享 MLP 主干**。

    各模态的"池化表示":
      text    : token embedding 的 [mean, last] 拼接 -> Linear -> d_model
                (文本任务相应改为"由上下文池化预测末位 token")
      wave    : window_agg (5*n_feat)        -> Linear -> d_model
      causal  : window_agg (5*CAUSAL_FEAT)   -> Linear -> d_model

    归一化默认 fixed (冻结全局 mu/sd) —— BatchNorm 的 running stats
    会随域漂移, 是持续学习的隐性杀手。定点统计量在投影**之后**做。
    """

    def __init__(self, vocab, n_feat, n_wave_classes, n_causal_classes=5,
                 d_model=192, hidden=512, n_layers=2, head_type="linear",
                 pln_d=64, pool="cat", agg_path=False, norm="fixed"):
        super().__init__()
        self.norm_mode = norm
        self.pool = pool
        self.d_model = d_model
        # ── 各模态 -> 公共 d_model ──
        self.text_emb = nn.Embedding(vocab, d_model)
        self.text_proj = nn.Linear(2 * d_model, d_model)
        self.wave_proj = nn.Linear(5 * n_feat, d_model)
        self.causal_proj = nn.Linear(5 * CAUSAL_FEAT, d_model)
        # ── 共享主干 ──
        if norm == "batchnorm":
            nml = nn.BatchNorm1d(d_model)
        elif norm == "fixed":
            nml = nn.BatchNorm1d(d_model)
        else:
            nml = nn.Identity()
        self.norm_layer = nml
        mods = [nml, nn.Linear(d_model, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.GELU()]
        self.body = nn.Sequential(*mods)
        d_head = hidden
        self.d_head = d_head
        # ── 任务头 ──
        self.text_head = nn.Linear(d_head, vocab)
        if head_type == "swifttd":
            from hibs_lnn.pln_head import SwiftTDAdapter
            self.wave_head = SwiftTDAdapter(d_head, n_wave_classes)
            self.causal_head = SwiftTDAdapter(d_head, n_causal_classes)
        elif head_type == "pln":
            from hibs_lnn.pln_head import PLNHead
            self.wave_head = PLNHead(d_head, pln_d, n_wave_classes)
            self.causal_head = PLNHead(d_head, pln_d, n_causal_classes)
        else:
            self.wave_head = nn.Linear(d_head, n_wave_classes)
            self.causal_head = nn.Linear(d_head, n_causal_classes)

    # ---- 文本: 由上下文池化预测末位 token ----
    def forward_text(self, ids):
        e = self.text_emb(ids)                       # (B, L, d)
        pooled = torch.cat([e.mean(1), e[:, -1, :]], dim=-1)
        return self.text_head(self.body(self.text_proj(pooled)))

    # ---- 电磁波 / 因果: 窗口统计 ----
    def encode_wave(self, x):
        return self.body(self.wave_proj(window_agg(x)))

    def forward_wave(self, x):
        return self.wave_head(self.encode_wave(x))

    def encode_causal(self, x):
        return self.body(self.causal_proj(window_agg(x)))

    def forward_causal(self, x):
        return self.causal_head(self.encode_causal(x))

    # ---- fixed 归一化 ----
    def set_fixed_stats(self, feats):
        """feats: (N, d_model) 各类模态投影后的拼接, 用于算全局 mu/sd。"""
        if self.norm_mode != "fixed":
            return
        with torch.no_grad():
            self.norm_layer.running_mean.copy_(feats.mean(0))
            self.norm_layer.running_var.copy_(feats.std(0).clamp(min=1e-6) ** 2)
            self.norm_layer.num_batches_tracked.fill_(1)
        self.norm_layer.eval()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "norm_mode", None) == "fixed":
            self.norm_layer.eval()
        return self


# ============================================================
# 数据
# ============================================================
def load_text_domains(data_dir, limit=2_000_000):
    """en / zh / code 三个文本域。缺文件则跳过。"""
    dom = {}
    for name, fname in [("en", "en.txt"), ("zh", "zh.txt"), ("code", "code.txt")]:
        p = Path(data_dir) / fname
        if p.exists():
            dom[name] = p.read_text(errors="ignore")[:limit]
    return dom


def make_text_chunks(ids, rng, batch, chunk=CHUNK):
    """x = 上下文 (B, chunk-1); y = 末位 token (B,) 要预测的目标。

    统一成"由上下文预测末位 token" —— 统计池化骨干只能出单点预测,
    SSM 骨干也改用同一任务, 两边才可比。
    """
    n = len(ids)
    if n < chunk + 2:
        return None
    starts = [rng.randrange(0, n - chunk - 1) for _ in range(batch)]
    x = torch.tensor([ids[s:s + chunk - 1] for s in starts], dtype=torch.long)
    y = torch.tensor([ids[s + chunk - 1] for s in starts], dtype=torch.long)
    return x, y


# ============================================================
# 因果域数据 (S4WorldSCM)
# ============================================================
CAUSAL_FEAT = 10       # [6 位生成元 sig] + [cyc] + [3 维 obs]


def build_causal_data(n=20000, L=8, seed=0, n_bins=5):
    """从 S4WorldSCM 生成因果域序列分类数据。

    S4WorldSCM 的因果结构:  G -> cyc;  G -> obs;  cyc -> obs;  obs,cyc -> acc

    每样本:
      x (L, CAUSAL_FEAT)  每步 = [6 位生成元 sig, cyc, obs(3)]
      y = acc 分桶         (n_bins 类)          <- **观测**结果
      y_do = do() 干预后的 acc 分桶              <- **干预**结果

    关键: y 与 y_do 的差就是"相关 vs 因果"的判别面。
    只学到相关性的模型在 y_do 上会塌, 学到因果结构的模型能撑住。
    """
    scm = S4WorldSCM(seed=seed)
    rng = np.random.default_rng(seed)
    perms = list(scm.factor.keys())
    X = np.zeros((n, L, CAUSAL_FEAT), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    y_do = np.zeros(n, dtype=np.int64)
    for i in range(n):
        perm = perms[int(rng.integers(len(perms)))]
        sig = np.array(scm.sig_bits(perm), dtype=np.float32)
        cyc = float(scm.cyc(perm))
        # 随机"已学生成元"子集 -> 让 acc 有丰富分布
        k = int(rng.integers(0, 7))
        atoms = set(rng.choice(6, size=k, replace=False).tolist())
        for t in range(L):
            obs = np.asarray(scm.observe(perm, rng), dtype=np.float32)
            X[i, t] = np.concatenate([sig, [cyc], obs])
        # ★ 观测/干预必须用**同一个**噪声水平, 否则两边标签难度不同,
        #   "干预比观测准"会是纯假象 (do_intervene 内部写死 noise=0)。
        a = scm.acc(perm, atoms)
        # ★ 干预只点亮**部分**生成元。旧写法把该 perm 的全部生成元并进已学集合,
        #   known/total ≡ 1 -> acc 恒 ~0.95 -> y_do 有 78.9% 挤在同一类,
        #   多数类基线 0.789, 模型原始准确率 0.734 竟低于常数预测器 ->
        #   "干预 vs 观测"的对照完全失去区分度 (实测见 docs/lm5_*).
        #   干预强度**均匀覆盖**: acc = 0.35 + 0.6*(已学数/总生成元数), 若只允许
        #   "点亮"则 acc 只会往上走, y_do 仍偏斜 (实测多数类 0.663)。改成先均匀
        #   抽一个"目标已学数"再补齐, 让 y_do 覆盖整个 [0.35, 0.95] 值域,
        #   同时也包含"无效干预"(_need=0, 干预后与原观测一致)。
        _sig = scm.sig_bits(perm)
        _tot = max(1, int(sum(_sig)))
        _cur = sum(1 for i in range(6) if _sig[i] and i in atoms)
        _cand = [i for i in range(6) if _sig[i] and i not in atoms]
        _target = int(rng.integers(_cur, _tot + 1))
        _need = min(max(0, _target - _cur), len(_cand))
        _do_atoms = (list(rng.choice(_cand, size=_need, replace=False))
                     if _cand and _need else [])
        a_do = scm.do_intervene(perm, atoms, atoms=_do_atoms)
        # 给干预版补上同样的观测噪声
        a_do_n = float(np.clip(a_do + rng.normal(0, 0.08), 0.02, 0.98))
        y[i] = min(n_bins - 1, max(0, int(a * n_bins)))
        y_do[i] = min(n_bins - 1, max(0, int(a_do_n * n_bins)))
    return X, y, y_do


# ============================================================
# 持续学习主循环
# ============================================================
def balanced_acc(pred, y, n_classes):
    """宏平均召回 (balanced accuracy) —— 对类不平衡免疫。

    为什么必须要它: 因果域 y_do 的类分布是 {0:0.001, 1:0.029, 2:0.015,
    3:0.165, 4:0.789}, 多数类基线就高达 0.789。此时**原始准确率毫无意义**
    —— 一个常数预测器就能拿到 0.789, 看起来"干预任务学得很好"。
    必须用宏平均召回, 每类等权, 才能看出模型是否真的学到了结构。
    """
    pred = np.asarray(pred); y = np.asarray(y)
    rs = []
    for c in range(n_classes):
        m = (y == c)
        if m.sum() > 0:
            rs.append(float((pred[m] == c).mean()))
    return float(np.mean(rs)) if rs else float('nan')


def majority_baseline(y, n_classes):
    """常数预测器的得分 (取最高频类) —— 任何低于它的"准确率"都是假象。"""
    y = np.asarray(y)
    if len(y) == 0:
        return float('nan')
    cnt = np.bincount(y, minlength=n_classes)
    return float(cnt.max() / len(y))


def batched_pred(fwd, X, bs=512):
    """分批前向再 argmax —— 一次性喂整个测试集会 OOM
    (SSM 的 Ab 张量是 (B,L,d,s) 复数, B=1.2万 时 ~5GB)。"""
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(fwd(X[i:i + bs]).argmax(-1))
    return torch.cat(outs)


def run_mm(text_domains, wave_X, wave_y, wave_test, device,
           causal=None, n_causal_classes=5, backbone="mlp", norm="fixed",
           d_model=192, d_state=12, n_layers=2, vocab=1000,
           n_feat=30, n_wave_classes=6, head_type="pln",
           text_steps=300, wave_epochs=300, batch=32, lr=1e-3,
           cl_method="replay", replay_ratio=1.0, seed=42, pool="cat",
           agg_path=False, inner_k=2, inner_lr=0.1, reptile_lr=0.0,
           consolidate_every=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)

    if backbone == "mlp":
        model = MultiModalMLP(vocab, n_feat, n_wave_classes,
                              n_causal_classes=n_causal_classes, d_model=d_model,
                              hidden=d_model, n_layers=n_layers,
                              head_type=head_type, pool=pool,
                              agg_path=agg_path, norm=norm).to(device)
    else:
        model = MultiModalSSM(vocab, n_feat, n_wave_classes,
                              n_causal_classes=n_causal_classes, d_model=d_model,
                              d_state=d_state, n_layers=n_layers,
                              head_type=head_type, pool=pool,
                              agg_path=agg_path).to(device)
    # fixed 归一化: 从各模态的投影特征算一次全局 mu/sd 然后冻结。
    # (BatchNorm 的 running stats 随域漂移 = 持续学习的隐性杀手)
    if backbone == "mlp" and norm == "fixed":
        with torch.no_grad():
            rows = []
            for d in list(text_domains)[:3]:
                ids = torch.tensor(text_domains[d][:30000], dtype=torch.long)
                if len(ids) < CHUNK + 2:
                    continue
                hi = len(ids) - CHUNK - 1
                st = np.random.randint(0, hi, min(256, hi))
                xx = torch.stack([ids[t:t + CHUNK - 1] for t in st]).to(device)
                e = model.text_emb(xx)
                rows.append(model.text_proj(torch.cat([e.mean(1), e[:, -1, :]], -1)))
            if wave_X is not None and len(wave_X):
                rows.append(model.wave_proj(window_agg(torch.as_tensor(
                    wave_X[:256]).float().to(device))))
            if causal is not None and causal.get("X") is not None and len(causal["X"]):
                cx = torch.as_tensor(causal["X"][:256]).float().to(device)
                rows.append(model.causal_proj(window_agg(cx)))
            if rows:
                model.set_fixed_stats(torch.cat(rows, 0))
                print(f"[norm] fixed 统计量: {sum(len(r) for r in rows)} 个投影特征",
                      flush=True)

    use_pln = head_type in ("pln", "swifttd")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt_outer = opt
    if use_pln and cl_method == "oml2":
        opt_outer = torch.optim.Adam(model.parameters(), lr=lr)

    # 域序列: 文本域(有数据且非空) + 电磁波域
    domains = [d for d in TEXT_DOMAINS if d in text_domains and len(text_domains[d]) > 2000]
    domains = domains + ["wave"]
    if causal is not None:
        domains = domains + ["causal"]
    n_dom = len(domains)
    # 随机基线
    chance = {d: 1.0 / vocab for d in domains if d not in ("wave", "causal")}
    chance["wave"] = 1.0 / n_wave_classes
    if "causal" in domains:
        chance["causal"] = 1.0 / n_causal_classes
        chance["causal_do"] = 1.0 / n_causal_classes
        # 平衡准确率的随机线 = 1/类数 (宏平均召回下常数预测器只能拿到 ~1/K)
        chance["causal_bal"] = 1.0 / n_causal_classes
        chance["causal_do_bal"] = 1.0 / n_causal_classes

    # 回放缓冲: 每域固定量
    replay_buf = {}
    acc_hist = []

    def eval_all():
        """返回 {域: 指标}。文本=下一个 token 准确率, 电磁波=密度域准确率。"""
        model.eval()
        out = {}
        with torch.no_grad():
            for d in domains:
                if d == "wave":
                    if d in wave_test:
                        Xte, yte = wave_test[d]
                        pred = batched_pred(model.forward_wave, Xte)
                        out[d] = float((pred == yte).float().mean())
                    else:
                        out[d] = float('nan')
                elif d == "causal":
                    # 关键: 同时报**观测**与**干预**两个准确率。
                    # 只学到相关性的模型会在干预版上塌 —— 那个 gap 就是因果能力的度量。
                    if causal is None:
                        out["causal"] = float('nan')
                        out["causal_do"] = float('nan')
                    else:
                        Xo, yo = causal["test_obs"]
                        Xd, yd = causal["test_do"]
                        po = batched_pred(model.forward_causal, Xo).cpu().numpy()
                        pd_ = batched_pred(model.forward_causal, Xd).cpu().numpy()
                        yo_n, yd_n = yo.cpu().numpy(), yd.cpu().numpy()
                        out["causal"] = float((po == yo_n).mean())
                        out["causal_do"] = float((pd_ == yd_n).mean())
                        # ★ 平衡准确率: 因果域的标签严重偏斜, 原始准确率会骗人
                        out["causal_bal"] = balanced_acc(po, yo_n, n_causal_classes)
                        out["causal_do_bal"] = balanced_acc(pd_, yd_n, n_causal_classes)
                        # 首次评估时把常数基线记下来, 供报告判读
                        if "causal_base" not in out:
                            out["causal_base"] = majority_baseline(yo_n, n_causal_classes)
                            out["causal_do_base"] = majority_baseline(yd_n, n_causal_classes)
                else:
                    ids = torch.tensor(text_domains[d][:60000], dtype=torch.long)
                    if len(ids) < CHUNK + 2:
                        out[d] = float('nan'); continue
                    s = tot = 0
                    for _ in range(8):
                        st = rng.randrange(0, len(ids) - CHUNK - 1)
                        x = ids[st:st + CHUNK - 1].unsqueeze(0).to(device)
                        y = ids[st + CHUNK - 1].unsqueeze(0).to(device)
                        s += int((model.forward_text(x).argmax(-1) == y).sum())
                        tot += y.numel()
                    out[d] = s / max(1, tot)
        model.train()
        return out

    t0 = time.time()
    for step, dom in enumerate(domains):
        print(f"\n--- 域 {step+1}/{n_dom}: {dom} ---", flush=True)
        model.train()

        if dom in ("wave", "causal"):
            # ── 分类域 (电磁波密度 / 因果 acc): 同一套逻辑, 只换 stem+head+数据 ──
            if dom == "wave":
                CX, CY = wave_X, wave_y
                enc_fn, head_fn, fwd_fn = (model.encode_wave, model.wave_head,
                                           model.forward_wave)
            else:
                CX, CY = causal["X"], causal["y"]
                enc_fn, head_fn, fwd_fn = (model.encode_causal, model.causal_head,
                                           model.forward_causal)
            tr_idx = np.arange(len(CX))
            it = 0
            while it < wave_epochs:
                sel = np.random.RandomState(seed + it).choice(
                    tr_idx, size=min(batch, len(tr_idx)), replace=False)
                xb = torch.from_numpy(CX[sel]).to(device)
                yb = torch.from_numpy(CY[sel]).to(device)
                if cl_method == "oml2" and use_pln:
                    parts_x, parts_y = [xb], [yb]
                    if replay_buf.get(dom) is not None:
                        _h, _y = replay_buf[dom]
                        k = max(1, int(batch * replay_ratio))
                        sidx = torch.randint(0, len(_y), (k,), device=device)
                        parts_x.append(_h[sidx]); parts_y.append(_y[sidx])
                    XB, YB = torch.cat(parts_x), torch.cat(parts_y)
                    # ★ 同 run_lm4_wave: 必须随机切, 否则 support=新域 / query=旧域
                    _perm = torch.randperm(XB.shape[0], device=device)
                    XB, YB = XB[_perm], YB[_perm]
                    half = max(2, XB.shape[0] // 2)
                    with torch.no_grad():
                        h_sup = enc_fn(XB[:half])
                    h_qry = enc_fn(XB[half:])
                    fn = oml_step_swifttd if head_type == "swifttd" else oml_step
                    kw = dict(K=inner_k,
                              consolidate=(consolidate_every > 0 and it % consolidate_every == 0),
                              reptile_lr=reptile_lr)
                    if head_type != "swifttd":
                        kw["per_feature"] = True
                    lq, _ = fn(model, head_fn, h_sup, YB[:half], h_qry, YB[half:],
                               opt_outer, **kw)
                    # lm3 oml2 的第二部分: 常规训练步 (含 replay)
                    loss_n = F.cross_entropy(fwd_fn(XB), YB)
                    opt.zero_grad(); loss_n.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                else:
                    loss = F.cross_entropy(fwd_fn(xb), yb)
                    if cl_method == "replay" and replay_buf.get(dom) is not None:
                        rh, ry = replay_buf[dom]
                        k = max(1, int(batch * replay_ratio))
                        sidx = torch.randint(0, len(ry), (k,), device=device)
                        loss = loss + F.cross_entropy(fwd_fn(rh[sidx]), ry[sidx])
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                it += 1
            # 存回放 (按域)
            n_rb = min(2000, len(CX))
            sel = np.random.RandomState(seed).choice(len(CX), n_rb, replace=False)
            replay_buf[dom] = (torch.from_numpy(CX[sel]).to(device),
                               torch.from_numpy(CY[sel]).to(device))
        else:
            # ── 文本域: 下一个 token 预测 ──
            ids = text_domains[dom]
            for it in range(text_steps):
                batch_sz = batch
                if cl_method == "replay" and replay_buf:
                    # 其他域各抽一点混进来
                    pass
                c = make_text_chunks(ids, rng, batch_sz)
                if c is None:
                    break
                x, y = c[0].to(device), c[1].to(device)
                loss = F.cross_entropy(model.forward_text(x), y)
                if cl_method == "replay":
                    for rd, (rids,) in replay_buf.items():
                        if rd == "wave":
                            continue
                        rc = make_text_chunks(rids, rng, max(4, batch_sz // 8))
                        if rc is None:
                            continue
                        rx, ry = rc[0].to(device), rc[1].to(device)
                        rl = F.cross_entropy(model.forward_text(rx), ry)
                        loss = loss + replay_ratio * rl
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            # 存文本回放 (存 token 流片段)
            replay_buf[dom] = (ids[:200000],)

        row = eval_all()
        acc_hist.append(row)
        seen = [f"{d}:{row[d]:.3f}" for d in domains if d in row and not math.isnan(row[d])]
        print(f"  [{cl_method}] 已见域 → {' '.join(seen)}  ({time.time()-t0:.0f}s)", flush=True)

    # 注意: report_doms 可以比 acc_hist 的列多 (causal_do 由同一次评估派生),
    # 报告端必须按 row.get 取值, 不能用位置索引。
    report_doms = list(domains)
    if "causal" in domains:
        report_doms.append("causal_do")     # 干预版单列, 与观测版对比
        # ★ 平衡准确率列 —— y_do 的多数类基线高达 0.789, 原始准确率会骗人
        report_doms.append("causal_bal")
        report_doms.append("causal_do_bal")
    return acc_hist, report_doms, chance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-data", default=str(ROOT / "data" / "text"))
    ap.add_argument("--wave-data", default=str(ROOT / "data" / "wave"))
    ap.add_argument("--cl-method", default="replay", choices=["naive", "replay", "oml2"])
    ap.add_argument("--head", choices=["linear", "pln", "swifttd"], default="linear",
                    help="任务头; linear=普通线性(默认) pln/swifttd=元学习头")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--d-state", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--text-steps", type=int, default=300)
    ap.add_argument("--wave-epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--replay-ratio", type=float, default=1.0)
    ap.add_argument("--inner-k", type=int, default=2)
    ap.add_argument("--reptile-lr", type=float, default=0.0)
    ap.add_argument("--pool", default="cat", choices=["last", "mean", "max", "cat"])
    ap.add_argument("--agg-path", action="store_true")
    ap.add_argument("--domains", type=int, default=6)
    ap.add_argument("--no-causal", action="store_true",
                    help="不加因果域 (默认加: en/zh/code -> wave -> causal)")
    ap.add_argument("--causal-n", type=int, default=20000, help="因果域样本数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", choices=["mlp", "ssm"], default="mlp",
                    help="统一骨干: mlp=统计特征MLP(新, 默认) ssm=复值SSM(旧)")
    ap.add_argument("--norm", choices=["fixed", "batchnorm", "none"], default="fixed",
                    help="MLP 归一化; fixed=冻结全局mu/sd (默认, 抗 BN 漂移)")
    ap.add_argument("--out", default=str(ROOT / "results" / "lm5_mm"))
    args = ap.parse_args()
    device = torch.device(args.device)

    # ── 文本数据 ──
    td = load_text_domains(args.text_data)
    print(f"[text] 域: { {k: len(v) for k, v in td.items()} }", flush=True)
    if not td:
        print("!! 没有文本数据, 请在 --text-data 下放 en.txt / zh.txt / code.txt")
        sys.exit(1)
    # BPE —— 必须缓存: 纯 Python 训练在 120 万字符上要十几分钟, 而多配置跑会重复 N 次
    bpe = BPE()
    bpe_cache = str(Path(args.text_data) / f"bpe_lm5_{args.seed}.json")
    if os.path.exists(bpe_cache):
        ck = json.load(open(bpe_cache))
        bpe.base = ck["base"]
        bpe.merges = {tuple(int(x) for x in k.split("_")): v
                      for k, v in ck["merges"].items()}
        bpe.vocab = ck["vocab"]
        print(f"[bpe] 命中缓存 vocab={bpe.vocab}", flush=True)
    else:
        t_b = time.time()
        bpe.train([v[:400000] for v in td.values()], num_merges=800)
        json.dump({"base": bpe.base,
                   "merges": {f"{a}_{b}": v for (a, b), v in bpe.merges.items()},
                   "vocab": bpe.vocab}, open(bpe_cache, "w"))
        print(f"[bpe] 训练完成 vocab={bpe.vocab}, {time.time()-t_b:.0f}s (已缓存)", flush=True)
    vocab = bpe.vocab + 2
    enc_cache = str(Path(args.text_data) / f"enc_lm5_{args.seed}.json")
    if os.path.exists(enc_cache):
        enc = json.load(open(enc_cache))
        enc = {k: v for k, v in enc.items()}
        print(f"[bpe] 编码命中缓存", flush=True)
    else:
        t_e = time.time()
        enc = {k: bpe.encode(v) for k, v in td.items()}
        json.dump(enc, open(enc_cache, "w"))
        print(f"[bpe] 编码完成 {time.time()-t_e:.0f}s (已缓存)", flush=True)
    print(f"[bpe] 编码后长度: { {k: len(v) for k, v in enc.items()} }", flush=True)

    # ── 电磁波数据 ──
    times, dens, Fm, ok = build_feature_matrix(args.wave_data, bands=13,
                                               use_wfr=True, use_lshell=False)
    X, y_log = build_windows(times, dens, Fm, ok)
    y_dom, _ = assign_domains(y_log, args.domains)
    n_feat = X.shape[-1]
    rngs = np.random.RandomState(args.seed)
    idx = rngs.permutation(len(X))
    n_te = max(200, len(idx) // 10)
    te_idx, tr_idx = idx[:n_te], idx[n_te:]
    wave_X, wave_y = X[tr_idx].astype(np.float32), y_dom[tr_idx]
    wave_test = {"wave": (torch.from_numpy(X[te_idx].astype(np.float32)).to(device),
                          torch.from_numpy(y_dom[te_idx]).to(device))}
    print(f"[wave] 训练 {wave_X.shape}, 测试 {X[te_idx].shape}, 特征 {n_feat} 维", flush=True)
    # NaN 防护
    wave_X = np.nan_to_num(wave_X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 因果域 (S4WorldSCM) ──
    causal_data = None
    if not args.no_causal:
        cX, cy, cydo = build_causal_data(n=args.causal_n, seed=args.seed)
        n_cte = max(100, len(cX) // 10)
        c_idx = np.random.RandomState(args.seed).permutation(len(cX))
        c_te, c_tr = c_idx[:n_cte], c_idx[n_cte:]
        causal_data = {
            "X": cX[c_tr], "y": cy[c_tr],
            "test_obs": (torch.from_numpy(cX[c_te]).to(device),
                         torch.from_numpy(cy[c_te]).to(device)),
            "test_do": (torch.from_numpy(cX[c_te]).to(device),
                        torch.from_numpy(cydo[c_te]).to(device)),
        }
        print(f"[causal] 训练 {causal_data['X'].shape}, 测试 {cX[c_te].shape}, "
              f"类数 5, 观测/干预双标签", flush=True)

    hist, doms, chance = run_mm(
        enc, wave_X, wave_y, wave_test, device, causal=causal_data,
        d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
        vocab=vocab, n_feat=n_feat, n_wave_classes=args.domains,
        head_type=args.head, text_steps=args.text_steps,
        wave_epochs=args.wave_epochs, batch=args.batch, lr=args.lr,
        cl_method=args.cl_method, replay_ratio=args.replay_ratio,
        seed=args.seed, pool=args.pool, agg_path=args.agg_path,
        inner_k=args.inner_k, reptile_lr=args.reptile_lr,
        backbone=args.backbone, norm=args.norm)

    # ── 报告 ──
    Path(args.out).mkdir(parents=True, exist_ok=True)
    M = np.array([[row.get(d, float('nan')) for d in doms] for row in hist])
    print("\n=== 跨模态遗忘矩阵 (行=学到第几域, 列=各域指标) ===")
    print("       " + " ".join(f"{d:>9s}" for d in doms))
    for i in range(M.shape[0]):          # ← 按矩阵行数迭代, 不是按 doms
        tag = doms[i] if i < len(doms) else f"step{i+1}"
        print(f"  域{i+1}({tag:>4s}) " + " ".join(f"{v:9.3f}" for v in M[i]))
    # 遗忘
    forgets = {}
    for j, d in enumerate(doms):
        col = M[:, j]
        col = col[~np.isnan(col)]
        if len(col) >= 2:
            forgets[d] = float(col.max() - col[-1])
    print("\n=== 遗忘 (历史最佳 - 最终) ===")
    for d, f in forgets.items():
        print(f"  {d:>6s}: {f:+.4f}  (随机基线 {chance.get(d, float('nan')):.4f})")
    rep = {"config": vars(args), "domains": doms,
           "matrix": M.tolist(), "forgets": forgets, "chance": chance}
    Path(args.out, "lm5_mm_results.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"\n报告: {args.out}/lm5_mm_results.json")


if __name__ == "__main__":
    main()
