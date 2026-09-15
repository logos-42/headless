#!/usr/bin/env python3
"""measure_temporal_structure.py — 先问:**这个任务有没有可被 option 利用的时间结构?**

## 用户的质疑(我认同, 所以先证伪自己)

「option 存在, 但可能是你的标准和 benchmark 没设置对。」

在我拿负面结论说服任何人放弃 option 之前, 必须先排除**最致命的一种可能**:

    如果任务里**动作之间近似可交换**, 那么
        "先训 a 再训 b" ≈ "只训 a" + "只训 b 带来的增益"
    即**不存在多步规划空间** —— option 无论怎么实现都不可能有用。
    那种情况下我的 E9/E10 对照测的是 **benchmark 的性质, 不是 option 的价值**。

## 量化: 一阶交互 + 不对称性

对一对动作 (a, b), 在与主实验**同构**的模型上:

    I(a,b) = acc_b(先训 a 再训 b) − acc_b(只训 b)
    A(a,b) = I(a,b) − I(b,a)        <- **这才是"顺序有意义"的证据**

    A ≈ 0 (与 bootstrap 噪声同量级) -> 动作可交换 -> **无时序结构**
                                      -> 我的对照对 option 不公平, benchmark 该改
    A 显著 ≠ 0                     -> 顺序有关 -> option 本应有用
                                      -> **我的 option 实现该背锅**

注意: 只看 I(a,b) ≠ 0 不够 —— 普通迁移(训练 a 让 b 更容易)不需要 option,
只要一个"正确的课程"。**只有不对称性才需要多步规划** (a→b 与 b→a 不等价)。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from run_lm4_wave import (  # noqa: E402
    WINDOW, StatMLP, assign_domains, build_feature_matrix, build_windows,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "wave"))
    ap.add_argument("--wfr-bands", type=int, default=13)
    ap.add_argument("--no-wfr", action="store_true")
    ap.add_argument("--no-lshell", action="store_true")
    ap.add_argument("--domains", type=int, default=6)
    ap.add_argument("--fine-bins", type=int, default=18)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--steps-pair", type=int, default=60,
                    help="每个动作训多少步")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "temporal_structure.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    t, dens, F, ok = build_feature_matrix(
        a.data, bands=a.wfr_bands, use_wfr=not a.no_wfr, use_lshell=not a.no_lshell)
    X, y_log = build_windows(t, dens, F, ok, stride=WINDOW)
    y_dom, _ = assign_domains(y_log, a.domains)
    d_feat = X.shape[-1]
    fine, _ = assign_domains(y_log, a.fine_bins)
    nf = int(fine.max()) + 1
    print("[data] %d 窗口 x %d x %d, 粗域 %d, fine bins %d"
          % (len(X), WINDOW, d_feat, a.domains, nf), flush=True)

    # 每个细区间切出 train / probe
    tr, pr = {}, {}
    for f in range(nf):
        idx = np.where(fine == f)[0]
        if len(idx) < 200:
            continue
        cut = int(len(idx) * 0.8)
        tr[f], pr[f] = idx[:cut], idx[cut:]
    keys = sorted(tr)
    print("[data] 可用细区间 %d 个 (%s)" % (len(keys), keys[:8]), flush=True)

    Xt = torch.from_numpy(X.astype(np.float32))
    Yt = torch.from_numpy(y_dom.astype(np.int64))

    def fresh(seed):
        torch.manual_seed(seed)
        m = StatMLP(n_feat=d_feat, n_classes=a.domains, hidden=a.d_model,
                    n_layers=a.n_layers, head_type="linear", norm="fixed").to(dev)
        return m

    def train_on(model, f, steps, seed):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        idx = tr[f].copy()
        np.random.RandomState(seed * 131 + steps).shuffle(idx)
        pos, it = 0, 0
        model.train()
        while it < steps:
            if pos + a.batch > len(idx):
                pos = 0
            b = idx[pos:pos + a.batch]
            pos += a.batch
            loss = torch.nn.functional.cross_entropy(
                model(Xt[b].to(dev)), Yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
        return model

    @torch.no_grad()
    def acc_on(model, f):
        model.eval()
        b = pr[f]
        return float((model(Xt[b].to(dev)).argmax(1) == Yt[b].to(dev)).float().mean())

    rng = np.random.RandomState(0)
    pairs = []
    while len(pairs) < a.pairs and len(keys) >= 2:
        x, y = int(rng.choice(keys)), int(rng.choice(keys))
        if x != y:
            pairs.append((x, y))
    pairs = list(dict.fromkeys(pairs))

    rows = []
    for si in range(a.seeds):
        seed = 42 + si
        for (aa, bb) in pairs:
            m = fresh(seed);                                          b_none = acc_on(m, bb)
            m = fresh(seed);                                          a_none = acc_on(m, aa)
            m = fresh(seed); train_on(m, bb, a.steps_pair, seed);      b_only = acc_on(m, bb)
            # ★ 纯迁移: **只**训 a, 直接测 b (不给 b 任何训练)。
            #   若它 ≈ b_none, 说明训 a 对 b 一点用都没有 -> 零结构。
            m = fresh(seed); train_on(m, aa, a.steps_pair, seed);      b_transfer = acc_on(m, bb)
            a_transfer = None
            m = fresh(seed); train_on(m, bb, a.steps_pair, seed);      a_transfer = acc_on(m, aa)
            m = fresh(seed); train_on(m, aa, a.steps_pair, seed);      a_only = acc_on(m, aa)
            m = fresh(seed)
            train_on(m, aa, a.steps_pair, seed)
            train_on(m, bb, a.steps_pair, seed + 7);                   b_after_a = acc_on(m, bb)
            m = fresh(seed)
            train_on(m, bb, a.steps_pair, seed)
            train_on(m, aa, a.steps_pair, seed + 7);                   a_after_b = acc_on(m, aa)
            rows.append(dict(seed=seed, a=aa, b=bb,
                             b_none=b_none, a_none=a_none,
                             b_only=b_only, a_only=a_only,
                             b_transfer=b_transfer,
                             transfer_gain=b_transfer - b_none,
                             b_after_a=b_after_a, a_after_b=a_after_b,
                             I_ab=b_after_a - b_only, I_ba=a_after_b - a_only,
                             asym=(b_after_a - b_only) - (a_after_b - a_only),
                             b_random=0.0))
        print("  seed=%d 完成 (%d 对)" % (seed, len(pairs)), flush=True)

    Iab = np.array([r["I_ab"] for r in rows]); Iba = np.array([r["I_ba"] for r in rows])
    A = np.array([r["asym"] for r in rows])
    print()
    print("=" * 92)
    print("一阶交互 (n=%d 对 x %d seed = %d 次测量)" % (len(pairs), a.seeds, len(rows)))
    print("=" * 92)
    print("  I(a,b) = acc_b(先a后b) − acc_b(只b)    均值 %+.4f   std %.4f" % (Iab.mean(), Iab.std()))
    print("  I(b,a)                                均值 %+.4f   std %.4f" % (Iba.mean(), Iba.std()))
    print("  不对称 A = I(a,b) − I(b,a)             均值 %+.4f   std %.4f" % (A.mean(), A.std()))
    bn = np.array([r["b_none"] for r in rows])
    bt = np.array([r["b_transfer"] for r in rows])
    tg = np.array([r["transfer_gain"] for r in rows])
    bo2 = np.array([r["b_only"] for r in rows])
    print("  ── 结构体检 (这三行才是关键) ──")
    print("     b 零训练 acc     均值 %.4f  (随机 = %.4f)" % (bn.mean(), 1.0 / a.domains))
    print("     只训 b 后 acc    均值 %.4f   <-- 若 ≈1.0 说明区间**平凡可分**" % bo2.mean())
    print("     纯迁移(只训a测b) 均值 %.4f   <-- 若 ≈b零训练, 说明**训 a 对 b 毫无影响**"
          % bt.mean())
    print("     迁移增益 Δ       均值 %+.4f  std %.4f" % (tg.mean(), tg.std()))
    print("  |A| > 0.05 的比例                      %.1f%%" % (100 * (np.abs(A) > 0.05).mean()))
    bs = np.array([A[np.random.RandomState(k).randint(0, len(A), len(A))].mean()
                   for k in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("  A 的 bootstrap 95%% CI = [%+.4f, %+.4f]" % (lo, hi))
    sig = bool(not (lo <= 0 <= hi))
    print()
    print("=" * 92)
    if sig:
        print("结论: **存在顺序依赖** (A 的 CI 不含 0)")
        print("      -> 任务有多步规划空间, option 本应有用")
        print("      -> **我的 option 实现该背锅** (选择非价值驱动 / 终止没用 / 视野太短)")
    else:
        print("结论: **顺序无关** (A 的 CI 含 0, 与噪声同量级)")
        print("      -> 这个 benchmark 里动作近似可交换, **不存在可供 option 抽象的时间结构**")
        print("      -> 我的 E9/E10 对照测的是 **benchmark 的性质, 不是 option 的价值**")
        print("      -> 要检验 option 必须换一个**有时序结构**的 benchmark")
    print("=" * 92)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"I_ab_mean": float(Iab.mean()), "I_ab_std": float(Iab.std()),
         "I_ba_mean": float(Iba.mean()), "I_ba_std": float(Iba.std()),
         "asym_mean": float(A.mean()), "asym_std": float(A.std()),
         "asym_ci": [float(lo), float(hi)], "order_dependent": sig,
         "frac_abs_A_gt_0.05": float((np.abs(A) > 0.05).mean()),
         "n_rows": len(rows), "pairs": pairs, "seeds": a.seeds,
         "steps_pair": a.steps_pair, "rows": rows}, indent=1, ensure_ascii=False))
    print("已写 %s" % a.out)


if __name__ == "__main__":
    main()
