#!/usr/bin/env python3
"""L4 批(L3 的 seed 扩展)判定 —— **结果出来前就写死的口径**。

## 为什么先写这个

L3 的教训:判据本身错了会把「样本量不足」读成「未排除」,从而把一场盲跑
当成证据。所以本脚本在数据出来之前就固定三件事:

  ① 主口径 = **配对**(同 seed),非配对作敏感性 —— **两个都报**;
     若两口径结论不一致, 必须明写不一致, **不许挑有利的那个**。
  ② 判定是**三值**:`worse` / `ns` / `inconclusive`。
     `|Δ| < MDE` ⇒ **inconclusive**, 不是 `ns`。
  ③ `df` 从真实样本量推临界值(`n=5 ⇒ df=4 ⇒ 2.776`), 不用正态近似。

## 判定规则(事前)

对每个 option 档 vs `e9`:
  · `p < 0.05` 且 `Δ < 0`                ⇒ **worse**
  · `|Δ| < MDE`  (即本批无分辨力)         ⇒ **inconclusive**
  · 其余                                  ⇒ **ns**

并额外报:要得到 `p<0.05` 还需多少 seed(未达显著时)。

## 效率指标(事后, 新旧臂同一份代码)

`efficiency` 字段在 runner 里是**空的**(25/25 臂), 所以从 `any_time_curve`
与 `acc_matrix` 事后算, 保证 L3 与 L4 可比:
  · `anytime_auc`   = mean(any_time_curve)
  · `t_adapt_proxy` = 每域"切回后恢复到该域历史峰值 80% 所需步数"的中位数
"""
import json
import glob
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Student-t 双尾 p(不完全 beta; 自检见 selftest)────────────────
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = d = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 3e-16:
        d = 3e-16
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 3e-16:
            d = 3e-16
        c = 1.0 + aa / c
        if abs(c) < 3e-16:
            c = 3e-16
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 3e-16:
            d = 3e-16
        c = 1.0 + aa / c
        if abs(c) < 3e-16:
            c = 3e-16
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_p_two_tailed(t, df):
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def f_p_two_tailed(F, df1, df2):
    """双尾 F 检验 p(F | df1, df2)。F<=0 或 df<1 时返回 1."""
    if F <= 0 or df1 < 1 or df2 < 1:
        return 1.0
    p = _betai(df1 / 2.0, df2 / 2.0, df1 * F / (df1 * F + df2))
    return min(1.0, 2.0 * min(p, 1.0 - p))


def bootstrap_var_ratio(a, b, B=20000, seed=0):
    """var(b)/var(a) 的自助法 95% 分位区间。a, b 独立重采样。

    `a` = 基线臂样本, `b` = 处理臂样本 ⇒ 比值 R = var(基线)/var(处理)。
    R > 1 表示**处理臂方差更小**。
    """
    import random as _rnd
    rng = _rnd.Random(seed)
    out = []
    for _ in range(B):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        va, vb = st.pvariance(ra), st.pvariance(rb)
        if va > 0:
            out.append(vb / va)
    out.sort()
    if not out:
        return float("nan"), float("nan")
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


# 临界值表(双尾 0.05) —— 只用它反推"还需多少 seed"
_TC = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
       30: 2.042, 60: 2.000, 120: 1.980, 10 ** 6: 1.960}


def t_crit(df):
    if df in _TC:
        return _TC[df]
    ks = sorted(_TC)
    if df < ks[0] or df > ks[-1]:
        return 1.96
    lo = max(k for k in ks if k < df)
    hi = min(k for k in ks if k > df)
    return _TC[lo] + (_TC[hi] - _TC[lo]) * (df - lo) / (hi - lo)


def selftest():
    checks = [(2.776, 4, 0.05), (4.604, 4, 0.01), (0.0, 4, 1.0)]
    for t, df, want in checks:
        got = t_p_two_tailed(t, df)
        assert abs(got - want) < 2e-3, f"selftest 失败 t={t} df={df}: {got} vs {want}"
    print("✓ t 分布自检通过 (2.776/df4→0.0500, 4.604/df4→0.0100)")


# ── 数据装载 ──────────────────────────────────────────────────────
def load(patterns):
    """arm -> {seed: metrics}。同时吃 L3 与 L4 目录, 两者配置逐字相同。"""
    arms = {}
    for pat in patterns:
        for d in sorted(glob.glob(str(ROOT / pat))):
            f = os.path.join(d, "lm4_wave_results.json")
            if not os.path.exists(f):
                continue
            name = os.path.basename(d)
            # oakL3_goal_term_s100 / oakL4_goal_term_s100  ->  (goal_term, 100)
            stem = name.split("_", 1)[1]
            arm, _, seed = stem.rpartition("_s")
            if not arm or not seed.isdigit():
                continue
            rep = json.load(open(f)).get("replay", {})
            atc = rep.get("any_time_curve") or []
            acc = rep.get("final_mean_acc")
            if acc is None or not atc:
                continue
            arms.setdefault(arm, {})[int(seed)] = {
                "acc": float(acc),
                "auc": st.mean(atc),
                "atc": atc,
                "acc_matrix": rep.get("acc_matrix") or [],
                "efficiency_populated": bool(rep.get("efficiency")),
            }
    return arms


def t_adapt_proxy(acc_matrix, frac=0.8):
    """⚠ **已废弃, 不要用** —— 实测在所有臂上恒返回 1.0。

    原因: `acc_matrix` 是 300 步 × 6 域的**逐步准确率**, 但 6 个域在
    `--stream revisit` 下被反复访问, 列里没有"该域刚开始训练"的标记,
    所以"从谷底恢复到峰值 80%"的判定在相邻两步内就满足 ⇒ 步数恒 = 1。
    保留函数只为记录这次失败, **不参与任何报告**。

    真正要测 T_adapt 需要 runner 输出**逐域逐 step 的损失/准确率曲线**,
    而当前 runner 的 `curves` 字段是空的 (见 main 里的死字段报告)。
    """
    return None


def _deprecated_proxy_body(acc_matrix, frac=0.8):
    """每域"切回后恢复到该域历史峰值 frac 所需步数"的中位数。

    `acc_matrix` = rounds x domains 的逐步准确率。代理量, 不是 runner 内定义;
    对新旧臂用同一份代码算, 所以**可比**。
    """
    if not acc_matrix:
        return None
    n_dom = len(acc_matrix[0])
    steps = []
    for j in range(n_dom):
        col = [row[j] for row in acc_matrix if len(row) > j]
        if not col:
            continue
        peak = max(col)
        if peak <= 0:
            continue
        thr = frac * peak
        # 每个"从谷底回升"的区间: 找低于 thr 之后首次达到 thr 的步数
        below = False
        last_below = 0
        for i, v in enumerate(col):
            if v < thr:
                below = True
                last_below = i
            elif below:
                steps.append(i - last_below)
                below = False
    return st.median(steps) if steps else None


def verdict(delta, sd_pair, sd_a, sd_b, n, pre_specified="paired"):
    """三值判定 + 还需多少 seed。pre_specified 只是记录用途。"""
    if pre_specified == "paired":
        se = sd_pair / math.sqrt(n)
    else:
        se = math.sqrt(sd_a ** 2 / n + sd_b ** 2 / n)
    if se <= 0:
        return "ns", float("nan"), float("nan"), 0
    t = delta / se
    p = t_p_two_tailed(abs(t), n - 1)
    mde = t_crit(n - 1) * se
    need = 0
    if p >= 0.05:
        for nn in range(n + 1, 5000):
            if t_crit(nn - 1) * se * math.sqrt(n / nn) <= abs(delta):
                need = nn
                break
    if p < 0.05 and delta < 0:
        v = "worse"
    elif abs(delta) < mde:
        v = "inconclusive"
    else:
        v = "ns"
    return v, t, p, need


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results/oakL3_*,results/oakL4_*,results/oakL5_*",
                    help="逗号分隔的结果目录 glob")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    selftest()
    arms = load([p for p in args.prefix.split(",") if p])
    if "e9" not in arms:
        sys.exit("✗ 找不到 e9 臂")
    base = arms["e9"]
    print(f"\n{'臂':<20}{'n':>4}{'acc 均值':>11}{'std':>8}")
    for k in sorted(arms, key=lambda a: -st.mean([v['acc'] for v in arms[a].values()])):
        v = [x["acc"] for x in arms[k].values()]
        print(f"{k:<20}{len(v):>4}{st.mean(v):>11.4f}{st.pstdev(v):>8.4f}")

    print(f"\n{'=' * 96}")
    print("对 e9 的比较 —— 配对(主口径) / 非配对(敏感性)。`inconclusive` = 本批无分辨力")
    print(f"{'=' * 96}")
    hdr = (f"{'臂':<20}{'|Δ|':>8}{'配对 p':>9}{'配对判定':>11}"
           f"{'非配对 p':>10}{'非配对判定':>12}{'一致?':>7}{'还需 seed':>10}")
    print(hdr)
    mismatch = []
    for k in ["goal", "goal_term", "goal_term_override", "fixed"]:
        if k not in arms:
            continue
        shared = sorted(set(arms[k]) & set(base))
        if len(shared) < 3:
            continue
        a = [arms[k][s]["acc"] for s in shared]
        b = [base[s]["acc"] for s in shared]
        d = [x - y for x, y in zip(a, b)]
        delta = st.mean(d)
        vp, tp, pp, np_ = verdict(delta, st.pstdev(d), st.pstdev(a), st.pstdev(b),
                                  len(shared), "paired")
        vu, tu, pu, nu = verdict(delta, st.pstdev(d), st.pstdev(a), st.pstdev(b),
                                 len(shared), "unpaired")
        ok = "✓" if vp == vu else "✗ 不一致"
        if vp != vu:
            mismatch.append(k)
        print(f"{k:<20}{abs(delta):>8.4f}{pp:>9.4f}{vp:>11}{pu:>10.4f}{vu:>12}{ok:>7}"
              f"{np_ or nu:>10}")

    if mismatch:
        print(f"\n⚠ 配对与非配对结论**不一致**的臂: {mismatch}")
        print("  按纪律: 明写不一致, 不挑有利口径。两口径都在上表。")

    # ── ★ 方差口径:H2 / H2b / H3(判定规则事前写在 docs/l5_variance_replication_spec.md)──
    print(f"\n{'=' * 96}")
    print("★ 方差口径 —— H2: 闭环档的跨 seed 方差是否低于基线")
    print(f"{'=' * 96}")
    VAR_ARMS = ["goal_term", "goal_term_override"]
    print(f"{'臂':<20}{'n':>4}{'sd':>9}{'var':>12}{'R=var(e9)/var(臂)':>19}"
          f"{'F p(双尾)':>11}   自助法 95% CI (B={args.bootstrap})")
    var_res = {}
    for k in ["goal_term", "goal_term_override", "goal", "fixed"]:
        if k not in arms:
            continue
        av = [x["acc"] for x in arms[k].values()]
        bv = [x["acc"] for x in base.values()][:len(av)]
        if len(av) < 3 or len(bv) < 3:
            continue
        va, vb = st.pvariance(av), st.pvariance(bv)
        R = vb / va if va > 0 else float("nan")
        p = f_p_two_tailed(R, len(bv) - 1, len(av) - 1)
        if len(av) < 8:
            # 方差比在 n<8 时不可用(自助法区间会宽到 [0.01, 24.8] 这种程度, 无信息)
            var_res[k] = dict(n=len(av), sd=st.pstdev(av), var=va, R=R, p=p,
                              ci=(float("nan"), float("nan")), usable=False)
            print(f"{k:<20}{len(av):>4}{st.pstdev(av):>9.4f}{va:>12.6f}{R:>19.3f}"
                  f"{p:>11.5f}   ★ n<8, 方差比不可用")
            continue
        lo, hi = bootstrap_var_ratio(av, bv, B=args.bootstrap)
        var_res[k] = dict(n=len(av), sd=st.pstdev(av), var=va, R=R, p=p,
                          ci=(lo, hi), usable=True)
        print(f"{k:<20}{len(av):>4}{st.pstdev(av):>9.4f}{va:>12.6f}{R:>19.3f}"
              f"{p:>11.5f}   [{lo:.2f}, {hi:.2f}]")

    if var_res:
        # H3 天花板检查 —— **先过这一条才解释 H2**
        # ★ 作用域: **只对 H2 的比较臂集**(e9 + VAR_ARMS)求均值极差。
        #   `goal` 是 H2b 的机制对照、`fixed` 是已判定档且 n=5, 二者都不在 H2 的比较里;
        #   把它们算进来会让"均值极差"反映**别的东西**(goal 确实有害 / fixed 样本不足),
        #   而不是"被比较的臂是否可比"。见 spec §3 的 2026-09-29 修订记录。
        h3_arms = ["e9"] + [k for k in VAR_ARMS if k in arms]
        means = {k: st.mean([x["acc"] for x in arms[k].values()]) for k in h3_arms}
        spread = max(means.values()) - min(means.values())
        margin = 1.0 - max(means.values())
        h3 = (spread < 0.02) and (margin > 0.10)
        print(f"\n  H3 天花板检查 [作用域: {', '.join(h3_arms)}]")
        for k, m in sorted(means.items(), key=lambda x: -x[1]):
            print(f"      {k:<20}{m:.4f}")
        print(f"    均值极差 {spread:.4f} (需 <0.02) | "
              f"距上限 {margin:.4f} (需 >0.10) ⇒ {'PASS' if h3 else '★ FAIL'}")
        if not h3:
            print("     ⇒ H3 不过: 方差差异可能是饱和伪影, **H2 判为无效**(无论 p 多小)")

        # H2 判定(事前规则)
        ok = [k for k in VAR_ARMS if k in var_res and var_res[k].get("usable")
              and var_res[k]["p"] < 0.05 and var_res[k]["ci"][0] > 1.0]
        hit = [k for k in VAR_ARMS if k in var_res and var_res[k].get("usable")]
        if h3 and len(ok) == len(hit) and hit:
            v2 = "★ **复现**"
        elif h3 and ok:
            v2, = [f"部分复现({','.join(ok)} 通过, 其余未过)"]
        else:
            v2 = "**未复现**" if h3 else "无效(H3 未过)"
        print(f"  H2 判定: {v2}")
        print(f"     规则: F p<0.05 且 自助法 CI 下界>1, 两臂都要满足")

        # H2b 机制
        if "goal" in var_res:
            g = var_res["goal"]
            print(f"  H2b 机制(`goal` 无 β_o): R={g['R']:.3f}, p={g['p']:.5f}, "
                  f"CI=[{g['ci'][0]:.2f}, {g['ci'][1]:.2f}] ⇒ "
                  f"{'方差降低来自 option 层本身' if g['p'] < 0.05 and g['ci'][0] > 1 else 'β_o 是必要条件'}")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(var_res, indent=2, default=str,
                                                     ensure_ascii=False))
            print(f"  方差结果已写: {args.json_out}")

    eff = [k for k, v in arms.items()
           for x in v.values() if x["efficiency_populated"]]
    print(f"\n★ runner 死字段报告 (与 leo 点名的一等指标直接相关):")
    print(f"  `efficiency` 被填充的臂数: {len(eff)} / {len(arms)}"
          f"   ⇒ 学习/更新效率**从未被 runner 采集过**")
    print(f"  `curves`     同样为空 ⇒ 逐域逐 step 的学习曲线也没有记录")
    print(f"  ⇒ 本报告里的 `anytime AUC` 是**唯一**可用的效率类代理"
          f"(= mean(any_time_curve), 逐步准确率的平均)")
    print(f"  ⇒ T_adapt 无法从现有产物算出, 需改 runner —— 那是另一个变量, 本批不做")

    print(f"\n{'臂':<20}{'anytime AUC':>13}{'final acc':>11}{'峰值':>9}")
    for k in sorted(arms, key=lambda a: -st.mean([v['acc'] for v in arms[a].values()])):
        v = list(arms[k].values())
        print(f"{k:<20}{st.mean(x['auc'] for x in v):>13.4f}"
              f"{st.mean(x['acc'] for x in v):>11.4f}"
              f"{st.mean(max(x['atc']) for x in v):>9.4f}")


if __name__ == "__main__":
    main()
