#!/usr/bin/env python3
"""学习效率首次测量的分析 —— 判据在 `docs/learning_efficiency_spec.md` §4。

核心对照: **同一个 run 内部同时训练 `naive` 与 `replay`** ⇒ 天然配对
(同 seed、同域、同数据顺序、同模型初始化), 唯一的差别是"要不要复习旧域"。

主判据 = `retention_gain[d] = mean(start_level[k≥2]) − start_level[1]`
        ⇒ "重遇到这个域时, 你一上来就会多少" 相对第一次的改善。

判据(事前):
  ① 填充闸门: `_n_visits` 全部 ≥1 且 `steps_to_abs` 非全空, 否则判"指标未采集成功"
  ② 覆盖率: 报 `_already_ratio` / `_none_ratio`
  ③ 保留: `mean(replay − naive) > 0` 且自助法区间下界 > 0 ⇒ 有保留; 否则照实写"无保留"
"""
from __future__ import annotations

import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = 20000


def load(pat="results/lmeff_s*"):
    runs = {}
    for d in sorted(glob.glob(str(ROOT / pat))):
        f = Path(d) / "lm4_wave_results.json"
        if not f.exists():
            continue
        j = json.load(open(f))
        seed = int(Path(d).name.rsplit("_s", 1)[1])
        runs[seed] = j
    return runs


def rg_of(j, arm):
    e = (j.get(arm) or {}).get("efficiency") or {}
    return e


def boot_ci(x, n=B, seed=0):
    if not x:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    m = []
    for _ in range(n):
        m.append(st.mean([x[rng.randrange(len(x))] for _ in range(len(x))]))
    m.sort()
    return m[int(.025 * len(m))], m[int(.975 * len(m))]


runs = load()
print("=" * 92)
print(f"学习效率首次测量 —— {len(runs)} 个 run: {sorted(runs)}")
print("=" * 92)
if not runs:
    sys.exit("✗ 没有结果目录")

# ── ① 填充闸门 ────────────────────────────────────────────────────
print("\n① 填充闸门")
gate_ok = True
for seed, j in sorted(runs.items()):
    for arm in ("naive", "replay"):
        e = rg_of(j, arm)
        nv = e.get("_n_visits") or {}
        if not nv:
            print(f"  ✗ seed={seed} {arm}: _n_visits 空 ⇒ 指标未采集")
            gate_ok = False
            continue
        print(f"  seed={seed} {arm:6s} n_visits={nv}  already={e.get('_already_ratio'):.3f}"
              f"  none={e.get('_none_ratio'):.3f}  reachable_max={e.get('reachable_max')}")
print(f"  ⇒ 填充闸门: {'PASS' if gate_ok else '**FAIL(指标未采集成功)'}")

# ── ② 保留增益: 逐域对照 ──────────────────────────────────────────
print("\n" + "=" * 92)
print("② ★ 保留增益 retention_gain —— naive vs replay(同 seed 同域, 天然配对)")
print("=" * 92)
pairs = []          # (seed, dom, naive, replay)
doms = set()
for seed, j in sorted(runs.items()):
    en, er = rg_of(j, "naive"), rg_of(j, "replay")
    rn, rr = en.get("retention_gain") or {}, er.get("retention_gain") or {}
    for d in sorted(set(rn) | set(rr), key=lambda x: int(x[1:])):
        a, b = rn.get(d), rr.get(d)
        if a is None or b is None:
            continue
        doms.add(d)
        pairs.append((seed, d, float(a), float(b)))

print(f"{'域':<8}{'n seed':>7}{'naive 均值':>12}{'replay 均值':>13}{'Δ(replay−naive)':>18}")
per_dom = {}
for d in sorted(doms, key=lambda x: int(x[1:])):
    ps = [(a, b) for _, dd, a, b in pairs if dd == d]
    if not ps:
        continue
    na = st.mean([a for a, _ in ps]); rp = st.mean([b for _, b in ps])
    per_dom[d] = (na, rp, len(ps))
    print(f"{d:<8}{len(ps):>7}{na:>12.4f}{rp:>13.4f}{rp - na:>+18.4f}")

if pairs:
    dn = [a for _, _, a, _ in pairs]
    dr = [b for _, _, _, b in pairs]
    dif = [b - a for _, _, a, b in pairs]
    print(f"\n  全域汇总(n={len(pairs)} 对):")
    print(f"    naive  均值 {st.mean(dn):+.4f}  (std {st.pstdev(dn):.4f})")
    print(f"    replay 均值 {st.mean(dr):+.4f}  (std {st.pstdev(dr):.4f})")
    lo, hi = boot_ci(dif)
    n_pos = sum(1 for x in dif if x > 0)
    n_neg = sum(1 for x in dif if x < 0)
    print(f"    Δ(replay−naive) = {st.mean(dif):+.4f}  自助法 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    replay 更好的域: {n_pos}/{len(dif)}   naive 更好: {n_neg}   平: {len(dif)-n_pos-n_neg}")
    print(f"  ⇒ ③ 判据: mean>0 且 CI 下界>0 ? "
          f"{'**有保留**' if (st.mean(dif) > 0 and lo > 0) else '**未达到(照实写)'}")

# ── ③ 起点水平本身(不依赖阈值, 更贴近"记住了多少") ─────────────────
print("\n" + "=" * 92)
print("③ 起点水平 start_level(k=1 vs 以后) —— 绝对口径")
print("=" * 92)
print(f"{'域':<8}{'arm':<8}{'k=1 起点':>10}{'k≥2 均值':>11}{'起点极差':>10}")
for seed, j in sorted(runs.items()):
    for arm in ("naive", "replay"):
        sl = (rg_of(j, arm).get("start_level") or {})
        for d in sorted(sl, key=lambda x: int(x[1:])):
            v = sl[d]
            ks = sorted(v, key=lambda x: int(x))
            first, later = v[ks[0]], [v[k] for k in ks[1:]]
            if later:
                print(f"{d:<8}{arm:<8}{first:>10.3f}{st.mean(later):>11.3f}"
                      f"{max(later)-min(later):>10.3f}   seed={seed}")

# ── ④ 其它口径 ────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("④ 其它口径(更新效率 / 访问内增益)")
print("=" * 92)
for arm in ("naive", "replay"):
    wg, sp = [], []
    for seed, j in sorted(runs.items()):
        e = rg_of(j, arm)
        for d, v in (e.get("within_gain") or {}).items():
            wg += list(v.values())
        for d, v in (e.get("slope_early") or {}).items():
            sp += list(v.values())
    print(f"  {arm:6s} within_gain 均值 {st.mean(wg):+.5f} (n={len(wg)})"
          f" | slope_early 均值 {st.mean(sp):+.6f} (n={len(sp)})")
