#!/usr/bin/env python3
"""benchmark_b_matrix.py — 用户框架里的 **B 矩阵** (在 KeyDoorMDP 上)。

## 测什么

    B2/B3  A→B          适应速度 T_adapt
    B4/B5  A→B→C        反复适应
    B6/B7  A→B→A        **知识复用** T_A^(2) < T_A^(1)?
    B8/B9  多 regime     sample efficiency R(N)

对照: `primitive` (纯 Q-learning) vs `rediscover` (Q + 跨 regime 保留的技能库)。

## 为什么这个环境能回答, lm4 不能

    lm4      : A ≡ 0 (动作完全可交换)      -> option 无空间, 测不出
    KeyDoor  : A ∈ [-1,+1], 33% 显著       -> **目标导向的多步行为真的有价值**

## 判据 (用户给的)

    若 option 在某项上优于 primitive -> 那就是 option 体现价值的地方
    若各项都不优于 (且 CI 不含 0)     -> 才有资格讨论"option 不适用"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hibs_lnn.skill_agent import DEFAULT_REGIMES, run_regime_chain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", default="primitive,rediscover")
    ap.add_argument("--cycles", type=int, default=2,
                    help="regime 链重复几轮 (2 = A B C A B C)")
    ap.add_argument("--n-pos", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "b_matrix.json"))
    a = ap.parse_args()

    chain = DEFAULT_REGIMES * a.cycles
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    print("regime 链: %s" % [f"{r['key_pos']}/{r['door_pos']}/{r['goal_pos']}" for r in chain])
    print("模式: %s   seed x %d   episodes/regime=%d"
          % (modes, a.seeds, a.episodes_per))
    print()

    all_rows = {}
    for mode in modes:
        per_seed = []
        for si in range(a.seeds):
            rows, agent = run_regime_chain(mode=mode, chain=chain, n_pos=a.n_pos,
                                           episodes_per=a.episodes_per,
                                           seed=42 + si)
            per_seed.append(rows)
            print("  [%s seed=%d] T_adapt=%s  最终成功率=%s  n_options=%d"
                  % (mode, 42 + si, [r["t_adapt"] for r in rows],
                     [round(r["final_succ"], 2) for r in rows],
                     rows[-1]["n_options"]), flush=True)
        all_rows[mode] = per_seed

    print()
    print("=" * 96)
    print("★ 知识复用: 同一 regime 在第 2 轮 (重复出现) 是否恢复更快?")
    print("=" * 96)
    n_seg = len(chain)
    half = n_seg // a.cycles if a.cycles > 1 else n_seg
    print("%-14s %-28s %-28s %s" % ("mode", "第1轮 T_adapt (均值)", "第2轮 T_adapt (均值)", "Δ / 加速比"))
    summary = {}
    for mode in modes:
        arr = np.array([[r["t_adapt"] for r in rows] for rows in all_rows[mode]], float)
        first = arr[:, :half].mean(axis=1)
        second = arr[:, half:half * 2].mean(axis=1) if a.cycles > 1 else first
        d = second - first
        try:
            from scipy import stats
            t, p = stats.ttest_rel(first, second)
        except Exception:
            t, p = float("nan"), float("nan")
        speed = float(first.mean() / max(second.mean(), 1e-9))
        summary[mode] = {"first": float(first.mean()), "second": float(second.mean()),
                         "delta": float(d.mean()), "speedup": speed,
                         "p": float(p) if np.isfinite(p) else None}
        print("%-14s %-28.2f %-28.2f %+.1f (%.2fx) p=%.3f"
              % (mode, first.mean(), second.mean(), d.mean(), speed,
                 p if np.isfinite(p) else float("nan")))
    print()
    print("  判据: Δ < 0 且 p < 0.05 -> **复用成立** (第 2 轮确实更快)")
    print()
    print("=" * 96)
    print("逐段 T_adapt (跨 seed 均值 ± std)")
    print("=" * 96)
    for mode in modes:
        arr = np.array([[r["t_adapt"] for r in rows] for rows in all_rows[mode]], float)
        print("  %-14s %s" % (mode, " ".join("%.0f±%.0f" % (m, s)
                                             for m, s in zip(arr.mean(0), arr.std(0)))))
    print()
    if len(modes) >= 2 and "primitive" in modes:
        prim = np.array([[r["t_adapt"] for r in rows] for rows in all_rows["primitive"]], float)
        for mode in modes:
            if mode == "primitive":
                continue
            arr = np.array([[r["t_adapt"] for r in rows] for rows in all_rows[mode]], float)
            d = prim.mean(0) - arr.mean(0)      # 正 = option 更快
            print("  %-14s 相对 primitive 的加速 (逐段, 正=更快): %s"
                  % (mode, " ".join("%+.0f" % x for x in d)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"modes": modes, "seeds": a.seeds, "episodes_per": a.episodes_per,
         "chain": chain, "summary": summary, "rows": all_rows},
        indent=1, ensure_ascii=False))
    print("\n已写 %s" % a.out)


if __name__ == "__main__":
    main()
