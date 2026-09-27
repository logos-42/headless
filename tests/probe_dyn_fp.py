#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_dyn_fp.py — A 类否证实验的**决定性补充**。

⑨ 判据抓到: A1/A2 对 `chaos`(不稳定) 与 `harm`(稳定但错误) 的反应
**逐位相同**。这留下两种互相矛盾的解读:

  (甲) A 是个**发散的检测器** —— 它测的是"这个 option 把我带去了别的
       地方", 而不是"这个 option 坏"。若是甲, 那么一个**健康的探索性
       option** 也会被它标红 -> 它会把好东西标成风险 -> 不能用于决策。
  (乙) A 只是在两个故障上恰好都触发了, 但基线里它其实是安静的。

区分办法只有一个: **在 normal 场景(无故障)上看它误报多少。**
  · normal 上误报接近 0  -> 甲不成立, 它确实只对异常有反应(但不可分辨种类)
  · normal 上大量误报    -> 甲成立, A 类是"发散检测器", 只能当覆盖率辅助
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/Users/apple/Downloads/headless")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_three_state import FROZEN, SCENARIOS, run_arm  # noqa: E402

print(f"{'臂':>5}{'场景':>8}{'段':>5}{'转移数':>9}{'基线压力':>9}"
      f"{'触发风险':>9}{'option数':>9}   按 option")
print("-" * 96)
tot = {}
for kind in ("A1", "A2", "A3"):
    cfg = dict(FROZEN, rr_three_state=True, rr_dyn=kind)
    for sc in ("normal", "inert", "harm", "chaos"):
        segs, _ = run_arm(cfg, sc, 200, 0)
        for i, s in enumerate(segs):
            dy = s.get("dyn")
            if not dy:
                continue
            po = dy["per_option"]
            hit = sum(1 for v in po.values() if v["risk"])
            nz = {k: v for k, v in po.items() if v["n"] > 0}
            tot.setdefault((kind, sc), []).append(hit)
            if i in (0, 2, len(segs) - 1) or hit:
                nm = list(nz)[:2]
                det = " ".join(f"{k.split('_')[-1]}:{v['mean']:.3f}"
                               f"{'*' if v['risk'] else ''}" for k, v in
                               [(k, nz[k]) for k in nm])
                print(f"{kind:>5}{sc:>8}{i:>5}{dy['transitions']:>9}"
                      f"{dy['base_mean']:>9.3f}{hit:>9}{len(nz):>9}   {det}")

print("\n" + "=" * 96)
print("误报汇总 (风险触发次数 / 段数)")
print("=" * 96)
for kind in ("A1", "A2", "A3"):
    row = "  ".join(f"{sc}={sum(tot[(kind, sc)])}/{len(tot[(kind, sc)])}"
                    for sc in ("normal", "inert", "harm", "chaos"))
    print(f"  {kind:>4}   {row}")
print("""
判读: normal 是**无故障**场景。A 在 normal 上触发的次数就是**误报数**。
  normal=0 且 harm/chaos>0  -> A 只对异常有反应(但是"发散"还是"有害"它分不清)
  normal>0 且量级相当        -> A 是"发散检测器", 健康探索也会被标红
""")
