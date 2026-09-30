#!/usr/bin/env python3
"""学习效率指标的单元测试 —— **用人造曲线**。判据见 `docs/learning_efficiency_spec.md`。

要验的(都是"这个指标会不会自欺"):
  ① 学不到(全程低于阈值) ⇒ 达标步数必须是 `None`, **不是**回填自己的峰值
  ② ★ **起点已超阈 ⇒ 记 `"already"`, 不是 0/1 步**
     —— 把"早就知道"与"1 步学会"混成一个数字, 会让 revisit 流下所有域
     都被报成"1 步学会", 复用增益机械归零(**冒烟测实测到过**)。
  ③ 第二次访问起点更高 ⇒ `retention_gain > 0`(记住了)
  ④ 两次起点一样 ⇒ `retention_gain == 0`; 起点更低 ⇒ `< 0`(遗忘)
  ⑤ 没有独立的可达上限 ⇒ **不算绝对口径**, 且不退回臂自己的 max
  ⑥ 早期斜率 = Δacc/Δstep
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    "rlw", str(Path(__file__).resolve().parents[1] / "tests" / "run_lm4_wave.py"))
rlw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rlw)

eff = rlw.efficiency_report
OK, FAIL = [], []
RM = 0.8065          # 6 域的可达上限; 50% 阈值 = 0.40325, 90% = 0.72585


def ck(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{extra}]" if extra else ""))


def curve(pairs):
    return [(int(s), float(a)) for s, a in pairs]


print("=" * 88)
print("① 学不到 ⇒ None(不回填自己的峰值)")
print("=" * 88)
low = {0: [curve([(0, 0.10), (50, 0.15), (100, 0.20)])]}
r = eff({}, [0], reachable_max=RM, visits=low)
abs1 = r["steps_to_abs"]["D0"]["1"]["abs"]
ck("①a 50% 够不着 ⇒ None", abs1["50"] is None, f"50→{abs1['50']}")
ck("①b 90% 够不着 ⇒ None", abs1["90"] is None)
ck("①c ★ 没回填成自己的峰值步数(旧实现会给出 0)", abs1["50"] is None)
ck("①d 旧相对口径里确实是 0(证明缺陷真实存在)",
   r["rel_steps"]["D0"]["1"]["50"] == 0)
ck("①e 未达标占比 = 1.0", abs(r["_none_ratio"] - 1.0) < 1e-9)

print()
print("=" * 88)
print("② ★ 起点已超阈 ⇒ 'already'(不是 1 步)")
print("=" * 88)
# 起点 0.60 > 50% 阈值 0.403; 但 90% 阈值 0.726 起点未达
al = {0: [curve([(0, 0.60), (20, 0.70), (60, 0.80)])]}
r2 = eff({}, [0], reachable_max=RM, visits=al)
a2 = r2["steps_to_abs"]["D0"]["1"]["abs"]
ck("②a 起点 0.60 ≥ 50%阈值 ⇒ 'already'", a2["50"] == "already", f"50→{a2['50']}")
ck("②b ★ 不是 0 也不是 1", a2["50"] not in (0, 1))
ck("②c 90% 阈值(0.726)起点未达 ⇒ 正常算步数",
   isinstance(a2["90"], int) and a2["90"] == 60, f"90→{a2['90']}")
ck("②d already 占比 = 0.5(两个阈值里一个 already)",
   abs(r2["_already_ratio"] - 0.5) < 1e-9, f"{r2['_already_ratio']}")

print()
print("=" * 88)
print("③ 第二次访问起点更高 ⇒ 有保留")
print("=" * 88)
v1 = curve([(0, 0.10), (50, 0.30), (100, 0.60), (150, 0.70)])     # 起点 0.10
v2 = curve([(0, 0.55), (10, 0.62), (50, 0.70)])                    # 起点 0.55
r3 = eff({}, [0], reachable_max=RM, visits={0: [v1, v2]})
ck("③a 第1次起点 = 0.10", abs(r3["start_level"]["D0"]["1"] - 0.10) < 1e-9)
ck("③b 第2次起点 = 0.55", abs(r3["start_level"]["D0"]["2"] - 0.55) < 1e-9)
ck("③c ★ retention_gain = +0.45(记住了)",
   r3["retention_gain"]["D0"] is not None and abs(r3["retention_gain"]["D0"] - 0.45) < 1e-9,
   f"gain={r3['retention_gain']['D0']}")
ck("③d within_gain 两次都落在 [-1,1]",
   all(-1.0 <= v <= 1.0 for v in r3["within_gain"]["D0"].values()),
   f"{r3['within_gain']['D0']}")
ck("③e n_visits = 2", r3["_n_visits"]["D0"] == 2)

print()
print("=" * 88)
print("④ 起点一样 ⇒ 无保留; 起点更低 ⇒ 遗忘")
print("=" * 88)
r4 = eff({}, [0], reachable_max=RM, visits={0: [v1, list(v1)]})
ck("④a ★ retention_gain == 0(诚实报'无保留')",
   abs(r4["retention_gain"]["D0"]) < 1e-9, f"gain={r4['retention_gain']['D0']}")
v3 = curve([(0, 0.02), (50, 0.30), (100, 0.60)])
r4b = eff({}, [0], reachable_max=RM, visits={0: [v1, v3]})
ck("④b ★ 第二次起点更低 ⇒ retention_gain < 0(遗忘)",
   r4b["retention_gain"]["D0"] < 0, f"gain={r4b['retention_gain']['D0']}")

print()
print("=" * 88)
print("⑤ 没有独立可达上限 ⇒ 不算绝对口径, 也不退回臂自己的 max")
print("=" * 88)
r5 = eff({}, [0], reachable_max=None, visits={0: [v1, v2]})
ck("⑤a 标记 _no_reachable_max", r5["_no_reachable_max"] is True)
ck("⑤b steps_to_abs 的每次访问子表全为空",
   all(all(kv.get("abs") == {} for kv in v.values()) for v in r5["steps_to_abs"].values()),
   f"{r5['steps_to_abs']}")
ck("⑤c ★ 没有偷偷用曲线峰值当基准",
   r5["_none_ratio"] is None and r5["_already_ratio"] is None)
ck("⑤d 但 start_level / retention_gain 仍可算(不依赖阈值)",
   r5["retention_gain"]["D0"] == r5["retention_gain"]["D0"],
   f"gain={r5['retention_gain']['D0']}")

print()
print("=" * 88)
print("⑥ 早期斜率 = Δacc/Δstep")
print("=" * 88)
sl = {0: [curve([(0, 0.10), (25, 0.35), (50, 0.60), (100, 0.70)])]}
r6 = eff({}, [0], reachable_max=RM, visits=sl, K_slope=50)
ck("⑥a slope ≈ 0.01", abs(r6["slope_early"]["D0"]["1"] - 0.01) < 1e-9,
   f"slope={r6['slope_early']['D0']['1']}")

print()
print("=" * 88)
print(f"{len(OK)}/{len(OK) + len(FAIL)} 通过")
for f in FAIL:
    print(f"  ✗ {f}")
print("=" * 88)
sys.exit(0 if not FAIL else 1)
