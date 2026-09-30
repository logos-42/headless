#!/usr/bin/env python3
"""验证证据库 (`hibs_lnn/evidence_bank.py`)。

判据事前写在 `docs/stage5_evidence_bank_spec.md` §8, 本文件逐条实现:

  ① **不许平均**: 构造功劳相等但符号相反的极端组合, 断言输出**不是**它们的均值,
     且 bad/healthy 引用清单非空。
  ② **缺席 ≠ 健康**: 全部证据不可用时判定必须是 `unknown`, **不是** `good`。
  ③ **动力学不能升级**: 动力学全 healthy + outcome 缺失 ⇒ `unknown`, 不是 `good`。
  ④ 逐条证据按**自己的尺度**归一(换一条 GVF 的尺度不影响另一条)。
  ⑤ 互补性度量的边界: 完全冗余 ⇒ complementarity ≈ 0; 完全互补 ⇒ > 0。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hibs_lnn.evidence_bank import EvidenceBank, complementarity, _auc  # noqa: E402

OK = []
FAIL = []


def ck(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{extra}]" if extra else ""))


print("=" * 84)
print("① 不许平均 —— 符号相反的极端组合")
print("=" * 84)
B = EvidenceBank(dim=4, n_min=1, stale_hl=10 ** 9)
phi = np.ones(4)
# 人为把四条证据设成: 两条极大误差, 两条极小误差 ⇒ 均值会落在中间(中性)
for i, (name, *_rest) in enumerate(B.spec):
    e = B.ev[name]
    e.n = 50
    e.buf.clear()
    for _ in range(50):
        e.buf.append(1.0)
    e.err_ref = 1.0
    e.err_ema = 10.0 if i < 2 else 0.0
sup = B.support(step=100)
has_mean = any(isinstance(v, float) and abs(v - np.mean([10.0, 10.0, 0.0, 0.0])) < 1e-9
               for v in sup.values())
ck("①a 返回值里没有'四条误差的均值'这个数", not has_mean,
   f"verdict={sup['verdict']}")
ck("①b bad 引用清单非空(两条高误差被点名)", len(sup["bad"]) == 2,
   f"bad={[b['src'] for b in sup['bad']]}")
ck("①c verdict=bad(有 bad 票就不看均值)", sup["verdict"] == "bad")
ck("①d 每条证据的完整记录都在", len(sup["sources"]) == len(B.spec))

print()
print("=" * 84)
print("② 缺席 ≠ 健康 —— 全部证据不可用")
print("=" * 84)
B2 = EvidenceBank(dim=4, n_min=10, stale_hl=100)
sup2 = B2.support(step=0)
ck("②a 全部 low_coverage ⇒ verdict = unknown", sup2["verdict"] == "unknown",
   f"verdict={sup2['verdict']}")
ck("②b 四条都进 missing 且原因可追溯",
   len(sup2["missing"]) == 4 and all(m["why"] == "low_coverage" for m in sup2["missing"]))
ck("②c usable_n = 0(缺席没被当成 0 平均进去)", sup2["usable_n"] == 0)
ck("②d 结论**不是** good", sup2["verdict"] != "good")

print()
print("=" * 84)
print("③ 动力学证据不能升级")
print("=" * 84)
B3 = EvidenceBank(dim=4, n_min=10, stale_hl=10 ** 9)   # ★ n_min=10: 才能造出"缺覆盖"
# 三条动力学 healthy + reward(outcome)缺覆盖
for name, *_r in B3.spec:
    e = B3.ev[name]
    e.err_ref, e.err_ema = 1.0, 0.1 if name != "reward" else 0.0
    e.n = 50 if name != "reward" else 3        # reward 只有 3 次 < n_min=10
    e.buf.clear()
    for _ in range(50):
        e.buf.append(1.0)
sup3 = B3.support(step=100)
ck("③a 动力学全 healthy + outcome 缺失 ⇒ verdict = unknown",
   sup3["verdict"] == "unknown", f"verdict={sup3['verdict']}")
ck("③b 动力学票被标记 capped(可追溯但不升级)",
   any(h.get("capped") for h in sup3["healthy"]))
ck("③c reward 进 missing", any(m["src"] == "reward" for m in sup3["missing"]))

# 反面对照: outcome 也 healthy ⇒ 才允许 good
e = B3.ev["reward"]
e.n = 50
B3.support(step=100)
sup3b = B3.support(step=100)
ck("③d ★ 只有 outcome 也 healthy 时才判 good(反面对照)",
   sup3b["verdict"] == "good", f"verdict={sup3b['verdict']}")

print()
print("=" * 84)
print("④ 每条证据用自己的尺度")
print("=" * 84)
B4 = EvidenceBank(dim=4, n_min=1, stale_hl=10 ** 9)
for name, *_r in B4.spec:
    e = B4.ev[name]
    e.n = 50
    e.buf.clear()
    # transition 的尺度比 prediction 大 100 倍
    base = 100.0 if name == "transition" else 1.0
    for _ in range(50):
        e.buf.append(base)
    e.err_ref = base
    e.err_ema = base * 3.0        # 两条都是 3 倍自身尺度
sup4 = B4.support(step=100)
zs = {s["src"]: s["z"] for s in sup4["sources"].values()}
ck("④a 不同尺度的两条 GVF 给出相同的 z(尺度已归一)",
   abs(zs["transition"] - zs["prediction"]) < 1e-6,
   f"z(transition)={zs['transition']:.3f} z(prediction)={zs['prediction']:.3f}")
ck("④b 全部 z=3.0 ≥ z_bad ⇒ 四条都是 bad 票", sup4["verdict"] == "bad")

print()
print("=" * 84)
print("⑤ 互补性度量边界")
print("=" * 84)
rng = np.random.default_rng(0)
n = 400
shifted = np.array([0] * (n // 2) + [1] * (n - n // 2))

# (a) 完全冗余: 四条只是同一序列的缩放
base = np.concatenate([rng.normal(1.0, 0.1, n // 2),
                       rng.normal(4.0, 0.1, n - n // 2)])
red = {f"g{i}": base * (1 + 0.01 * i) for i in range(4)}
cr = complementarity(red, shifted)
ck("⑤a 完全冗余 ⇒ complementarity ≈ 0", abs(cr["complementarity"]) < 0.02,
   f"comp={cr['complementarity']:+.4f} best={cr['best_single']:.3f}")
ck("⑤b 冗余时 corr 很高", all(v > 0.99 for v in cr["corr_err"].values()),
   f"min corr={min(cr['corr_err'].values()):.4f}")

# (b) 完全互补: 每条只在**自己那一段**变化时升高 ⇒ 单条 AUC 都低, 并集高
comp_series = {}
seg = (n - n // 2) // 4          # ★ 变化后那一段再切成 4 份, 索引不能越界
for i in range(4):
    a = np.concatenate([rng.normal(1.0, 0.1, n // 2),
                        rng.normal(1.0, 0.1, n - n // 2)])
    lo_, hi_ = n // 2 + i * seg, n // 2 + (i + 1) * seg
    a[lo_:hi_] = rng.normal(5.0, 0.1, hi_ - lo_)
    comp_series[f"g{i}"] = a
cc = complementarity(comp_series, shifted, z_bad=1.5)
ck("⑤c 互补构造 ⇒ complementarity > 0", cc["complementarity"] > 0.02,
   f"comp={cc['complementarity']:+.4f} best_single={cc['best_single']:.3f} "
   f"any={cc['auc_any']:.3f} cons={cc['auc_consensus']:.3f}")

# (c) AUC 的正确性: 完美分离 = 1.0, 反向 = 0.0
ck("⑤d _auc 完美分离 = 1.0", abs(_auc(np.array([0., 0, 1, 1]), np.array([0, 0, 1, 1])) - 1.0) < 1e-9)
ck("⑤e _auc 完全反向 = 0.0", abs(_auc(np.array([1., 1, 0, 0]), np.array([0, 0, 1, 1])) - 0.0) < 1e-9)
ck("⑤f _auc 全并列 = 0.5", abs(_auc(np.array([1., 1, 1, 1]), np.array([0, 0, 1, 1])) - 0.5) < 1e-9)

print()
print("=" * 84)
print(f"{len(OK)}/{len(OK) + len(FAIL)} 通过")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print(f"  ✗ {f}")
print("=" * 84)
sys.exit(0 if not FAIL else 1)
