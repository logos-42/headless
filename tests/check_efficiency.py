#!/usr/bin/env python3
"""查: 现有结果里到底有没有记录"学习效率"(达标步数曲线)。
    lm4 的 JSON 有 efficiency / curves 字段; lm5 有 matrix(逐步行)。"""
import glob
import json
import os

R = "/work/liuyuanjie/headless/results"

print("=" * 80)
print("LM4: efficiency / curves 字段的填充情况")
print("=" * 80)
has_eff, empty_eff = [], []
for p in sorted(glob.glob(f"{R}/*/lm4_wave_results.json")):
    tag = os.path.basename(os.path.dirname(p))
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for meth in ("naive", "replay"):
        v = d.get(meth)
        if not isinstance(v, dict):
            continue
        eff = v.get("efficiency") or {}
        cur = v.get("curves") or {}
        if eff or cur:
            has_eff.append((tag, meth, len(eff), len(cur)))
        else:
            empty_eff.append(tag)
print("  有 learning-efficiency 数据的 run:")
if has_eff:
    for t, m, ne, nc in has_eff:
        print("    %-28s %-7s efficiency=%d 项  curves=%d 项" % (t, m, ne, nc))
else:
    print("    (无)")
print("  无 efficiency 数据的 run 数: %d (去重 %d)" %
      (len(empty_eff), len(set(empty_eff))))
if empty_eff:
    uniq = sorted(set(empty_eff))
    print("    " + ", ".join(uniq[:20]) + (" ..." if len(uniq) > 20 else ""))

print()
print("=" * 80)
print("LM5: matrix 的粒度 (能否算学习效率)")
print("=" * 80)
for p in sorted(glob.glob(f"{R}/lm5_*/lm5_mm_results.json")):
    tag = os.path.basename(os.path.dirname(p))
    d = json.load(open(p))
    M = d.get("matrix") or []
    cfg = d.get("config", {})
    print("  %-26s backbone=%-7s 步数=%s  矩阵 %dx%d" %
          (tag, cfg.get("backbone", "?"), cfg.get("text_steps", "?"),
           len(M), len(M[0]) if M else 0))

print()
print("=" * 80)
print("关键: matrix 的行是『学完第 i 个域后』的评估, 不是『训练中每 N 步』的评估,")
print("      所以只能算『域间遗忘』, 算不出『达标步数』这类学习效率。")
print("=" * 80)
