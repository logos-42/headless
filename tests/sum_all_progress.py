#!/usr/bin/env python3
"""汇总 lm4 / lm5 全部实验进展 —— 只读结果文件, 不凭记忆。"""
import json
import glob
import os
import numpy as np

R = "/work/liuyuanjie/headless/results"


def load(tag):
    p = os.path.join(R, tag, "lm4_wave_results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def lm4_row(tag):
    d = load(tag)
    if d is None:
        return None
    cfg = d.get("_config", {}) or d.get("config", {})
    def g(k):
        v = d.get(k)
        return v.get("final_mean_acc") if isinstance(v, dict) else None
    return {
        "tag": tag, "backbone": cfg.get("backbone", "?"),
        "norm": cfg.get("norm", "?"), "head": cfg.get("head", "?"),
        "cl": cfg.get("cl_method", "?"), "rr": cfg.get("replay_ratio", "?"),
        "naive": g("naive"), "replay": g("replay"),
        "joint": (d.get("joint") or {}).get("final_mean_acc"),
        "doms": cfg.get("domains", "?"),
    }


print("=" * 78)
print("LM4 —— 电磁波持续学习 (真实 RBSP-A 数据)")
print("=" * 78)

tags = sorted({os.path.basename(os.path.dirname(p))
               for p in glob.glob(os.path.join(R, "*", "lm4_wave_results.json"))})
rows = [r for r in (lm4_row(t) for t in tags) if r]
print(f"共 {len(rows)} 个结果目录\n")

# 只看关键组
groups = {
    "① 8-seed 稳健性验证 (fx_*)": [r for r in rows if r["tag"].startswith("fx_")],
    "② 干净 CL 方法矩阵 (c1_*)": [r for r in rows if r["tag"].startswith("c1_")],
    "③ SSM 原版对照 (q1_/q5_)": [r for r in rows if r["tag"].startswith(("q1_", "q5_"))],
    "④ 跨年 (cross_*)": [r for r in rows if r["tag"].startswith("cross_")],
}
for name, rs in groups.items():
    if not rs:
        print(f"{name}: (无)")
        continue
    print(f"{name}  n={len(rs)}")
    print("   %-22s %-8s %-8s %-7s %-7s %-7s %s" %
          ("tag", "backbone", "norm", "naive", "replay", "joint", "doms"))
    for r in rs:
        f = lambda v: ("%.4f" % v) if isinstance(v, (int, float)) else str(v)
        print("   %-22s %-8s %-8s %-7s %-7s %-7s %s" %
              (r["tag"], r["backbone"], r["norm"], f(r["naive"]),
               f(r["replay"]), f(r["joint"]), r["doms"]))
    rv = [r["replay"] for r in rs if isinstance(r["replay"], (int, float))]
    if rv:
        print("   -> replay: n=%d  %.4f ± %.4f  (CV %.1f%%)" %
              (len(rv), np.mean(rv), np.std(rv), 100 * np.std(rv) / max(1e-9, abs(np.mean(rv)))))
    print()

# 跨年明细
print("-" * 78)
print("跨年泛化 (2015 训练 -> 2016 测试)")
for tag in [t for t in tags if t.startswith("cross_")]:
    d = load(tag)
    if not d:
        continue
    ce = d.get("cross_external") or {}
    for meth, row in ce.items():
        if row:
            v = [x for x in row if x == x]
            print("   %-20s %-8s 平均 %.4f  各域 %s" %
                  (tag, meth, sum(v) / len(v), " ".join("%.3f" % x for x in row)))

print()
print("=" * 78)
print("LM5 —— 文本+电磁波+因果 多模态")
print("=" * 78)
for tag in sorted({os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(R, "*", "lm5_mm_results.json"))}):
    d = json.load(open(os.path.join(R, tag, "lm5_mm_results.json")))
    M = d["matrix"]
    fin = M[-1]
    print("\n   %s  backbone=%s" % (tag, d["config"].get("backbone", "?")))
    print("     最终: " + "  ".join("%s=%.3f" % (k, v) for k, v in zip(d["domains"], fin)))
    print("     平均 %.4f | 遗忘 %s" % (sum(fin) / len(fin),
                                     {k: round(v, 3) for k, v in d["forgets"].items()}))
print()
print("参照线: lm4 SSM 原版 joint 0.7213 / replay 0.5852±0.0621 (9 seed)")
print("        lm4 探针可达上限 3/6/12 域 = 0.9159 / 0.8065 / 0.6638")
