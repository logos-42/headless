#!/usr/bin/env python3
"""两天队列运行的完整汇总: Q1 天花板归因 / Q3 方法扫描 / Q4 任务定义 / Q5 多 seed.

用法: /work/liuyuanjie/envs/vllm-cu128/bin/python tests/summarize_two_days.py
"""
import json, glob, math, os, statistics as st
from pathlib import Path

RES = Path("results")


def load(tag):
    f = RES / tag / "lm4_wave_results.json"
    if not f.exists():
        return None
    return json.load(open(f))


def acc(d, key):
    v = d.get(key)
    return v.get("final_mean_acc") if v else None


def forget(d, key):
    v = d.get(key)
    return v.get("mean_forget") if v else None


def per_dom(d, key):
    v = d.get(key)
    return v.get("per_domain") if v else None


# ---------- 家族定义 ----------
FAMILIES = [
    # (家族名, tag列表, 说明)
    ("A 基线 (pool=last, 6域, 无L-shell)", ["lm4A_wfr", "q1_pool_last"], None),
    ("Q1 天花板归因", ["q1_pool_last", "q1_pool_cat_agg", "q1_pool_cat_agg_ls"], "仅 joint 上限"),
    ("Q3 agg-path 方法扫描", ["q3_agg_rr1", "q3_agg_rr2", "q3_agg_rr5", "q3_agg_rr10"], "pool=cat+agg"),
    ("Q3 L-shell 变体", ["q3_agg_ls_rr2", "q3_agg_ls_rr5"], "pool=cat+agg+ls"),
    ("Q5 agg-path 多 seed", ["q3_agg_rr2", "q5_agg_s7", "q5_agg_s2026", "q5_agg_s123",
                             "q5_agg_s1", "q5_agg_s2", "q5_agg_s3"], "rr=2.0"),
    ("Q4 域数扫描", ["q4_d3", "q4_d8", "q4_d12"], "3/8/12 域"),
    ("Q4 顺序与特征", ["q4_d6_shuf", "q4_d6_ls", "q4_d6_agg"], "随机序/L-shell/agg"),
]

ALL_TAGS = ["lm4A_wfr", "q1_pool_last", "q1_pool_cat_agg", "q1_pool_cat_agg_ls",
            "q3_agg_rr1", "q3_agg_rr2", "q3_agg_rr5", "q3_agg_rr10",
            "q3_agg_ls_rr2", "q3_agg_ls_rr5",
            "q5_agg_s7", "q5_agg_s2026", "q5_agg_s123", "q5_agg_s1", "q5_agg_s2", "q5_agg_s3",
            "q4_d3", "q4_d8", "q4_d12", "q4_d6_shuf", "q4_d6_ls", "q4_d6_agg"]

print("=" * 100)
print("表 1 · 所有 run 原始结果")
print("=" * 100)
print(f"{'tag':18s} {'域数':>4s} {'naive':>7s} {'replay':>7s} {'joint':>7s} {'保留率':>7s} {'遗忘':>7s}  关键配置")
print("-" * 100)
data = {}
for t in ALL_TAGS:
    d = load(t)
    if not d:
        print(f"{t:18s}  ❌ 缺失")
        continue
    data[t] = d
    c = d.get("_config", {})
    nv, rp, jo = acc(d, "naive"), acc(d, "replay"), acc(d, "joint")
    fg = forget(d, "replay")
    ret = (rp / jo * 100) if (rp and jo) else None
    cfg = f"pool={c.get('pool')} agg={int(bool(c.get('agg_path')))} ls={int(bool(c.get('use_lshell')))} " \
          f"shuf={int(bool(c.get('shuffle_domains')))} rr={c.get('replay_ratio')} seed={c.get('seed')}"
    f_nv = f"{nv:.4f}" if nv is not None else "  nan "
    f_rp = f"{rp:.4f}" if rp is not None else "  nan "
    f_jo = f"{jo:.4f}" if jo is not None else "  nan "
    f_ret = f"{ret:6.1f}%" if ret is not None else "   nan "
    f_fg = f"{fg:.4f}" if fg is not None else "  nan "
    print(f"{t:18s} {c.get('domains', 6):4d} {f_nv} {f_rp} {f_jo} {f_ret} {f_fg}  {cfg}")

print()
print("=" * 100)
print("表 2 · 家族统计 (多 seed 才有 std)")
print("=" * 100)
print(f"{'家族':38s} {'n':>2s} {'replay':>16s} {'joint':>16s} {'保留率':>16s}")
print("-" * 100)
for name, tags, note in FAMILIES:
    rs = [acc(data[t], "replay") for t in tags if t in data and acc(data[t], "replay") is not None]
    js = [acc(data[t], "joint") for t in tags if t in data and acc(data[t], "joint") is not None]
    rets = [acc(data[t], "replay") / acc(data[t], "joint") * 100
            for t in tags if t in data and acc(data[t], "replay") and acc(data[t], "joint")]

    def fmt(v):
        if not v:
            return "            nan  "
        if len(v) == 1:
            return f"          {v[0]:.4f}"
        return f"{st.mean(v):.4f}±{st.stdev(v):.4f}"
    print(f"{name:38s} {len(rs):2d} {fmt(rs):>16s} {fmt(js):>16s} {fmt(rets):>16s}")

print()
print("=" * 100)
print("表 3 · Q4 域数扫描 (任务定义)")
print("=" * 100)
print(f"{'tag':14s} {'域数':>4s} {'naive':>7s} {'replay':>7s} {'joint':>7s} {'保留率':>7s} {'遗忘':>7s}")
print("-" * 100)
for t in ["q4_d3", "lm4A_wfr", "q4_d8", "q4_d12"]:
    d = data.get(t)
    if not d:
        continue
    c = d.get("_config", {})
    nv, rp, jo = acc(d, "naive"), acc(d, "replay"), acc(d, "joint")
    ret = (rp / jo * 100) if (rp and jo) else None
    print(f"{t:14s} {c.get('domains', 6):4d} {nv:7.4f} {rp:7.4f} {jo:7.4f} "
          f"{ret:6.1f}% {forget(d,'replay'):7.4f}")

print()
print("=" * 100)
print("表 4 · 各 run 的 joint 分域天花板")
print("=" * 100)
for t in ["lm4A_wfr", "q1_pool_cat_agg", "q1_pool_cat_agg_ls", "q5_agg_s7", "q5_agg_s1",
          "q4_d3", "q4_d8", "q4_d12"]:
    d = data.get(t)
    if not d:
        continue
    pd_ = per_dom(d, "joint")
    if not pd_:
        continue
    vals = [pd_[k] for k in sorted(pd_, key=lambda x: int(x))]
    print(f"{t:20s} " + " ".join(f"{v:.3f}" for v in vals))

print()
print("=" * 100)
print("关键问题裁决")
print("=" * 100)
base_j = acc(data.get("lm4A_wfr", {}), "joint")
agg_j = acc(data.get("q1_pool_cat_agg", {}), "joint")
agg_ls_j = acc(data.get("q1_pool_cat_agg_ls", {}), "joint")
print(f"Q1 ① agg-path 能否把天花板从 {base_j:.4f} 拉高 (MLP 参考 0.8130)?")
print(f"     pool=cat+agg        joint = {agg_j:.4f}   ({agg_j-base_j:+.4f})")
print(f"     pool=cat+agg+lshell joint = {agg_ls_j:.4f}   ({agg_ls_j-base_j:+.4f})")
print(f"   → {'✅ 有效' if max(agg_j, agg_ls_j) > base_j + 0.02 else '❌ 无效, 0.70 天花板成立, 池化/agg-path 都不是瓶颈'}")
