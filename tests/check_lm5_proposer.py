#!/usr/bin/env python3
"""查 lm5 三个价值函数臂的提议谱, 判断池/提议比是否健康。"""
import json
import os

R = "/work/liuyuanjie/headless/results"
for tag in ["bm5_value_s42", "bm5_value-nofb_s42", "bm5_random-matched_s42"]:
    p = f"{R}/{tag}/lm5_mm_results.json"
    if not os.path.exists(p):
        print(f"{tag:26s} (缺失)")
        continue
    d = json.load(open(p))
    f = d.get("proposer_freq")
    st = d.get("proposer_stats") or {}
    cv = st.get("value_cv")
    print("%-26s freq=%s" % (tag, f))
    print("%-26s n_cand=%s picks=%s covered=%s value_cv=%s" %
          ("", st.get("n_candidates"), st.get("n_proposed_total"),
           st.get("covered_frac"),
           round(cv, 4) if isinstance(cv, (int, float)) else cv))
    tr = d.get("proposer_trace") or []
    if tr:
        print("%-26s 调度轨迹(前6): %s" %
              ("", [(t["domain"], t["value"]) for t in tr[:6]]))
    print()
print("健康判据: picks << n_candidates (V35.19 的坑1); value_cv 越大说明价值函数")
print("          的区分度越强 (cv→0 表示被抹平成常数).")
