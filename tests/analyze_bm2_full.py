#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LM4 + LM5 **修正版** benchmark 全臂汇总 (2026-09-14, bm2_* / bm5b_* + 任务流臂)。

用途: 直接打印 `docs/bm_benchmark_results.md` 里的表格行 (mean±std 用 ddof=1, Welch t 双侧 p)。
口径:
  - LM4 `random-matched` 主口径 = 与 `value` 臂同脚本修订版的 driver 日志值 (原 json 被 int64
    序列化 bug 截断); "当前目录口径" 单列为敏感性 (seed 1/7 事后被另一路会话用改过的脚本覆盖)。
  - 任务流臂来自更早的修订版 (`bm_bins_*` 05:16 / `bm5_*` 05:43 UTC), 严格可比性有保留。
无 scipy 依赖: t 分布双侧 p 用不完全 beta 自实现。

用法: scp 到服务器 /tmp 后
  /work/liuyuanjie/envs/vllm-cu128/bin/python /tmp/analyze_bm2_full.py
"""
# -*- coding: utf-8 -*-
"""修正版结算: LM4 rm 用与 value 同修订版 (原 driver 日志) 的值; 另给 08:14 后重跑作敏感性."""
import json, math, os
import numpy as np
R = "/work/liuyuanjie/headless/results"

def t_p2(t, df):
    # 无 scipy: 用不完全 beta
    def betacf(a,b,x):
        MAXIT,EPS,FPMIN=200,3e-16,1e-300
        qab,qap,qam=a+b,a+1.0,a-1.0; c=1.0; d=1.0-qab*x/qap
        if abs(d)<FPMIN: d=FPMIN
        d=1.0/d; h=d
        for m in range(1,MAXIT+1):
            m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2))
            d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; de=d*c; h*=de
            if abs(de-1.0)<EPS: break
        return h
    def betai(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        lb=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
        bt=math.exp(lb+a*math.log(x)+b*math.log(1-x))
        if x<(a+1)/(a+b+2): return bt*betacf(a,b,x)/a
        return 1.0-bt*betacf(b,a,1-x)/b
    return betai(df/2.0,0.5,df/(df+t*t))

def welch(a,b):
    a=np.array([x for x in a if x is not None and x==x],dtype=float)
    b=np.array([x for x in b if x is not None and x==x],dtype=float)
    if len(a)<2 or len(b)<2: return None
    va,vb=a.var(ddof=1),b.var(ddof=1); na,nb=len(a),len(b)
    se=math.sqrt(va/na+vb/nb); d=a.mean()-b.mean()
    if se==0: return dict(delta=float(d),se=0.0,t=float('nan'),df=float('nan'),p=float('nan'))
    t=d/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    return dict(delta=float(d),se=float(se),t=float(t),df=float(df),p=float(t_p2(t,df)))

MS={"mean":lambda v: float(np.mean(v)),"std":lambda v: float(np.std(v,ddof=1))}

def lm4(pat,key="replay"):
    out={}
    for s in ["42","1","7"]:
        p=f"{R}/{pat}/lm4_wave_results.json".replace("{s}",s)
        p=f"{R}/"+pat.format(s=s)+"/lm4_wave_results.json"
        try:
            j=json.load(open(p)); r=j[key]
            M=r["acc_matrix"]; last=M[-1]
            out[s]=dict(final=r["final_mean_acc"],any_time=r["any_time_acc"],
                        worst=r["worst_case_forget"],meanf=r["mean_forget_all"],last=last)
        except Exception as e:
            out[s]=None
    return out

def lm5(pat):
    out={}
    for s in ["42","1","7"]:
        p=f"{R}/"+pat.format(s=s)+"/lm5_mm_results.json"
        j=json.load(open(p)); M=j["matrix"]; last=M[-1]
        out[s]=dict(final=float(np.mean([x for x in last if x is not None])),any_time=j["any_time_acc"],
                    worst=j["worst_case_forget"],meanf=j["mean_forget_all"],last=last,domains=j["domains"])
    return out

def line(name,recs,keys=("final","any_time","worst","meanf")):
    vals={k:[recs[s][k] for s in ["42","1","7"] if recs.get(s)] for k in keys}
    txt=[]
    for k in keys:
        v=[x for x in vals[k] if x is not None]
        txt.append("—" if not v else (("%.4f±%.4f"%(MS["mean"](v),MS["std"](v))) if len(v)>1 else "%.4f"%MS["mean"](v)))
    seeds=" | ".join("%s:%s"%(s, ("%.4f"%recs[s]["final"]) if recs.get(s) else "—") for s in ["42","1","7"])
    return "| %s | %s | %s |" % (name, " | ".join(txt), seeds)

# ---- LM4 ----
V={n:lm4(n) for n in ["bm2_value_s{s}","bm2_value-nofb_s{s}","bm2_random_s{s}","bm_bins_fixed_s{s}","bm_bins_perm_s{s}","bm_bins_revisit_s{s}","bm_bins_nonstationary_s{s}"]}
CUR=lm4("bm2_random-matched_s{s}")
# 修订版一致口径 (原 driver 日志, s1/s7 的 json 已被后续修订版重跑覆盖)
V1={"42":dict(final=0.7635,any_time=None,worst=None,meanf=0.1178,last=None),
    "1":dict(final=0.7542,any_time=None,worst=None,meanf=0.1982,last=None),
    "7":dict(final=0.7594,any_time=None,worst=None,meanf=0.1116,last=None)}
V["bm2_random-matched_s{s}"]=V1

print("### LM4 表行 (final | any_time | worst | meanf | 逐seed)")
for n,v in V.items():
    print(line(n,v))
print()
print("### LM4 random-matched 当前目录口径 (混合修订版, 仅敏感性)")
print(line("rm_cur",CUR))
print()
print("### LM4 关键对比")
def cmp4(a,b,key="final",label=""):
    w=welch([V[a][s][key] for s in ["42","1","7"]],[V[b][s][key] for s in ["42","1","7"]])
    return (label or (a+" vs "+b+" ["+key+"]")), w
for a,b,k in [("bm2_value_s{s}","bm2_random-matched_s{s}","final"),
              ("bm2_value_s{s}","bm2_random-matched_s{s}","meanf"),
              ("bm2_value_s{s}","bm2_value-nofb_s{s}","final"),
              ("bm2_value_s{s}","bm2_value-nofb_s{s}","any_time"),
              ("bm2_value_s{s}","bm2_random_s{s}","final"),
              ("bm2_value_s{s}","bm_bins_perm_s{s}","final"),
              ("bm2_value_s{s}","bm_bins_fixed_s{s}","final"),
              ("bm2_value_s{s}","bm_bins_revisit_s{s}","final"),
              ("bm2_value_s{s}","bm2_random_s{s}","any_time"),
              ("bm2_value_s{s}","bm_bins_fixed_s{s}","worst")]:
    l,w=cmp4(a,b,k)
    print("| %s | %s | Δ=%+.4f | SE=%.4f | t=%+.2f | df=%.1f | p=%.4f | %s |"%(l,k,w["delta"],w["se"],w["t"],w["df"],w["p"],"★" if abs(w["t"])>=2 else "ns"))
# 敏感性: value vs 当前 rm 目录
w=welch([V["bm2_value_s{s}"][s]["final"] for s in ["42","1","7"]],[CUR[s]["final"] for s in ["42","1","7"]])
print("| [敏感性] value vs rm(当前目录) | final | Δ=%+.4f | t=%+.2f | p=%.4f | %s |"%(w["delta"],w["t"],w["p"],"★" if abs(w["t"])>=2 else "ns"))
w=welch([V["bm2_value_s{s}"][s]["any_time"] for s in ["42","1","7"]],[CUR[s]["any_time"] for s in ["42","1","7"]])
print("| [敏感性] value vs rm(当前目录) | anytime | Δ=%+.4f | t=%+.2f | p=%.4f | %s |"%(w["delta"],w["t"],w["p"],"★" if abs(w["t"])>=2 else "ns"))
# value vs nofb 逐 seed / 检查 picks 是否一致
for s in ["42","1","7"]:
    a=json.load(open(f"{R}/bm2_value_s{s}/lm4_wave_results.json"))["replay"]
    b=json.load(open(f"{R}/bm2_value-nofb_s{s}/lm4_wave_results.json"))["replay"]
    print("lm4 value vs nofb seed %s picks_identical=%s freq_identical=%s"%(s,
        [t["picks"] for t in a["proposer_trace"]]==[t["picks"] for t in b["proposer_trace"]],
        a.get("proposer_freq")==b.get("proposer_freq")))

print()
# ---- LM5 ----
L={n:lm5(n) for n in ["bm5b_value_s{s}","bm5b_value-nofb_s{s}","bm5b_random-matched_s{s}","bm5_fixed_s{s}","bm5_perm_s{s}","bm5_revisit_s{s}","bm5_nonstationary_s{s}"]}
print("### LM5 表行")
for n,v in L.items():
    print(line(n,v))
print()
print("### LM5 逐域 (末轮, 3 seed 均值)")
doms=L["bm5b_value_s{s}"]["42"]["domains"]
print("| 臂 | "+" | ".join(doms)+" |")
for n,v in L.items():
    row=[]
    for i in range(len(doms)):
        xs=[v[s]["last"][i] for s in ["42","1","7"] if v[s]["last"][i] is not None]
        row.append("%.3f"%np.mean(xs) if xs else "—")
    print("| %s | %s |"%(n," | ".join(row)))
print()
print("### LM5 关键对比")
for a,b,k in [("bm5b_value_s{s}","bm5b_random-matched_s{s}","final"),
              ("bm5b_value_s{s}","bm5b_random-matched_s{s}","any_time"),
              ("bm5b_value_s{s}","bm5b_random-matched_s{s}","worst"),
              ("bm5b_value_s{s}","bm5b_random-matched_s{s}","meanf"),
              ("bm5b_value_s{s}","bm5b_value-nofb_s{s}","final"),
              ("bm5b_value_s{s}","bm5_perm_s{s}","final"),
              ("bm5b_value_s{s}","bm5_fixed_s{s}","final"),
              ("bm5b_value_s{s}","bm5_nonstationary_s{s}","final"),
              ("bm5b_value_s{s}","bm5_perm_s{s}","any_time"),
              ("bm5b_random-matched_s{s}","bm5_fixed_s{s}","final")]:
    w=welch([L[a][s][k] for s in ["42","1","7"]],[L[b][s][k] for s in ["42","1","7"]])
    print("| %s vs %s | %s | Δ=%+.4f | SE=%.4f | t=%+.2f | df=%.1f | p=%.4f | %s |"%(a,b,k,w["delta"],w["se"],w["t"],w["df"],w["p"],"★" if abs(w["t"])>=2 else "ns"))
for s in ["42","1","7"]:
    a=json.load(open(f"{R}/bm5b_value_s{s}/lm5_mm_results.json"))
    b=json.load(open(f"{R}/bm5b_value-nofb_s{s}/lm5_mm_results.json"))
    print("lm5 value vs nofb seed %s matrix_identical=%s picks=%s"%(s,a["matrix"]==b["matrix"],[t["domain"] for t in a["proposer_trace"]]))
# CV
for n,v in L.items():
    f=[v[s]["final"] for s in ["42","1","7"]]
    print("CV %-24s %.1f%%"%(n,100*np.std(f,ddof=1)/abs(np.mean(f))))
for n,v in V.items():
    f=[v[s]["final"] for s in ["42","1","7"]]
    print("CV %-24s %.1f%%"%(n,100*np.std(f,ddof=1)/abs(np.mean(f))))
