#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark_rr_online.py — **在线持续学习**消融矩阵(OaK 的 SubTask→Option 层)。

与 `benchmark_b_matrix.py` 的差别只有一处:**多了 reward-respecting 子任务库
这一族臂**;指标、链、T_adapt 定义、判据全部与 B 矩阵**逐字一致**, 这样新臂
可以直接与已钉死的 `primitive 1.82x / rediscover 1.17x / macro 1.08x` 基线对读。

## 为什么不复用 `run_regime_chain`

`run_regime_chain` 把 `SkillAgent` **硬编码**在函数体里。为了不改动它(服务器
上 L3 批正在跑, 中途改 `skill_agent.py` 会让后续臂用上新代码 -> 整批作废),
本文件自带一个**逐字同构**的链运行器, 只多一个 `agent_cls` 参数。

## 消融矩阵(相邻两臂只差**一个**变量)

  臂              相对上一臂改的东西
  ──────────────  ────────────────────────────────────────────
  primitive       (下界基线)
  rr-exact        + reward-respecting 子任务库        ← 机制本体
  rr-zeroV        z 里 V_main → 0                    ← 削**价值函数**
  rr-learned      V_main 由 oracle → agent 自己的 Q  ← 去掉 oracle
  rr-nobonus      w̄ → w_i (bonus ≡ 0)               ← 削**停止奖励**
  rr-nocum        c → 0                              ← 削**沿途主任务奖励**
  rr-nofeat       特征 → 与任务无关的均匀特征        ← 削**特征语义**
  bottleneck      (论文点名的**已知弱**, 负对照)
  random          (随机目标, 负对照)
  macro           (既有的瓶颈状态 option 路径, 参照点)

## 剂量响应 + 内部哨兵(§2 / Pitfall 34)

`opt_prob ∈ {0, 0.25, 0.5, 1.0}`。**0 必须复现 primitive** —— 哨兵不一致
说明诊断自身的实现错了, 其余剂量点全部作废。

## 判据(写死在此, 跑完**先读这里再读数字**)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hibs_lnn.skill_mdp import KeyDoorMDP          # noqa: E402
from hibs_lnn.skill_agent import DEFAULT_REGIMES, SkillAgent  # noqa: E402
from hibs_lnn.rr_agent import RRSkillAgent         # noqa: E402

INTERPRETATION_RULES = [
    "① 内部哨兵: opt_prob=0 的 rr 臂必须与 primitive 臂**逐位相同**。"
    "不一致 -> 诊断自身实现有误, 全部剂量点作废。",
    "② 机制存活: rr 臂的 rr_bookkeeping() 必须显示 selections>0 且 terms 非空。"
    "空台账 = '机制从未运行', 与'机制无效'是不同结论(Pitfall 34) —— 不可写后者。",
    "③ 主判据: T_adapt 逐段 vs primitive, 配 n/mean/std + 配对 t 检验。"
    "|t|<2 或 p>=0.05 -> 写'此 n 下不可分辨', 不写方向。",
    "④ 复用判据: 第2轮 T_adapt < 第1轮 且 p<0.05 -> 复用成立(与 B 矩阵同判据)。",
    "⑤ 价值函数假设: rr-exact vs rr-zeroV 若不可分辨 -> 在该试验台上价值函数"
    "**不是**区分因素; 若 rr-zeroV 更差 -> 价值函数是收益来源。两种都如实写。",
    "⑥ lock-in 检: mean_duration / horizon > 0.5 的臂记为'锁死', 不当抽象读。",
    "⑦ 单 seed 结论不算; 至少 3 seed, 且 std < mean/2 才称可信(项目标准)。",
]


def run_chain(agent_cls, mode, chain, episodes_per, seed, thresh=0.8,
              window=20, n_pos=8, **kw):
    """与 `skill_agent.run_regime_chain` **逐字同构**, 只多 `agent_cls`。"""
    mdp = KeyDoorMDP(n_pos=n_pos, horizon=30, **chain[0])
    agent = agent_cls(mdp, mode=mode, seed=seed, **kw)
    out = []
    for ci, reg in enumerate(chain):
        mdp.set_regime(reg["key_pos"], reg["door_pos"], reg["goal_pos"])
        if mode != "primitive" and not isinstance(agent, RRSkillAgent):
            agent.maybe_discover(force=True)
        if isinstance(agent, RRSkillAgent):
            agent._rr_ensure()          # regime 变了 -> 子任务库重建
        succ, steps_ok, t_adapt = [], [], None
        for ep in range(episodes_per):
            ok, nstep = agent.run_episode()
            succ.append(1.0 if ok else 0.0)
            if ok:
                steps_ok.append(int(nstep))
            if t_adapt is None and len(succ) >= window:
                if float(np.mean(succ[-window:])) >= thresh:
                    t_adapt = ep + 1
        tail = slice(max(0, len(succ) - window), len(succ))
        out.append({"regime": ci, "key": reg["key_pos"], "door": reg["door_pos"],
                    "goal": reg["goal_pos"],
                    "t_adapt": int(t_adapt if t_adapt is not None else episodes_per),
                    "final_succ": float(np.mean(succ[tail])),
                    # ★ 主指标之二: **成功回合的平均步数**。
                    #   t_adapt 与成功率在"任务被 option 平凡解出"时会双双**饱和**
                    #   (rr 贴 window 下限、primitive 贴 episodes 上限), 那时两者
                    #   都看不见效应。steps-to-goal 会动, 是 §8 要求的替代尺子。
                    "mean_steps_success": (float(np.mean(steps_ok)) if steps_ok
                                           else float(np.nan)),
                    "mean_steps_tail": float(np.mean(
                        [s for s, o in zip(steps_ok[-window:], succ[-window:]) if o]))
                        if any(succ[tail]) else float(np.nan),
                    "n_success": len(steps_ok),
                    "n_options": (len(agent.om.options) if getattr(agent, "om", None)
                                  else (len(agent.rr_lib_obj) if
                                        isinstance(agent, RRSkillAgent) and
                                        agent.rr_lib_obj else 0))})
    return out, agent


ARM_DEFS = {
    # name -> (agent_cls, kwargs)
    "primitive":   (SkillAgent, {}),
    "macro":       (SkillAgent, {}),                       # mode 由 --modes 传 macro
    "rr-exact":    (RRSkillAgent, {}),
    "rr-zeroV":    (RRSkillAgent, dict(rr_z_mode="zeroV")),
    "rr-nobonus":  (RRSkillAgent, dict(rr_z_mode="nobonus")),
    "rr-nocum":    (RRSkillAgent, dict(rr_cumulant="zero")),
    "rr-learned":  (RRSkillAgent, dict(rr_v_main="learned")),
    "bottleneck":  (RRSkillAgent, dict(rr_lib="bottleneck")),
    "random":      (RRSkillAgent, dict(rr_lib="random")),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--modes", default="primitive,rr-exact,rr-zeroV,rr-nocum,"
                                       "rr-learned,bottleneck,random")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--opt-prob", type=float, default=1.0)
    ap.add_argument("--n-pos", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "rr_online.json"))
    a = ap.parse_args()

    chain = [dict(r) for r in DEFAULT_REGIMES] * a.cycles
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    print("regime 链: %s" % [f"{r['key_pos']}/{r['door_pos']}/{r['goal_pos']}" for r in chain])
    print("臂: %s | seed x%d | episodes/regime=%d | opt_prob=%.2f"
          % (modes, a.seeds, a.episodes_per, a.opt_prob))
    print()

    all_rows, all_bk, t0 = {}, {}, time.time()
    for mode in modes:
        cls, kw = ARM_DEFS[mode]
        kw = dict(kw)
        mode_val = "macro" if mode == "macro" else ("rr" if cls is RRSkillAgent else mode)
        kw["opt_prob"] = a.opt_prob
        per_seed, bks = [], []
        for si in range(a.seeds):
            rows, agent = run_chain(cls, mode_val, chain, a.episodes_per,
                                    42 + si, n_pos=a.n_pos, **dict(kw))
            per_seed.append(rows)
            bk = (agent.rr_bookkeeping() if isinstance(agent, RRSkillAgent)
                  else (agent.option_bookkeeping() or {}))
            bks.append(bk)
            print("  [%-11s seed=%d] T_adapt=%s 成功率=%s  n_opt=%d  台账=%s"
                  % (mode, 42 + si, [r["t_adapt"] for r in rows],
                     [round(r["final_succ"], 2) for r in rows],
                     rows[-1]["n_options"],
                     {k: bk.get(k) for k in ("selections", "terms")} if bk else "-"),
                  flush=True)
        all_rows[mode], all_bk[mode] = per_seed, bks
        print("    elapsed %.1fs" % (time.time() - t0), flush=True)

    # ── 汇总 ────────────────────────────────────────────────────────
    n_seg = len(chain)
    half = n_seg // a.cycles
    print()
    print("=" * 92)
    print("★ 逐段 T_adapt (T, 上) 与 成功回合平均步数 (S, 下)  —— 跨 seed 均值±std")
    print("=" * 92)
    summary = {}
    for mode in modes:
        arr = np.array([[r["t_adapt"] for r in rows] for rows in all_rows[mode]], float)
        first, second = arr[:, :half].mean(1), arr[:, half:].mean(1)
        d = second - first
        try:
            from scipy import stats
            t, p = stats.ttest_rel(first, second)
        except Exception:
            t, p = float("nan"), float("nan")
        stp = np.array([[r["mean_steps_success"] for r in rows]
                        for rows in all_rows[mode]], float)
        print("  %-11s T=%s" % (mode, " ".join("%.0f±%.0f" % (m, s)
                                               for m, s in zip(arr.mean(0), arr.std(0)))))
        print("  %-11s S=%s" % ("", " ".join("%.1f±%.1f" % (m, s)
                                             for m, s in zip(np.nanmean(stp, 0),
                                                             np.nanstd(stp, 0)))))
        summary[mode] = {
            "t_adapt_mean": [float(x) for x in arr.mean(0)],
            "t_adapt_std": [float(x) for x in arr.std(0)],
            "steps_mean": [float(x) for x in np.nanmean(stp, 0)],
            "steps_std": [float(x) for x in np.nanstd(stp, 0)],
            "reuse_first": float(first.mean()), "reuse_second": float(second.mean()),
            "reuse_delta": float(d.mean()), "reuse_t": float(t) if np.isfinite(t) else None,
            "reuse_p": float(p) if np.isfinite(p) else None,
            "speedup": float(first.mean() / max(second.mean(), 1e-9)),
        }

    print()
    print("★ 复用(第2轮 vs 第1轮, Δ<0 且 p<0.05 = 复用成立)")
    for mode in modes:
        s = summary[mode]
        print("  %-11s %.1f -> %.1f  Δ=%+.1f (%.2fx) p=%s"
              % (mode, s["reuse_first"], s["reuse_second"], s["reuse_delta"],
                 s["speedup"], f"{s['reuse_p']:.4f}" if s["reuse_p"] is not None else "na"))

    if "primitive" in modes:
        print()
        print("★ 相对 primitive 的逐段加速(正 = 该臂更快)")
        prim = np.array([[r["t_adapt"] for r in rows]
                         for rows in all_rows["primitive"]], float).mean(0)
        for mode in modes:
            if mode == "primitive":
                continue
            arr = np.array([[r["t_adapt"] for r in rows]
                            for rows in all_rows[mode]], float).mean(0)
            d = prim - arr
            print("  %-11s %s   |Δ|均=%.1f" % (mode, " ".join("%+6.0f" % x for x in d),
                                              float(np.mean(np.abs(d)))))

    print()
    print("★ 哨兵检(§2): opt_prob=0 必须复现 primitive —— 见 tests/bench_sentinel.sh")
    print("★ 机制存活检(判据②): 各臂台账")
    for mode in modes:
        for i, bk in enumerate(all_bk[mode]):
            if bk:
                print("  %-11s seed=%d  %s" % (mode, 42 + i,
                      {k: bk.get(k) for k in ("selections", "exec_steps", "terms",
                                              "mean_duration", "lockin_ratio",
                                              "beta_nonempty")}))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"modes": modes, "seeds": a.seeds, "episodes_per": a.episodes_per,
         "opt_prob": a.opt_prob, "cycles": a.cycles, "chain": chain,
         "summary": summary, "bookkeeping": all_bk, "rows": all_rows,
         "rules": INTERPRETATION_RULES}, indent=1, ensure_ascii=False))
    print("\n已写 %s  (总耗时 %.1fs)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()