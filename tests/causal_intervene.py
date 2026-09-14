"""LM4 因果层 · 干预与反事实 (对电磁波结构做 do() 与反事实查询)。

配套 lm4_causal_waves.py 的发现结果使用。

思路:
  1. 用**物理先验**给稳定边定向(发现只给出无向骨架):
       L → B          (偶极场 B ∝ L^-3)
       L → log_dens   (等离子体层顶依赖 L)
       log_dens → ele_*   (hiss 波由等离子体层电子产生)
       log_dens → mag_mid/hi (密度影响磁强计频段波活动)
       logB → mag_lo/mid/hi (磁强计频段本就源自 B)
       absmaglat → ele_hi    (高纬沉降影响高频电场)
  2. 每个节点对父节点做**线性回归**拟合机制 -> 得到线性 SCM
  3. **干预** do(X=x): 沿拓扑序前向传播, 下游节点按机制重算
  4. **反事实** (Pearl 三步):
       Abduction  由观测值反推该样本的外生噪声 U = 观测 - f(父节点观测)
       Action     施加 do()
       Prediction 用同一 U 沿拓扑序重算 -> 该样本"若当初……会怎样"

用法:
  python3 tests/causal_intervene.py --sample 20000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from lm4_causal_waves import build_variables
from hibs_lnn.causal.dag import Graph

# ── 物理先验定向: node -> 父节点列表 (按拓扑序给出) ──
PHYSICS_PARENTS = {
    "logL": [],
    "absmaglat": [],
    "logB": ["logL"],                       # 偶极场 B ∝ L^-3
    "log_dens": ["logL", "absmaglat"],      # 等离子体层顶位置依赖 L 与磁纬
    "mag_lo": ["logB"],
    "mag_mid": ["logB", "log_dens"],
    "mag_hi": ["logB", "log_dens"],
    "ele_lo": ["log_dens", "logL"],
    "ele_mid": ["log_dens", "logL"],
    "ele_hi": ["log_dens", "absmaglat"],
}
# 拓扑序 (父在前) —— 上面字典的键序已满足
TOPO = ["logL", "absmaglat", "logB", "log_dens",
        "mag_lo", "mag_mid", "mag_hi", "ele_lo", "ele_mid", "ele_hi"]


class LinearSCM:
    """线性结构因果模型 (加性噪声), 支持 do() 与反事实。"""

    def __init__(self, parents, topo, names):
        self.parents = parents
        self.topo = [n for n in topo if n in names]
        self.names = names
        self.coef = {}          # node -> (w, b)  w 对应 parents 顺序

    def fit(self, X, names):
        idx = {n: i for i, n in enumerate(names)}
        for node in self.topo:
            pa = self.parents.get(node, [])
            y = X[:, idx[node]]
            if not pa:
                self.coef[node] = (np.zeros(0), float(y.mean()))
                continue
            A = np.stack([X[:, idx[p]] for p in pa], axis=1)
            A1 = np.concatenate([A, np.ones((len(A), 1))], axis=1)
            w, *_ = np.linalg.lstsq(A1, y, rcond=None)
            self.coef[node] = (w[:-1], float(w[-1]))
        return self

    def _f(self, node, values, ref):
        """按机制计算 node 的确定性部分。values: dict node->array;
        ref: 任一数组, 仅用于取形状 (根节点的父母为空, 不能从 values 取)。"""
        pa = self.parents.get(node, [])
        w, b = self.coef[node]
        out = np.full_like(ref, b)
        for k, p in enumerate(pa):
            out = out + w[k] * values[p]
        return out

    def counterfactual(self, X, names, interventions):
        """返回反事实下的所有变量值 (dict)。

        interventions: {node: array} 或 {node: 标量}
        用观测残差做 abduction —— 这就是 Pearl 三步里的"同一个人、同一个 U"。
        """
        idx = {n: i for i, n in enumerate(names)}
        obs = {n: X[:, idx[n]] for n in self.topo}
        # Abduction: 外生噪声 U = 观测 - f(父节点观测)
        ref = obs[self.topo[0]]
        U = {n: obs[n] - self._f(n, obs, ref) for n in self.topo}
        # Action + Prediction: 沿拓扑序重算
        cf = {}
        for n in self.topo:
            if n in interventions:
                v = interventions[n]
                cf[n] = np.full_like(obs[n], v) if np.isscalar(v) else np.asarray(v)
            else:
                cf[n] = self._f(n, cf, ref) + U[n]
        return cf, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "wave"))
    ap.add_argument("--bands", type=int, default=13)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "lm4_causal"))
    args = ap.parse_args()

    X, names = build_variables(args.data, bands=args.bands,
                               sample=args.sample, seed=args.seed)
    scm = LinearSCM(PHYSICS_PARENTS, TOPO, names).fit(X, names)

    print("=== 拟合的线性机制 (w·父节点 + b) ===")
    for n in scm.topo:
        pa = PHYSICS_PARENTS.get(n, [])
        w, b = scm.coef[n]
        terms = " + ".join(f"{wi:+.3f}·{p}" for wi, p in zip(w, pa)) if pa else "(常数)"
        print(f"  {n:10s} = {terms}  {b:+.3f}")

    idx = {n: i for i, n in enumerate(names)}
    report = {"mechanisms": {n: {"parents": PHYSICS_PARENTS.get(n, []),
                                 "w": [float(x) for x in scm.coef[n][0]],
                                 "b": float(scm.coef[n][1])} for n in scm.topo},
              "interventions": {}, "counterfactual": {}}

    # ── 干预实验 ──
    print("\n=== do() 干预效应 (改变某变量 -> 下游各变量的平均变化) ===")
    for target, factor in [("log_dens", 0.1), ("log_dens", 10.0),
                           ("ele_hi", 2.0), ("ele_hi", 0.5),
                           ("logL", None)]:
        cur = X[:, idx[target]]
        if factor is None:
            newv = cur + 1.0                      # logL +1 (向外移动 1 个 L)
            label = f"do({target} = +1)"
        else:
            newv = cur + np.log10(factor)         # log 空间挪 log10(factor)
            label = f"do({target} = ×{factor})"
        cf, obs = scm.counterfactual(X, names, {target: newv})
        changes = {n: float(np.mean(cf[n] - obs[n])) for n in scm.topo if n != target}
        big = sorted(changes.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"\n  {label}:")
        for n, d in big:
            print(f"     {n:10s} Δ = {d:+.4f}")
        report["interventions"][label] = dict(changes)

    # ── 反事实查询 (单样本) ──
    print("\n=== 反事实查询 (逐个样本: 若密度低 10 倍, 高频电场会是多少?) ===")
    i = int(np.argmax(X[:, idx["ele_hi"]]))       # 取电场高频最强的那个样本
    newv = X[:, idx["log_dens"]] - 1.0            # log10 空间 -1 = 密度 ÷10
    cf, obs = scm.counterfactual(X, names, {"log_dens": newv})
    print(f"  样本 #{i} (ele_hi 最强):")
    print(f"     观测: log_dens={obs['log_dens'][i]:.3f}  ele_hi={obs['ele_hi'][i]:.3f}")
    print(f"     反事实(密度÷10): ele_hi = {cf['ele_hi'][i]:.3f} "
          f"(Δ = {cf['ele_hi'][i]-obs['ele_hi'][i]:+.3f})")
    report["counterfactual"] = {
        "sample": i,
        "obs": {n: float(obs[n][i]) for n in scm.topo},
        "cf_dens_div10": {n: float(cf[n][i]) for n in scm.topo},
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "causal_intervene.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n报告: {args.out}/causal_intervene.json")


if __name__ == "__main__":
    main()
