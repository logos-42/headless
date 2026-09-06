#!/usr/bin/env python3
"""headless · Causal AI 技术栈端到端演示

把《Causal AI》的因果技术落地到 headless 的 S4 元学习世界，并展示
"用因果发现改进自主数据提议器"的全链路：

  STEP 1  把 S4 世界建模为 SCM（生成元 -> 循环结构 -> 泛化表现）
  STEP 2  d-separation：识别因果路径的独立性
  STEP 3  do() 干预：覆盖生成元的因果效应
  STEP 4  因果发现：从观测 acc 反估计"生成元 -> 泛化"的依赖
  STEP 5  识别：后门调整集
  STEP 6  反事实：Pearl 三步（Abduction / Action / Prediction）
  STEP 7  因果提议器 vs 启发式提议器（自主探索效率对比）★
  STEP 8  Causal RL：状态/动作/奖励 + 因果 credit assignment

运行：python3 causal_tech_stack_demo.py    （纯 numpy，秒级）
"""
import os
import sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from hibs_lnn.causal import (                  # noqa: E402
    S4WorldSCM, Graph, d_separated,
    backdoor_adjustment_set, g_formula,
    CausalDiscovery, CausalProposer, CausalRLAgent,
)


def show(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    s = S4WorldSCM(seed=0)

    # ---- STEP 1: SCM ----
    show("STEP 1  把 S4 元学习世界建模为结构因果模型 (SCM)")
    print("  结构方程 (PARENTS):")
    for v, pa in S4WorldSCM.PARENTS.items():
        print("    %-5s  <-  %s" % (v, ", ".join(pa)))
    print("  变量：G 生成元集合(因) / cyc 循环类型 / obs 观测特征 / acc 泛化(果)")
    p4 = (1, 2, 3, 0)
    print("\n  例 perm=(1,2,3,0): cyc=%d, 生成元签名=%s"
          % (s.cyc(p4), s.sig_bits(p4)))
    print("  acc(未覆盖生成元)=%.3f, do(覆盖其生成元)=%.3f"
          % (s.acc(p4, set(), noise=0.0), s.do_intervene(p4, set())))

    # ---- STEP 2: d-separation ----
    show("STEP 2  d-separation —— 因果图中路径的独立性")
    Gc = Graph({'cyc': {'G'}, 'obs': {'G', 'cyc'}, 'acc': {'obs', 'cyc'}})
    print("  图: G->cyc->acc 链 + G->obs、cyc->obs、obs->acc（obs 是另一中间变量）")
    print("    仅条件 {cyc} 不够：还存在 G->obs->acc 链，需同时阻断 obs")
    print("    G 与 acc 无条件相连接   ->",
          d_separated(Gc, {'G'}, {'acc'}, set()))
    print("    G ┴ acc | {cyc, obs}    ->",
          d_separated(Gc, {'G'}, {'acc'}, {'cyc', 'obs'}))
    gc = Graph({'cyc': {'G'}, 'acc': {'cyc'}})
    print("    最短链 G->cyc->acc |{cyc}->",
          d_separated(gc, {'G'}, {'acc'}, {'cyc'}))

    # ---- STEP 3: do() 干预 ----
    show("STEP 3  do() 干预 —— 覆盖生成元对泛化的因果效应")
    learned0 = [(0, 1, 2, 3), (1, 0, 2, 3)]
    atoms0 = set()
    for p in learned0:
        atoms0 |= {i for i in range(6) if s.sig_bits(p)[i]}
    print("  已学 perm 覆盖生成元原子: %s" % sorted(atoms0))
    eff = {}
    for p in s.factor:
        y_no = s.acc(p, set(), noise=0.0)
        y_do = s.do_intervene(p, atoms0)
        eff[p] = y_do - y_no
    cand = sorted(eff, key=eff.get, reverse=True)[:3]
    print("  干预(do)效应最大的 3 个候选 perm:")
    for p in cand:
        print("    %s  +%.3f" % (p, round(eff[p], 3)))
# ---- STEP 4: 因果发现 ----
    show("STEP 4  因果发现 —— 从观测因子反估计 (生成元->泛化)")
    learned = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2),
               (0, 2, 1, 3), (0, 3, 2, 1)]
    disc = CausalDiscovery(s, alpha=0.5)
    # 因果发现：在"历史观测"上反推哪些因子决定泛化
    # 为让离散统计稳定，对每个 perm 累积 8 次带噪观测
    import numpy as np
    all_perms = list(s.factor.keys())
    rows = []
    k = 0
    for p in all_perms:
        for _ in range(8):
            f = s.observe(p, rng=np.random.default_rng(1000 + k))
            rows.append([f[0], f[1], f[2], s.acc(p, set(), seed=2000 + k)])
            k += 1
    Xs = np.array(rows)
    from hibs_lnn.causal.discovery import ci_test
    p_cyc, mi_cyc = ci_test(Xs, 2, 3, [])            # cyc 与 acc
    p_gen, mi_gen = ci_test(Xs, 0, 3, [])            # gen_count 与 acc
    print("  历史观测样本数:", len(Xs))
    print("  条件独立检验 (p 越小越依赖):")
    print("    cyc(结构)     -> acc : p=%.3f  mi=%.3f  -> %s"
          % (p_cyc, mi_cyc, "相关(depend)" if p_cyc < 0.3 else "独立"))
    print("    gen_count     -> acc : p=%.3f  mi=%.3f  -> %s"
          % (p_gen, mi_gen, "相关(depend)" if p_gen < 0.3 else "独立"))
    print('  解读：未覆盖任何生成元时，泛化仅由循环结构(cyc)决定；')
    print('  这就是数据对【先学结构、再补生成元】的因果提示。')

    # ---- STEP 5: 识别 ----
    show("STEP 5  识别 —— 后门调整")
    Gg = Graph({'cyc': {'G'}, 'obs': {'G', 'cyc'}, 'acc': {'obs', 'cyc'}})
    Z = backdoor_adjustment_set(Gg, 'G', 'acc', {'G', 'cyc', 'obs'})
    fident, msg = g_formula(Gg, 'G', 'acc', Z)
    print("  后门调整集 Z = %r" % (Z,))
    print("  可识别性: %s" % msg)

    # ---- STEP 6: 反事实 ----
    show("STEP 6  反事实 —— Pearl 三步 (Abduction/Action/Prediction)")
    obs_p = (1, 2, 3, 0)
    y_obs = s.acc(obs_p, atoms0, noise=0.05)
    y_cf, meta = s.counterfactual(obs_p, y_obs, atoms0, do_perm=(2, 0, 3, 1))
    print("  观测: perm=%s acc=%.3f" % (obs_p, y_obs))
    print("  反事实(若早期探索另一 perm、覆盖其生成元): acc=%.3f" % y_cf)
    print("  归因: 推断外生扰动 u=%s, 生成元覆盖 %d -> %d"
          % (meta['u_acc'], meta['n_known_before'], meta['n_known_after']))

    # ---- STEP 7: ★ 因果提议器 vs 启发式提议器 ----
    show("STEP 7  ★ 因果提议器 vs 启发式提议器（自主探索效率对比）")
    scores = run_proposer_compare(s)
    print("  在同样的已学集合上各提议 6 个 unseen 排列：")
    print("    %-22s  %-16s %-14s"
          % ("提议器", "新生成元原子/步", "覆盖未见结构占比"))
    for name, new_atoms, frac in scores:
        print("    %-22s  %-16s %-14s"
              % (name, "%.2f" % new_atoms, "%.2f" % frac))
    print("\n  因果提议器优先选择【被生成元瓶颈卡住】的任务（do-效应/反事实增益高），")
    print("  用更少探索步数覆盖更多【原因原子】——自主数据生成器的因果改进。")

    # ---- STEP 8: Causal RL ----
    show("STEP 8  Causal RL —— 状态/动作/奖励 + 因果 credit assignment")
    agent = CausalRLAgent(learned[:3], k=1, seed=3)
    res = agent.run(4, verbose=True)
    print("  归因到的原子级因果价值 Q:", res['q_atom'])
    print("  已覆盖生成元原子数: %d / 6" % res['covered_atoms'])

    print("\n  完成：因果技术栈已在 headless 的 S4 世界完整落地。")
    print("  详见 docs/wiki/causal_stack.md")


def baseline_value(world, learned, p, freq):
    """启发式提议器价值（简约/自洽/覆盖的轻量复刻）。"""
    sim = 1.0
    for q in learned:
        if world.cyc(p) == world.cyc(q):
            sim -= 0.3
    learned_cyc = {world.cyc(q) for q in learned}
    con = 0.5 if world.cyc(p) in learned_cyc else 1.0
    cov = 1.0 / (1 + freq.get(p, 0))
    return sim + con + cov


def run_proposer_compare(world, n_steps=6, n_learned=6):
    import numpy as np
    perms = list(world.factor.keys())
    learned = perms[:n_learned]                    # 同一初始已学集合
    cands = [p for p in perms if p not in learned]

    # --- 启发式 ---
    freq_b = {}
    picked_b = []
    for _ in range(n_steps):
        sc = [baseline_value(world, learned, p, freq_b) for p in cands]
        for _ in range(4):
            bi = int(np.argmax(sc))
            p = cands[bi]
            if p in picked_b:
                sc[bi] = -1e9
                continue
            break
        picked_b.append(p)
        freq_b[p] = freq_b.get(p, 0) + 1

    # --- 因果 ---
    cp = CausalProposer(learned, k=n_steps, seed=7)
    picked_c = cp.propose(n=n_steps)

    def _new_atoms(picked):
        at = set()
        for p in learned:
            at |= {i for i in range(6) if world.sig_bits(p)[i]}
        tot = 0
        for p in picked:
            n0 = len(at)
            at |= {i for i in range(6) if world.sig_bits(p)[i]}
            tot += len(at) - n0
        return tot / max(1, len(picked))

    seen_cyc = {world.cyc(p) for p in learned}
    frac_b = sum(1 for p in picked_b if world.cyc(p) not in seen_cyc) / n_steps
    frac_c = sum(1 for p in picked_c if world.cyc(p) not in seen_cyc) / n_steps
    return [
        ("启发式 (sim/con/cov)", _new_atoms(picked_b), frac_b),
        ("因果 (do/反事实/发现)", _new_atoms(picked_c), frac_c),
    ]


if __name__ == "__main__":
    main()