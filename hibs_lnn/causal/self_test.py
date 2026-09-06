"""self_test.py — causal 因果技术栈自测（python3 -m hibs_lnn.causal.self_test）

覆盖：d-separation 经典三元组 · S4WorldSCM 干预/反事实 ·
    识别(后门) · 发现 · 提议器 · Causal RL
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def run():
    ok = 0
    total = 0

    def check(cond, msg):
        nonlocal ok, total
        total += 1
        ok += 1 if cond else 0
        print(("  OK " if cond else "  FAIL ") + msg)

    from hibs_lnn.causal.dag import Graph, d_separated, is_dag

    # 1) 对撞 A -> B <- C
    G = Graph({'B': {'A', 'C'}})
    check(d_separated(G, {'A'}, {'C'}, set()) is True, "collider: A indep C (marginal)")
    check(d_separated(G, {'A'}, {'C'}, {'B'}) is False, "collider: A not indep C | B")

    # 2) 链 A -> B -> C
    G2 = Graph({'B': {'A'}, 'C': {'B'}})
    check(d_separated(G2, {'A'}, {'C'}, set()) is False, "chain: A not indep C")
    check(d_separated(G2, {'A'}, {'C'}, {'B'}) is True, "chain: A indep C | B")
    check(is_dag(G2) is True, "chain is DAG")

    # 3) 叉 A <- B -> C（与链对称）
    G3 = Graph({'A': {'B'}, 'C': {'B'}})
    check(d_separated(G3, {'A'}, {'C'}, {'B'}) is True, "fork: A indep C | B")

    # 4) S4WorldSCM
    from hibs_lnn.causal.scm import S4WorldSCM, _PERMS24
    s = S4WorldSCM()
    p = (1, 2, 3, 0)
    check(s.cyc(p) >= 0 and len(s.sig_bits(p)) == 6, "S4 factor extraction")
    y0 = s.acc(p, set(), noise=0.0)
    y1 = s.do_intervene(p, set())
    check(y1 >= y0 - 1e-9, "do-intervene should not reduce acc (covers atoms)")
    y_cf, meta = s.counterfactual(p, y0, set(), do_perm=p)
    check(meta['n_known_after'] > meta['n_known_before'], "counterfactual: atoms covered")

    # 5) 识别（后门）
    from hibs_lnn.causal.identification import (
        backdoor_adjustment_set, g_formula)
    # 5a) S4 因子图：G 无外生父 → 无后门路径 → 空集可识别
    Gg = Graph({'cyc': {'G'}, 'obs': {'G', 'cyc'}, 'acc': {'obs', 'cyc'}})
    Z0 = backdoor_adjustment_set(Gg, 'G', 'acc', {'G', 'cyc', 'obs'})
    check(Z0 == set(), "backdoor is empty when treatment has no parents")
    ident, _ = g_formula(Gg, 'G', 'acc', set())
    check(ident, "g-formula identifies with empty adjustment set")
    # 5b) 经典混杂 M -> X, M -> Y, X -> Y：do(X) 需调整 M
    Gm = Graph({'X': {'M'}, 'Y': {'X', 'M'}})
    Zm = backdoor_adjustment_set(Gm, 'X', 'Y', {'M', 'X', 'Y'})
    check(Zm == {'M'}, "backdoor adjustment set = {M} for confounded case, got %r" % (Zm,))

    # 6) 发现
    from hibs_lnn.causal.discovery import CausalDiscovery
    learned = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)]
    cd = CausalDiscovery(s, alpha=0.4)
    vars_, adj, X = cd.optimize_adjustment(learned, set())
    check(len(vars_) == 4 and len(X) == len(learned), "discovery data built")

    # 7) 因果提议器
    from hibs_lnn.causal.causal_proposer import CausalProposer
    cp = CausalProposer(learned, k=3, seed=1)
    prop = cp.propose(n=2)
    check(len(prop) == 2 and all(p not in learned for p in prop),
          "causal proposer returns unseen perms")

    # 8) Causal RL
    from hibs_lnn.causal.causal_rl import CausalRLAgent
    agent = CausalRLAgent(learned[:3], k=1, seed=2)
    res = agent.run(2)
    check(res['covered_atoms'] >= 1 and len(res['q_atom']) == 6,
          "causal RL credits atoms: %r" % (res['q_atom'],))

    print("\n  %d/%d passed" % (ok, total))
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if run() else 1)