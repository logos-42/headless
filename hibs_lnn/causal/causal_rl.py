"""causal_rl.py — Causal RL：状态/动作/奖励 + 因果 credit assignment（第 12 章）

把"自主数据提议"建模为一个强化学习过程：
  State   : 已覆盖生成元集合 + 各结构区已有表现
  Action  : 提议探索某个 unseen 排列 perm（do(perm)）
  Reward  : 实际泛化表现 acc 提升（因果可分配的增益）
  Credit  : 不用朴素相关，而是用因果归因（do_intervene / counterfactual）
            把 reward 切分到每个贡献原子，解决"相关性≠因果"的 credit 混淆。
"""
import numpy as np
from .scm import S4WorldSCM, _PERMS24


class CausalRLAgent:
    def __init__(self, learned_perms, k=1, seed=0, lr=0.3,
                 world=None, beta=1.0):
        self.world = world or S4WorldSCM(seed=seed)
        self.k = k
        self.seed = seed
        self.lr = lr
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.learned_atoms = set()
        self.learned = list(learned_perms)
        for p in self.learned:
            self.learned_atoms |= {i for i in range(6)
                                   if self.world.sig_bits(p)[i]}
        self.cands = [p for p in _PERMS24 if p not in self.learned]
        # 原子级因果价值 Q（每探索一步，用归因更新）
        self.q_atom = np.zeros(6)
        # 动作级 Q（对每个候选 perm）
        self.q = {p: 0.0 for p in self.cands}

    # ---- 因果归因：把 acc 增益切分到原子 ----
    def _causal_credit(self, perm, delta_acc):
        """把 delta_acc 通过结构方程归因到各生成元原子。

        用 do 前后对比（干预归因）：覆盖 perm 的原子能带来多大增益，
        再按原子"已知/新增"拆解。
        """
        sig = self.world.sig_bits(perm)
        y0 = self.world.acc(perm, self.learned_atoms, noise=0.0)
        y1 = self.world.do_intervene(perm, self.learned_atoms)   # 干预
        gain = max(0.0, y1 - y0)
        # 归因：只有"新原子"才贡献因果（已学原子不增加覆盖）
        new = [i for i in range(6) if sig[i] and i not in self.learned_atoms]
        if not new:
            return {}
        # 按干预后的边际均分（简化但可复现的合法归因）
        per = gain / len(new)
        return {i: per for i in new}

    # ---- 一个交互回合 ----
    def step(self, verbose=False):
        """选一个动作（提议 perm），得到真实 acc，做因果归因更新 Q。"""
        if not self.cands:
            return None
        # 动作选择：exploration(依 Q+新颖性) / 简单 softmax 采样
        scores = np.array([
            self.q[p] + 0.35 * self._novelty(p) for p in self.cands])
        probs = np.exp(scores - scores.max()) / (np.exp(scores - scores.max()).sum()
                                                 + 1e-12)
        i = self.rng.choice(len(self.cands), p=probs)
        perm = self.cands[i]
        self.cands.pop(i)

        # 真实 reward：实际泛化（探索该 perm 之后，其生成元被覆盖）
        before = self._mean_acc()
        new_atoms = self.learned_atoms | {j for j in range(6)
                                          if self.world.sig_bits(perm)[j]}
        after = np.mean([self.world.acc(q, new_atoms, noise=0.0)
                         for q in self.cands + [perm]])
        r = max(0.0, after - before)

        # 因果 credit assignment（非线性相关）
        credit = self._causal_credit(perm, r)
        for atom, c in credit.items():
            self.q_atom[atom] += self.lr * (c - self.q_atom[atom])
        self.q[perm] = self.lr * (r - self.q[perm])

        # 提交：把新原子并入已学
        self.learned_atoms |= {j for j in range(6)
                               if self.world.sig_bits(perm)[j]}
        self.learned.append(perm)
        self.beta = self.beta
        if verbose:
            print("  [causalRL] act=%s r=%.3f credit_atoms=%s"
                  % (perm, round(r, 3),
                     {a: round(v, 3) for a, v in credit.items()}))
        return {'perm': perm, 'reward': r, 'credit': credit}

    def _novelty(self, p):
        new = {i for i in range(6) if self.world.sig_bits(p)[i]
               and i not in self.learned_atoms}
        return len(new) / 6.0

    def _mean_acc(self):
        pool = self.cands + self.learned
        if not pool:
            return 0.0
        return float(np.mean([self.world.acc(p, self.learned_atoms, noise=0.0)
                              for p in pool]))

    def run(self, n_steps, verbose=False):
        for _ in range(n_steps):
            self.step(verbose=verbose)
        return {'q_atom': self.q_atom.round(3).tolist(),
                'covered_atoms': int(len(self.learned_atoms)),
                'n_perms': len(self.learned)}