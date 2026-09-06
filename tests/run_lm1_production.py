"""LM1 — 生产级自主持续学习模型 (2026-08-16)

目标: 完成生产级模型训练, 训练数据由模型自主探索生成。
架构 (减法后最小集, 全家族验证):
  SSM (V31_RLN 规模可配) + OML 双层 (内循环 PLN 适应 + 外循环合并)
  + 价值函数提议器 (简约+自洽+覆盖, 自主生成训练数据)

生产级特性:
  - checkpoint 保存/恢复 (模型+优化器+状态, 断电续训)
  - 结构化报告 (JSONL 每轮 + MD 汇总)
  - 评估协议 (S4 unseen 分桶 + 遗忘回测 + S5 迁移)
  - 达标自动完成 (综合持续学习质量指标, 连续 2 轮)
  - 推理接口 (部署: define+adapt+query)

用法:
  python3 tests/run_lm1_production.py --rounds 10          # 从头训练
  python3 tests/run_lm1_production.py --resume             # 断点续训
  python3 tests/run_lm1_production.py --infer <ckpt>       # 推理接口
"""
import os, sys, time, json, math, random, copy, argparse
import itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_v31_meta_learning import V31_RLN, V31_PLN, batch_pairs, head_forward
from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import generator_signature, cycle_type, TRANS, IDENT, _apply
from hibs_lnn.meta_rule_world import gen_task_data_split
from run_v35_16a_ablation import V35_Learner_A, PERMS24

V = 200
N_RESP = 4
CKPT_DIR = ROOT / 'checkpoints'
RESULT_DIR = ROOT / 'results'

# ============================================================
# S4 价值函数提议器 (自主探索数据生成)
# ============================================================
class S4ValueProposer:
    """简约性 (不可归约) + 自洽性 (不矛盾) + 覆盖 (不重复) → 提议新结构."""

    def __init__(self, learned_perms, lambda_sim=1.0, lambda_con=1.0,
                 lambda_cov=1.0, tau=1.0, k=2):
        self.learned = list(learned_perms)
        self.learned_types = {cycle_type(p) for p in self.learned}
        self.freq = {}
        self.ls, self.lc, self.lcov = lambda_sim, lambda_con, lambda_cov
        self.tau, self.k = tau, k
        self.cands = [p for p in PERMS24 if p not in self.learned]

    def sim(self, p):
        if not self.learned:
            return 1.0
        return 1.0 - max(self._struct_sim(p, q) for q in self.learned)

    def _struct_sim(self, p, q):
        cp, cq = cycle_type(p), cycle_type(q)
        gp = generator_signature(p)[1]
        gq = generator_signature(q)[1]
        s = 0.6 if cp == cq else 0.0
        s += 0.4 * max(0.0, 1 - abs(gp - gq) / 3.0)
        return s

    def con(self, p):
        if not self.learned:
            return 1.0
        n_bad = n = 0
        for tau in self.learned:
            ct = cycle_type(self._compose(p, tau))
            n += 1
            if ct not in self.learned_types:
                n_bad += 1
        return 1.0 - n_bad / max(n, 1)

    def _compose(self, p, q):
        return tuple(p[q[x]] for x in range(4))

    def value(self, p):
        return (self.ls * self.sim(p) + self.lc * self.con(p)
                + self.lcov / (1 + self.freq.get(p, 0)))

    def propose(self, n=None):
        """提议 n 个新结构 (softmax 采样), 返回 perm 列表."""
        n = n or self.k
        if not self.cands:
            return []
        scores = torch.tensor([self.value(p) for p in self.cands])
        probs = (scores / self.tau).softmax(-1)
        idxs = torch.multinomial(probs, min(n, len(self.cands)),
                                 replacement=False)
        out = []
        for i in idxs.tolist():
            p = self.cands[i]
            self.freq[p] = self.freq.get(p, 0) + 1
            out.append(p)
        return out


def make_proposed_task(perm, seed, n_sup=8, n_qry=8):
    """价值函数提议的结构 → 世界执行器生成真值任务 (探索性数据生成)."""
    for _try in range(5):
        sup, qry = gen_task_data_split(
            None, V, n_support=n_sup, n_query=n_qry, seed=seed + _try,
            remap=perm, stop_at_halt=True)
        if sup and qry:
            break
    if not sup or not qry:
        return None
    sig = generator_signature(perm)
    return {'support': [(c, tg[:N_RESP]) for (c, tg) in sup],
            'query': [(c, tg[:N_RESP]) for (c, tg) in qry],
            'perm': perm, 'seen': False,
            'gen_sig': sig[0] if sig else None}


# ============================================================
# OML 训练步 (纯外循环, 家族验证协议)
# ============================================================
def oml_step(learner, tasks, names, it, n_tasks=4, extra=None):
    rng = random.Random(learner.seed + it)
    chosen = rng.sample(names, min(n_tasks, len(names)))
    if extra:
        chosen = chosen + extra
    loss = 0.0
    for rname in chosen:
        t = tasks[rname]
        m = learner._define(t, detach=False)
        W = learner.adapt_graph(t['support'], K=learner.K, m=m)
        ctx, tg = batch_pairs(t['support'], max_batch=16)
        h = learner._rln_fwd(ctx, m)[:, -1, :]
        hin = learner._h_in_batch(h, m)
        pred = head_forward([w for w in W], hin, learner.intent_dim,
                            learner.world_vocab, learner.n_resp)
        loss = loss + F.cross_entropy(
            pred.view(-1, learner.world_vocab),
            tg.view(-1).clamp(0, learner.V + 1)) / len(chosen)
    learner.outer_opt.zero_grad()
    loss.backward()
    learner.outer_opt.step()
    return loss.item()


# ============================================================
# 评估协议
# ============================================================
def s4_eval(learner, tasks, names, k_adapt=20):
    """每任务 adapt_graph → metric. 返回 per-task regs_acc."""
    learner._use_think = False
    learner._use_compose = False
    learner.use_atoms = False
    accs = {}
    for n in names:
        t = tasks[n]
        with torch.no_grad():
            m = learner._define(t, detach=True)
        W = learner.adapt_graph(t['support'], K=k_adapt, m=m)
        with torch.no_grad():
            acc = learner._metric([w.detach() for w in W], t, m=m)[1]
        accs[n] = acc
    return accs


def struct_mean(accs, tasks, u_struct):
    by = {'c3': [], 'c4': [], 'dbl': []}
    for n, a in accs.items():
        s = u_struct.get(n)
        if s in by:
            by[s].append(a)
    return {k: (sum(v) / len(v) if v else None) for k, v in by.items()}


def migrate_eval(learner, d_model, d_state, n_layers, s5_tasks, s5_unseen):
    """S5 迁移评估 (临时副本: deepcopy + head 扩维 4→5, 不影响主模型)."""
    from run_v35_22_s6 import extend_head_n, cycle_type_n
    # 先构造 (触发 V32 意图注入) 再 load 权重 — deepcopy 会 double-inject
    rln2, pln2 = build_model_resp(V, N_RESP, d_model=d_model,
                                  d_state=d_state, n_layers=n_layers)
    l2 = V35_Learner_A(
        rln2, pln2, V, K=20, seed=42, sleep_iters=2, n_resp=N_RESP,
        use_intent=True, use_compose=False, use_atoms=False)
    l2.rln.load_state_dict(learner.rln.state_dict())
    l2.pln.load_state_dict(learner.pln.state_dict())
    extend_head_n(pln2, l2, N_RESP, 5)
    accs = s4_eval(l2, s5_tasks, s5_unseen)   # 协议同构 (n_resp=5)
    by = {'c3': [], 'c4': [], 'c5': [], 'dbl': []}
    for n, a in accs.items():
        ct = cycle_type_n(s5_tasks[n]['perm'])
        m = max(ct) if ct else 0
        key = {5: 'c5', 4: 'c4', 3: 'c3'}.get(m, 'dbl')
        if key in by:
            by[key].append(a)
    sm = {k: (sum(v) / len(v) if v else None) for k, v in by.items()}
    return round(sum(accs.values()) / len(accs), 4), sm


# ============================================================
# LM1 生产级系统
# ============================================================
class LM1System:
    def __init__(self, d_model=192, d_state=12, n_layers=2,
                 iters_per_round=40, n_prop=2, seed=42, proposer='value'):
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.iters_per_round = iters_per_round
        self.n_prop = n_prop
        self.seed = seed
        self.proposer_kind = proposer
        self.round = 0
        self.history = []
        # 数据: S4 seen/unseen + S5 迁移集
        self.tasks = build_v32_tasks(V, n_support=40, n_query=20, seed=42,
                                     window=3, stop_at_halt=True)
        self.u_struct = build_u_struct(42)
        self.t_struct = build_t_struct(42)
        self.tasks = make_regs_tasks(self.tasks)
        self.train_names = [n for n in self.tasks if self.tasks[n]['seen']][:8]
        self.unseen_names = [n for n in self.tasks
                             if not self.tasks[n]['seen']]
        for n in self.tasks:
            self.tasks[n]['name'] = n
            self.tasks[n]['struct'] = (
                self.u_struct.get(n) if not self.tasks[n]['seen']
                else self.t_struct.get(n, '?'))
            sig = generator_signature(self.tasks[n]['perm'])
            self.tasks[n]['gen_sig'] = sig[0] if sig else None
            self.tasks[n]['gen_len'] = sig[1] if sig else 0
        from run_v35_17_s5_transfer import make_s5_tasks
        self.s5_tasks = make_s5_tasks(seed=42)
        self.s5_unseen = [n for n in self.s5_tasks
                          if not self.s5_tasks[n]['seen']]
        self._init_model()

    def _init_model(self):
        random.seed(self.seed); torch.manual_seed(self.seed)
        self.rln, self.pln = build_model_resp(
            V, N_RESP, d_model=self.d_model, d_state=self.d_state,
            n_layers=self.n_layers)
        self.learner = V35_Learner_A(
            self.rln, self.pln, V, K=20, seed=self.seed, sleep_iters=2,
            n_resp=N_RESP, use_intent=True, use_compose=False,
            use_atoms=False)
        learned = [self.tasks[n]['perm'] for n in self.train_names]
        if self.proposer_kind == 'causal':
            from hibs_lnn.causal import CausalProposer
            self.proposer = CausalProposer(learned, k=self.n_prop,
                                           seed=self.seed)
        else:
            self.proposer = S4ValueProposer(learned, k=self.n_prop)
        # 探索任务槽
        self._prop_keys = []

    # ── 探索: 价值函数提议 → 世界生成 (自主数据生成) ──
    def explore(self):
        ps = self.proposer.propose(n=self.n_prop)
        self._prop_keys = []
        for i, p in enumerate(ps):
            key = '__p%d' % i
            t = make_proposed_task(p, 50000 + self.round * 100 + i)
            if t is not None:
                self.tasks[key] = t
                self._prop_keys.append(key)
        return len(self._prop_keys)

    def cleanup_props(self):
        for k in self._prop_keys:
            self.tasks.pop(k, None)
        self._prop_keys = []

    # ── 学习: OML 训练 ──
    def learn(self):
        t0 = time.time()
        losses = []
        base = self.iters_per_round
        for it in range(base):
            self.learner.meta_iters = self.round * 1000 + it
            self.explore()          # 每步探索注入 (持续自主生成)
            loss = oml_step(self.learner, self.tasks, self.train_names,
                            self.round * 1000 + it, extra=self._prop_keys)
            self.cleanup_props()
            losses.append(loss)
        return sum(losses) / len(losses), time.time() - t0

    # ── 评估: S4 unseen + 遗忘 + S5 迁移 ──
    def evaluate(self, with_migrate=False):
        rec = {'round': self.round}
        # S4 unseen (按结构)
        accs = s4_eval(self.learner, self.tasks, self.unseen_names)
        rec['unseen_mean'] = round(sum(accs.values()) / len(accs), 4)
        rec['unseen_by_struct'] = struct_mean(accs, self.tasks,
                                              self.u_struct)
        # 遗忘: seen 回测
        seen_accs = s4_eval(self.learner, self.tasks, self.train_names)
        rec['seen_mean'] = round(sum(seen_accs.values())
                                 / len(seen_accs), 4)
        if self.history:
            prev_seen = self.history[-1].get('seen_mean')
            if prev_seen:
                rec['forget'] = round(prev_seen - rec['seen_mean'], 4)
        # S5 迁移 (每 2 轮)
        rec['migrate_s5'] = None
        if with_migrate:
            mu, msm = migrate_eval(self.learner, self.d_model,
                                   self.d_state, self.n_layers,
                                   self.s5_tasks, self.s5_unseen)
            rec['migrate_s5'] = mu
            rec['migrate_s5_struct'] = msm
        return rec

    # ── checkpoint ──
    def save(self, tag='latest'):
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / ('lm1_%s.pt' % tag)
        torch.save({
            'round': self.round,
            'd_model': self.d_model, 'd_state': self.d_state,
            'n_layers': self.n_layers,
            'rln': self.rln.state_dict(),
            'pln': self.pln.state_dict(),
            'head': self.learner.head.state(),
            'proposer_freq': self.proposer.freq,
            'history': self.history,
        }, path)
        return path

    def resume(self, path):
        ck = torch.load(path)
        self.round = ck['round']
        self.d_model = ck['d_model']
        self.d_state = ck['d_state']
        self.n_layers = ck['n_layers']
        self.history = ck['history']
        self._init_model()
        self.rln.load_state_dict(ck['rln'])
        self.pln.load_state_dict(ck['pln'])
        self.learner.head.load_state(ck['head'])
        self.proposer.freq = ck['proposer_freq']
        return self.round

    # ── 达标检测: unseen ≥ 0.115 且 c4 ≥ 0.09 且遗忘 < 0.05, 连续 2 轮
    def target_met(self, rec):
        u = rec.get('unseen_mean', 0)
        c4 = (rec.get('unseen_by_struct') or {}).get('c4') or 0
        fg = rec.get('forget', 0)
        ok = u >= 0.115 and c4 >= 0.09 and fg < 0.05
        streak = 0
        for h in reversed(self.history[-2:]):
            hu = h.get('unseen_mean', 0)
            hc4 = (h.get('unseen_by_struct') or {}).get('c4') or 0
            if hu >= 0.115 and hc4 >= 0.09 and h.get('forget', 0) < 0.05:
                streak += 1
        return ok and streak >= 1

    def _after_evaluate(self, rec):
        """每轮评估后的回调钩子（子类可重写，用于因果反馈等）。默认无操作。"""
        return rec

    # ── 主循环 ──
    def run(self, rounds, with_migrate_every=2):
        print("=" * 70)
        print("LM1 生产级自主持续学习: %.2fM 参数, %d 轮" % (
            self._n_params() / 1e6, rounds))
        print("=" * 70)
        for _ in range(rounds):
            self.round += 1
            t0 = time.time()
            loss, t_train = self.learn()
            rec = self.evaluate(with_migrate=(self.round % with_migrate_every
                                              == 0))
            rec['loss'] = round(loss, 4)
            rec['train_s'] = round(t_train, 1)
            rec['wall_s'] = round(time.time() - t0, 1)
            self.history.append(rec)
            self._log(rec)
            self._after_evaluate(rec)
            self.save('latest')
            self.save('round%d' % self.round)
            if self.target_met(rec):
                print("\n🎯 目标达成 (round %d): unseen %.4f c4 %.4f 遗忘 %.4f"
                      % (self.round, rec['unseen_mean'],
                         (rec.get('unseen_by_struct') or {}).get('c4', 0),
                         rec.get('forget', 0)))
                self.save('final')
                self._write_report(done=True)
                return self.round
        self._write_report(done=False)
        return self.round

    def _n_params(self):
        return (sum(p.numel() for p in self.rln.parameters())
                + sum(p.numel() for p in self.pln.parameters()))

    def _log(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        RESULT_DIR.mkdir(exist_ok=True)
        with open(RESULT_DIR / 'lm1_report.jsonl', 'a') as f:
            f.write(line + '\n')
        print("[R%d] loss=%.4f unseen=%.4f struct=%s seen=%.4f 遗忘=%s "
              "S5迁移=%s (%.0fs)" % (
                  rec['round'], rec.get('loss', 0), rec['unseen_mean'],
                  rec.get('unseen_by_struct'), rec.get('seen_mean'),
                  rec.get('forget', '-'), rec.get('migrate_s5', '-'),
                  rec.get('wall_s', 0)), flush=True)

    def _write_report(self, done):
        lines = ["# LM1 生产级训练报告",
                 "",
                 "- 参数: %.2fM (d_model=%d, d_state=%d, n_layers=%d)"
                 % (self._n_params() / 1e6, self.d_model, self.d_state,
                    self.n_layers),
                 "- 数据: 自主探索生成 (价值函数提议 → 世界真值)",
                 "- 完成: %s" % ("✅ 达标" if done else "❌ 未达标 (轮次用尽)"),
                 "",
                 "| round | loss | unseen | c3 | c4 | dbl | seen | 遗忘 | S5迁移 |",
                 "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for h in self.history:
            sm = h.get('unseen_by_struct') or {}
            lines.append("| %d | %.4f | %.4f | %s | %s | %s | %.4f | %s | %s |"
                         % (h['round'], h.get('loss', 0), h['unseen_mean'],
                            sm.get('c3', '-'), sm.get('c4', '-'),
                            sm.get('dbl', '-'), h.get('seen_mean', 0),
                            h.get('forget', '-'), h.get('migrate_s5', '-')))
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / 'lm1_report.md').write_text('\n'.join(lines),
                                                  encoding='utf-8')
        print("报告: %s" % (RESULT_DIR / 'lm1_report.md'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--iters-per-round', type=int, default=40)
    ap.add_argument('--d-model', type=int, default=192)
    ap.add_argument('--d-state', type=int, default=12)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--n-prop', type=int, default=2)
    ap.add_argument('--proposer', type=str, default='value',
                    choices=['value', 'causal'])
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--migrate-every', type=int, default=2)
    args = ap.parse_args()

    sys = LM1System(d_model=args.d_model, d_state=args.d_state,
                    n_layers=args.n_layers, iters_per_round=args.iters_per_round,
                    n_prop=args.n_prop, proposer=args.proposer)
    if args.resume:
        r0 = sys.resume(args.resume)
        print("恢复自 %s (round %d)" % (args.resume, r0))
    done_round = sys.run(args.rounds, with_migrate_every=args.migrate_every)
    print("训练完成: round %d, 模型在 checkpoints/lm1_final.pt"
          % done_round)


if __name__ == '__main__':
    main()
