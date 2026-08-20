"""V34.7 ComposeModule — 组合本身成为可学习模块 (打破 latent 凸包)

立论 (v34-architecture §2.5): V34.2-6 五连证明所有"latent 凸组合" (Jaccard/归一/
词序/迭代) 都落在原子原型凸包内, c4 (3 生成元复合) 无法表达. 唯一出路 = 组合
函数本身可学习, 由 BPTT 驱动, 而非零参数插值.

机制 (V34.7):
  原子: AtomRouteBank (V34.3, per-generator 原型, 训练滚动平均/部署检索).
  组合网络 ComposeModule (新增参数, 小):
    in  = [m_define (d,) ; comp_atoms (d,) ; gen_sig (6,)]
    h1  = SiLU(W1·in + b1);   out = W2·h1 + b2
    m'  = m_norm(m_define + out)
  监督: m' 参与 head 预测 query loss, 外循环 BPTT 更新 compose W1/W2
        → 组合函数学"如何把原子拼成未见规则" (组合后预测更准为驱动).

Cell:
  B : V33 base
  C : AtomRouteBank 纯原子 α=1 (V34.3 Cr 复现, 零参数对照)
  M : + ComposeModule (可学习组合)

指标: 结构分组 regs_acc, n_params (能量口径: M 增量), 时长.
"""
import os, sys, time, json, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_v31_meta_learning import batch_pairs, head_forward, V31_RLN, V31_PLN
from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    V32_1_Learner, build_model_resp, make_regs_tasks, build_u_struct,
    build_t_struct, tome_fraction)
from run_v33_drl import V33_Learner, V33_3_Learner
from run_v34_1_gen import generator_signature, cycle_type
from run_v34_3_cpg import AtomRouteBank, N_GEN

_gen_cache = {}


class ComposeModule(nn.Module):
    """可学习组合网络: [m_define; comp_atoms; sig] → 修正量加到 m_define.

    comp_atoms = 原子原型按 sig 的重组 (零参数提供凸包内基点),
    compose 学凸包外的修正 → 组合泛化。
    """

    def __init__(self, d, n_sig=6, hidden=None):
        super().__init__()
        h = hidden or d
        self.fc1 = nn.Linear(2 * d + n_sig, h)
        self.fc2 = nn.Linear(h, d)
        self.d = d
        self.n_sig = n_sig

    def forward(self, m_def, comp, sig):
        x = torch.cat([m_def, comp, sig], dim=-1)
        h1 = F.silu(self.fc1(x))
        return self.fc2(h1)


class V34_7_Learner(V33_Learner):
    """V33 + AtomRouteBank (零参数原子) + ComposeModule (可学习组合)."""

    def __init__(self, rln, pln, V, use_compose=False, alpha=1.0, use_atoms=True, **kw):
        lam_icl = kw.pop('lambda_icl', 0.0)
        self._use_compose = use_compose
        self._alpha = alpha
        self.use_atoms = use_atoms
        super().__init__(rln, pln, V, **kw)
        self.lambda_icl = lam_icl
        d = rln.d_model
        self.bank = AtomRouteBank(d, alpha=alpha)
        if use_compose:
            self.compose = ComposeModule(d)
            # 重建 outer_opt 纳入 compose 参数 (与 V33 同 lr)
            base = (list(rln.parameters()) + list(pln.parameters())
                    + list(self.intent_net.parameters())
                    + list(self.refine_net.parameters())
                    + list(self.probe_head.parameters()))
            self.outer_opt = torch.optim.Adam(
                base + list(self.compose.parameters()), lr=self.outer_lr)

    def _compose_inputs(self, task, m, sig):
        """返回 (comp_atoms, sig_vec). comp = bank 原子按 sig sum (detached)."""
        hit = [i for i in range(N_GEN) if sig[i] and self.bank.atoms[i] is not None]
        if hit:
            comp = torch.zeros_like(m)
            for i in hit:
                comp = comp + self.bank.atoms[i]
            comp = comp / (comp.norm() + 1e-8)
        else:
            comp = torch.zeros_like(m)
        sigv = torch.tensor(sig, dtype=torch.float)
        return comp, sigv

    def _post_define_aux_loss(self, m, t):
        """ICL 偏置: 用未适配共享头 W0 直接读组合后的 m 在 query 上预测.
        逼 m' 在纯 ICL (0 per-task 适配) 下就暴露组合规则 → 组合必须承载原子.
        m 为 attached (训练走 _define detach=False), 梯度反传进 compose."""
        if not self._use_compose or m is None or self.lambda_icl <= 0:
            return None
        with torch.no_grad():
            W0 = [w.detach() for w in self.pln.clone_params()]
        ctx, tgt = batch_pairs(t['query'], max_batch=16)
        hidden = self._rln_fwd(ctx, m)
        h = hidden[:, -1, :]
        h_in = self._h_in_batch(h, m)
        pred = head_forward(W0, h_in, self.intent_dim,
                            self.world_vocab, self.n_resp)
        pred = self._route_logits(pred, m)
        return F.cross_entropy(pred.view(-1, self.world_vocab),
                               tgt.view(-1).clamp(0, self.V + 1))

    def _define(self, task, detach=True, use_bank=True):
        m = super()._define(task, detach=False, use_bank=False)
        sig = task.get('gen_sig') or [0] * N_GEN
        use_bank = self.use_atoms
        if self._use_compose:
            if use_bank:
                if detach:
                    self.bank.route(m, sig)
                else:
                    self.bank.add(m, sig)
            comp, sigv = self._compose_inputs(task, m, sig)
            corr = self.compose(m, comp, sigv.to(m.device))
            m = self.m_norm(m + corr)
            if detach:
                m = m.detach()
            return m
        # 非 compose: 纯 base (不用 bank) 或纯原子 (use_atoms)
        if use_bank:
            if detach:
                m, _ = self.bank.route(m, sig)
                m = m.detach()
            else:
                self.bank.add(m, sig)
        elif detach:
            m = m.detach()
        return m


# ============================================================
# 驱动
# ============================================================
def run_cell(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
             seed=42, steps_per_task=60, verbose=True):
    lm = cfg['lm']
    tasks = make_regs_tasks(tasks)
    train_names = [n for n in tasks if tasks[n]['seen']][:8]
    unseen_names = [n for n in tasks if not tasks[n]['seen']]   # 全 12
    stream = train_names + unseen_names
    T_STRUCT = build_t_struct(42)
    for n in tasks:
        tasks[n]['name'] = n
        tasks[n]['struct'] = (U_STRUCT.get(n) if not tasks[n]['seen']
                              else T_STRUCT.get(n, '?'))
        sig = generator_signature(tasks[n]['perm'])
        tasks[n]['gen_sig'] = sig[0] if sig else None
        tasks[n]['gen_len'] = sig[1] if sig else 0
    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(200, 4, s1_state)
    use_compose = (lm == 'compose')
    learner = V34_7_Learner(rln, pln, 200, K=K, seed=seed, sleep_iters=2,
                            n_resp=4, use_intent=True,
                            use_compose=use_compose,
                            use_atoms=(lm != 'base'),
                            lambda_icl=cfg.get('lambda_icl', 0.0),
                            alpha=cfg.get('alpha', 1.0))
    t0 = time.time()
    n_iters = cfg.get('meta_iters', meta_iters)
    for it in range(n_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_names)
        if verbose and (it + 1) % 20 == 0:
            print("    [%s] it %d/%d loss=%.4f" % (key, it + 1, n_iters, loss), flush=True)
    print("    [%s] 训练 %.0fs" % (key, time.time() - t0), flush=True)
    trained_head = learner.head.state()
    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    print("    [%s] 长流 %.0fs" % (key, time.time() - t0), flush=True)
    by = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen_names:
        s = U_STRUCT[n]
        if s in by:
            by[s].append(st['acc_after'][n])
    struct_mean = {k: (sum(v) / len(v) if v else None) for k, v in by.items()}
    aux = {}
    if getattr(learner, 'bank', None) is not None:
        aux['atom_bank'] = learner.bank.stats()
    if getattr(learner, 'compose', None) is not None:
        cp = learner.compose
        aux['compose_params'] = sum(p.numel() for p in cp.parameters())
    n_params = sum(p.numel() for p in rln.parameters()) + sum(p.numel() for p in pln.parameters())
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'world': cfg.get('world', 'new'),
           'factors': {'LM': lm, 'alpha': cfg.get('alpha', 1.0),
                       'use_compose': use_compose,
                       'lambda_icl': cfg.get('lambda_icl', 0.0)},
           'time_s': round(time.time() - t0, 1),
           'n_params': n_params,
           'params_kb': round(n_params * 4 / 1024, 1),
           'seen_acc': {n: st['acc_after'][n] for n in train_names},
           'unseen_acc': {n: st['acc_after'][n] for n in unseen_names},
           'unseen_by_struct': struct_mean,
           'aux': aux}
    u = rec['unseen_acc']; ua = sum(u.values()) / len(u)
    print(("  [%s] unseen=%.4f struct=%s n_params=%d aux=%s" %
           (key, ua, struct_mean, n_params, aux)), flush=True)
    return rec


CELLS = {
    'B': dict(world='new', lm='base', label='B-base'),
    'C': dict(world='new', lm='atom', alpha=1.0, label='C-atom-pure'),
    'M': dict(world='new', lm='compose', alpha=1.0, label='M-compose-learn'),
'MI': dict(world='new', lm='compose', alpha=1.0,
               lambda_icl=0.5, label='M-compose-ICL-bias'),
    'MI3': dict(world='new', lm='compose', alpha=1.0,
                lambda_icl=1.0, label='M-compose-ICL-bias-l1.0'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='B,C,M')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70); print("V34.7 ComposeModule — 可学习组合模块 (破凸包)")
    print("=" * 70)
    tasks = build_v32_tasks(200, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
    s1 = None
    if not args.no_stage1:
        for cand in ('/tmp/v32_s1_state.pt', '/tmp/v31_2_s1_state.pt'):
            if os.path.exists(cand):
                s1 = torch.load(cand)
                s1 = {k: v.cpu().clone() for k, v in s1.items()}
                print("Stage1: %s" % cand); break
        else:
            print(">>> Stage1 缺失, 请先跑 run_v32_stage1.py"); return
    res = []
    for key in [c.strip() for c in args.cells.split(',')]:
        cfg = CELLS[key]
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell(key, cfg, s1, tasks, U_STRUCT, meta_iters=args.meta_iters)
        except Exception:
            import traceback; traceback.print_exc()
            print("!!! %s 失败" % key); continue
        res.append(r)
    out = ROOT / 'results' / 'v34_7_report.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s params=%d %s" %
              (r['cell'], ua, r['unseen_by_struct'], r['n_params'], r['aux']))


if __name__ == '__main__':
    main()