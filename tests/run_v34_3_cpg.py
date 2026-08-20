"""V34.3 CPG 原子路由 — 零参数原子级显式重组 (对标 CPG modularity)

立论 (v34-2-anchors): V34.2 的 ReuseBank 检索单元是"整规则 latent m"的线性加权,
组合发生在整体向量空间插值 → R=R3 (topk 无差别), 未见 c4 (需 3 生成元) 未被真正
拆解组合。CPG (Klinger arXiv:2309.16467) 的关键是 modularity: 每个 grammar rule
一个独立语义模块 + 递归组合。这里把检索单元缩到**单个生成元原子** prototype.

机制 (V34.3): AtomRouteBank — 不新增任何可学习参数 (per-generator prototype 只是
存表示, 复用 = 编码时线性操作).
  积累: 训练时把 seen 规则的 define m 按其生成元 signature (6 维 binary) 滚动
      平均聚合到 **每个生成元格位** 的原子原型 (per-generator-atom prototype).
      → 每个生成元一个"语义模块" (CPG grammar-rule module).
  路由/组合: 对查询 (train/unseen) 按 signature 显式选择命中原子, 把它们
      sum/normalize 重组为组合路由 m' (CPG computation-tree 组合的内存版).
  alpha: 混合 define 原 m 与原子路由 (alpha=1 纯重组).

特点 (vs V34.2): ① 检索单元 = 单生成元原子 (粒度细) 而非整规则; ② 显式选择
签名命中的原子进行重组 (路由语义), 而非对整 m 做 Jaccard 加权.

Cell:
  B  : V33 base d128 基线 (无 reuse) = 对照 V34.2 B
  C  : + CPG 原子路由 (alpha 混合)
  Cr : + CPG 原子路由 (alpha=1 纯重组)

指标: 结构分组 regs_acc, 原子命中统计, 参数量/时长 (能量口径).
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

N_GEN = 6


class AtomRouteBank:
    """CPG 原子路由: per-generator-atom prototype (零参数). 原子 = 6 个生成元格位.

    积累: seen 规则 m 按签名向每个命中格位滚动平均. → 每个生成元一个语义模块.
    路由: 查询按签名显式选择命中原子, sum/normalize 重组 → 组合 m'.
    """

    def __init__(self, d, alpha=0.5, norm_mode='l2'):
        self.d = d
        self.alpha = alpha              # 0..1, 1 = 纯原子重组
        self.norm_mode = norm_mode
        self.atoms = [None] * N_GEN     # 每格位原型向量 (rolling mean)
        self._count = [0] * N_GEN
        self.n_add = 0
        self.n_route = 0
        self.n_hit = 0

    def _norm(self, m):
        if self.norm_mode == 'l2':
            return m / (m.norm() + 1e-8)
        return m

    def add(self, m, sig):
        """训练: 把 m 向每个命中生成元格位滚动平均积累."""
        v = m.detach().float()
        for i in range(N_GEN):
            if sig[i]:
                c = self._count[i]
                if self.atoms[i] is None:
                    self.atoms[i] = self._norm(v)
                else:
                    self.atoms[i] = self._norm(
                        (self.atoms[i] * c + v) / (c + 1))
                self._count[i] = c + 1
        self.n_add += 1

    def route(self, m, sig):
        """部署: 按签名显式选择命中原子, 重组为组合 m'. 返回 (m', used)."""
        hit = [i for i in range(N_GEN) if sig[i] and self.atoms[i] is not None]
        if not hit:
            return m, False
        comp = sum(self.atoms[i] for i in hit)
        comp = self._norm(comp)
        out = self.alpha * comp + (1 - self.alpha) * m.detach()
        self.n_route += 1
        self.n_hit += len(hit)
        return self._norm(out), True

    def stats(self):
        return {'n_atom_slots': sum(1 for a in self.atoms if a is not None),
                'n_gen': N_GEN, 'n_add': self.n_add,
                'n_route': self.n_route, 'n_hit': self.n_hit}


class V34_3_Learner(V33_Learner):
    """V33 + CPG 原子路由: 训练积累 per-generator 原子, 部署显式路由重组. 零参数."""

    def __init__(self, rln, pln, V, alpha=0.5, **kw):
        self._alpha = alpha
        super().__init__(rln, pln, V, **kw)
        d = rln.d_model
        self.bank = AtomRouteBank(d, alpha=alpha)

    def _define(self, task, detach=True, use_bank=True):
        m = super()._define(task, detach=False, use_bank=False)
        sig = task.get('gen_sig') or [0] * N_GEN
        if detach:                      # 部署: 显式路由重组 (不积累)
            m, _ = self.bank.route(m, sig)
            m = m.detach()
        else:                            # 训练: 仅积累原子 (滚动平均)
            if self.bank is not None:
                self.bank.add(m, sig)
        return m


# ============================================================
# 驱动 (与 V34.2 同构, 复用其运行管线)
# ============================================================
def run_cell(key, cfg, s1_state, tasks, U_STRUCT, meta_iters=40, K=20,
             seed=42, steps_per_task=60, verbose=True):
    cpg = cfg['cpg']
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
    if cpg:
        learner = V34_3_Learner(rln, pln, 200, K=K, seed=seed, sleep_iters=2,
                                n_resp=4, use_intent=True,
                                alpha=cfg.get('alpha', 0.5))
    else:
        learner = V33_Learner(rln, pln, 200, K=K, seed=seed, sleep_iters=2,
                              n_resp=4, use_intent=True, n_define_refine=0)
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
    n_params = sum(p.numel() for p in rln.parameters()) + sum(p.numel() for p in pln.parameters())
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'world': cfg.get('world', 'new'),
           'factors': {'Cpg': cpg, 'alpha': cfg.get('alpha', None)},
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
    'B':  dict(world='new', cpg=False, label='B-base-d128'),
    'C':  dict(world='new', cpg=True, alpha=0.5, label='C-cpg-route-mix'),
    'Cr': dict(world='new', cpg=True, alpha=1.0, label='Cr-cpg-route-pure'),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='B,C,Cr')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("=" * 70); print("V34.3 CPG 原子路由 — 零参数原子级显式重组")
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
    out = ROOT / 'results' / 'v34_3_report.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values()) / len(u)
        print("%s unseen=%.4f struct=%s params=%d %s" %
              (r['cell'], ua, r['unseen_by_struct'], r['n_params'], r['aux']))


if __name__ == '__main__':
    main()