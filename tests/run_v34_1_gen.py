"""V34.1 生成元签名监督 — 从"排列实例记忆"到"生成元组合语言"

假设 (H34.1): V34 结构分离提示根因是组合泛化缺口 — RLN 把规则学成实例
key-value (perm → Δregs), 而非 S4 群生成元组合代数。无法把已学生成元
(swap) 组合成未见结构 (c4/dbl)，尽管 c4/dbl = 纯由已见 swap 生成元合成。

干预: 在 define 的 latent m 上加生成元签名头 (gen_head) + 循环类头
(cyc_head)，监督 m 预测该 perm 的"对换生成元组合签名"(6 维原子，全在
训练见过) + 循环类型 (ident/swap/c3/c4/dbl)。

关键点 (vs V32.1 P 失败): 旧 perm aux 解码 m→4×4 排列实例(含未见结构，
塌缩)。新签名头解码 m→train 已掌握的生成元原子 → 未见 c4/d = 已学原子的
绑定形式，组合泛化转移目标。

Cell:
  B   : V33 base (d128) = 基线，全 12 unseen (含 dbl)
  G  : + 生成元签名监督 (6 原子) 
  GC : + 生成元签名 + 循环类型碳竞银行分组 (结构利用)

指标: 结构分组 regs_acc (c3/c4/dbl)，生成元签名准确率 (train/unseen)。
"""
import os, sys, time, json, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from hibs_lnn.swifttd_head import SwiftTDHead
from run_v31_meta_learning import (
    batch_pairs, head_forward, stage1_pretrain_world,
    V31_RLN, V31_PLN)
from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    V32_1_Learner, build_model_resp, make_regs_tasks, build_u_struct,
    build_t_struct, tome_fraction)
from run_v33_drl import V33_Learner, V33_3_Learner

ROOT = Path(__file__).resolve().parent.parent

TRANS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
IDENT = (0,1,2,3)

def _apply(perm, g):
    p = list(perm); a, b = g; p[a], p[b] = p[b], p[a]
    return tuple(p)

_gen_cache = {}
def generator_signature(perm):
    """6 维 binary: 该 perm 的最短 wrote 中出现的实现原子集合. """
    if perm in _gen_cache:
        return _gen_cache[perm]
    from collections import deque
    q = deque([(IDENT, [])]); seen = {IDENT}; found = None
    while q and found is None:
        cur, w = q.popleft()
        if cur == perm:
            found = w
            break
        if len(w) >= 6:
            continue
        for gi, g in enumerate(TRANS):
            nxt = _apply(cur, g)
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, w + [gi]))
    if found is None:
        _gen_cache[perm] = None
        return None
    sig = [0]*6
    for gi in found:
        sig[gi] = 1
    _gen_cache[perm] = (sig, len(found))
    return _gen_cache[perm]

def cycle_type(perm):
    seen = set(); lens = []
    for i in range(4):
        if i in seen: continue
        c = []; j = i
        while j not in seen:
            seen.add(j); c.append(j); j = perm[j]
        if len(c) > 1: lens.append(len(c))
    lens.sort()
    if not lens: return 0
    if lens == [2]: return 1
    if lens == [3]: return 2
    if lens == [4]: return 3
    if lens == [2,2]: return 4
    return 0

CYCLET = 5

def build_gen_labels(tasks):
    labels = {}
    for name in tasks:
        sig = generator_signature(tasks[name]['perm'])
        labels[name] = {'sig': (sig[0] if sig else None),
                        'cyc': cycle_type(tasks[name]['perm'])}
    return labels


# ============================================================
# V34_1_Learner: V33 + 生成元签名监督
# ============================================================
class V34GenLearner(V33_Learner):
    """V33 (define-refine) + 生成元语义头: m → 6 原子 碳 label + 5 循环类."""
    def __init__(self, rln, pln, V, lambda_gen=0.3,
                 outer_lr=1e-3, n_define_refine=0,
                 refine_hidden=None, refine_layers=2, **kw):
        self.lambda_gen = lambda_gen
        self._outer_lr = outer_lr
        super().__init__(rln, pln, V, n_define_refine=n_define_refine,
                         outer_lr=outer_lr, refine_hidden=refine_hidden,
                         refine_layers=refine_layers, **kw)
        d = rln.d_model
        self.gen_head = nn.Linear(d, 6)
        self.cyc_head = nn.Linear(d, CYCLET)
        extra = list(self.gen_head.parameters()) + list(self.cyc_head.parameters())
        base = (list(rln.parameters()) + list(pln.parameters())
                + (list(self.rule_encoder.parameters()) if self.use_rule_enc
                   else list(self.intent_net.parameters()))
                + list(self.refine_net.parameters()) + list(self.probe_head.parameters()))
        self.outer_opt = torch.optim.Adam(base + extra, lr=self._outer_lr)
        self.gen_trace = []; self.cyc_trace = []

    def meta_step(self, tasks, train_rules, K=None):
        K = K or self.K
        self.outer_opt.zero_grad()
        total_loss = torch.tensor(0.0)
        rng = random.Random(self.seed + self._meta_iters)
        chosen = rng.sample(train_rules,
                            min(self.n_tasks_per_step, len(train_rules)))
        for rname in chosen:
            t = tasks[rname]
            m = self._define(t, detach=False) if self.use_intent else None
            W = self.adapt_graph(t['support'], m=m, K=K)
            ctx, tgt = batch_pairs(t['query'], max_batch=16)
            hidden = self._rln_fwd(ctx, m)
            h = hidden[:, -1, :]
            h_in = self._h_in_batch(h, m)
            pred = head_forward(W, h_in, self.intent_dim, self.world_vocab,
                                self.n_resp)
            pred = self._route_logits(pred, m)
            loss = F.cross_entropy(pred.view(-1, self.world_vocab),
                                   tgt.view(-1).clamp(0, self.V + 1))
            total_loss = total_loss + loss / len(chosen)
            if m is not None and t.get('gen_sig') is not None:
                g = torch.tensor(t['gen_sig'], dtype=torch.float)
                lg = F.binary_cross_entropy_with_logits(
                    self.gen_head(m).view(1,-1), g.view(1,-1))
                lc = F.cross_entropy(
                    self.cyc_head(m).view(1,-1),
                    torch.tensor([t['gen_cyc']], dtype=torch.long))
                total_loss = total_loss + (self.lambda_gen*lg)/len(chosen) \
                    + (self.lambda_gen*lc)/len(chosen)
                self.gen_trace.append(lg.item()); self.cyc_trace.append(lc.item())
        total_loss.backward()
        params = [p for p in self.rln.parameters() if p.requires_grad] \
            + [p for p in self.pln.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.outer_opt.step()
        self._meta_iters += 1
        return total_loss.item()

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
        gl = build_gen_labels({n: tasks[n]})
        tasks[n]['gen_sig'] = gl[n]['sig']
        tasks[n]['gen_cyc'] = gl[n]['cyc']
    random.seed(seed); torch.manual_seed(seed)
    rln, pln = build_model_resp(200, 4, s1_state)
    if lm == 'gen':
        learner = V34GenLearner(rln, pln, 200, K=K, seed=seed,
                                sleep_iters=2, n_resp=4, use_intent=True,
                                lambda_gen=cfg.get('lambda_gen', 0.3),
                                n_define_refine=cfg.get('define_refine', 0),
                                use_route_recur=cfg.get('route_recur', False),
                                use_route_action=cfg.get('route_action', False),
                                use_pattern_bank=cfg.get('bank', False),
                                bank_struct=cfg.get('bank_struct', False))
    else:
        learner = V33_Learner(rln, pln, 200, K=K, seed=seed, sleep_iters=2,
                              n_resp=4, use_intent=True,
n_define_refine=cfg.get('define_refine', 0))
    learner._task_struct = {n: tasks[n]['struct'] for n in tasks}
    t0 = time.time()
    n_iters = cfg.get('meta_iters', meta_iters)
    for it in range(n_iters):
        learner.meta_iters = it
        loss = learner.meta_step(tasks, train_names)
        if verbose and (it + 1) % 10 == 0:
            print("    [%s] it %d/%d loss=%.4f" % (key, it+1, n_iters, loss), flush=True)
    trained_head = learner.head.state()
    st = learner.long_stream(tasks, stream, steps_per_task=steps_per_task,
                             sleep_iters=2, head_state=trained_head)
    print("    [%s] stream done (%.0fs)" % (key, time.time()-t0), flush=True)
    by = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen_names:
        s = U_STRUCT[n]
        if s in by: by[s].append(st['acc_after'][n])
    struct_mean = {k: (sum(v)/len(v) if v else None) for k, v in by.items()}
    aux = {}
    if hasattr(learner, 'gen_head'):
        gt = gtot = gu = gtotu = 0
        with torch.no_grad():
            gen_heads_weight = learner.gen_head.weight.data
            gen_bias = learner.gen_head.bias.data if learner.gen_head.bias is not None else None
        gen_head_static = learner.gen_head
        for n in train_names:
            mm = learner.compute_goal_intent(tasks[n]) if hasattr(learner, 'compute_goal_intent') else learner._define(tasks[n], detach=True)
            p = gen_head_static(mm).sigmoid() > 0.5
            gt += (p == torch.tensor(tasks[n]['gen_sig']).float()).sum().item()
            gtot += 6
        for n in unseen_names:
            mm = learner.compute_goal_intent(tasks[n]) if hasattr(learner, 'compute_goal_intent') else learner._define(tasks[n], detach=True)
            p = gen_head_static(mm).sigmoid() > 0.5
            gu += (p == torch.tensor(tasks[n]['gen_sig']).float()).sum().item()
            gtotu += 6
        aux['gen_acc_train'] = round(gt/gtot, 4) if gtot else None
        aux['gen_acc_unseen'] = round(gu/gtotu, 4) if gtotu else None
        if learner.gen_trace: aux['gen_loss_last10'] = round(sum(learner.gen_trace[-10:])/10, 4)
        if learner.cyc_trace: aux['cyc_loss_last10'] = round(sum(learner.cyc_trace[-10:])/10, 4)
    rec = {'cell': key, 'label': cfg['label'], 'seed': seed,
           'world': cfg.get('world', 'new'),
           'factors': {'GenSig': lm == 'gen',
                       'DefineRefine': cfg.get('define_refine', 0) > 0,
                       'Bank': cfg.get('bank', False)},
           'time_s': round(time.time() - t0, 1),
           'seen_acc': {n: st['acc_after'][n] for n in train_names},
           'unseen_acc': {n: st['acc_after'][n] for n in unseen_names},
           'unseen_by_struct': struct_mean,
           'aux': aux}
    ua = rec['unseen_acc']
    print(("  [%s] unseen mean=%.4f struct=%s aux=%s" %
           (key, sum(ua.values())/len(ua), struct_mean, aux)), flush=True)
    return rec


CELLS = {
    'B':  dict(world='new', lm='base', define_refine=0,
               label='B-d128-base'),
    'G':  dict(world='new', lm='gen', define_refine=0, lambda_gen=0.3,
           label='G-gen-signature'),
    'GC': dict(world='new', lm='gen', define_refine=2, lambda_gen=0.3,
           label='GC-gen+refine'),
}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='B,G')
    ap.add_argument('--meta-iters', type=int, default=40)
    ap.add_argument('--no-stage1', action='store_true')
    args = ap.parse_args()
    print("="*70); print("V34.1 生成元签名监督"); print("="*70)
    tasks = build_v32_tasks(200, n_support=40, n_query=20, seed=42,
                            window=3, stop_at_halt=True)
    U_STRUCT = build_u_struct(42)
    print("未见表:", {n: build_u_struct(42)[n] for n in tasks if not tasks[n]['seen']})
    s1 = None
    if not args.no_stage1:
        for cand in ('/tmp/v32_s1_state.pt', '/tmp/v31_2_s1_state.pt'):
            if os.path.exists(cand):
                s1 = torch.load(cand); s1 = {k: v.cpu().clone() for k, v in s1.items()}
                print("Stage1: %s" % cand); break
        else:
            print(">>> Stage1 缺失, 请先跑 run_v32_stage1.py"); return
    res = []
    for key in [c.strip() for c in args.cells.split(',')]:
        cfg = CELLS[key]
        print("\n== %s %s ==" % (key, cfg['label']))
        try:
            r = run_cell(key, cfg, s1, tasks, U_STRUCT, meta_iters=args.meta_iters)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("!!! %s 失败: %s" % (key, e)); continue
        res.append(r)
    out = ROOT / 'results' / 'v34_1_report.json'
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print("\nJSON:", out)
    for r in res:
        u = r['unseen_acc']; ua = sum(u.values())/len(u)
        print("%s unseen=%.4f struct=%s aux=%s" % (r['cell'], ua, r['unseen_by_struct'], r['aux']))


if __name__ == '__main__':
    main()