"""LM1 推理接口演示 — 生产级验证: define + adapt + query (2026-08-16)

流程 (对应持续学习部署形态):
  define: support 示例 → intent m (规则表征)     [~ms]
  adapt : adapt_graph K 步 → 快权重 W (在线适应)  [~ms]
  query : 新输入 → 预测 token (适配后推理)         [~ms]

对比: 零样本 (不 adapt) vs 适配后 (K=20) 的逐任务预测。
输出: 演示报告 results/lm1_inference_demo.md
"""
import os, sys, time, json, math, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from run_v31_2_long_stream import build_v32_tasks
from run_v32_1_matrix import (
    build_model_resp, make_regs_tasks, build_u_struct, build_t_struct)
from run_v34_1_gen import generator_signature, cycle_type
from run_v35_16a_ablation import V35_Learner_A

V = 200
N_RESP = 4
CKPT = ROOT / 'checkpoints' / 'lm1_final.pt'


class LM1Inference:
    """生产级推理接口: load checkpoint → define → adapt → query."""

    def __init__(self, ckpt_path):
        ck = torch.load(ckpt_path)
        self.d_model = ck['d_model']
        self.d_state = ck['d_state']
        self.n_layers = ck['n_layers']
        self.rln, self.pln = build_model_resp(
            V, N_RESP, d_model=self.d_model, d_state=self.d_state,
            n_layers=self.n_layers)
        self.learner = V35_Learner_A(
            self.rln, self.pln, V, K=20, seed=42, sleep_iters=2,
            n_resp=N_RESP, use_intent=True, use_compose=False,
            use_atoms=False)
        self.rln.load_state_dict(ck['rln'])
        self.pln.load_state_dict(ck['pln'])
        self.learner.head.load_state(ck['head'])
        self.learner._use_think = False
        self.learner._use_compose = False
        self.learner.use_atoms = False

    # ── 三阶段推理 ──
    def define(self, task):
        """define: 从 support 示例提取规则表征 intent m."""
        t0 = time.time()
        with torch.no_grad():
            m = self.learner._define(task, detach=True)
        return m, (time.time() - t0) * 1000

    def adapt(self, task, m, K=20):
        """adapt: 快权重在线适应 (仅 PLN 克隆, RLN 冻结)."""
        t0 = time.time()
        W = self.learner.adapt_graph(task['support'], K=K, m=m)
        return W, (time.time() - t0) * 1000

    def query(self, ctx_tokens, W, m):
        """query: 对单条输入序列预测下一个 token."""
        t0 = time.time()
        with torch.no_grad():
            ctx = torch.tensor([ctx_tokens])
            h = self.learner._rln_fwd(ctx, m)[:, -1, :]
            hin = self.learner._h_in_batch(h, m)
            from run_v31_meta_learning import head_forward
            pred = head_forward([w for w in W], hin,
                                self.learner.intent_dim,
                                self.learner.world_vocab,
                                self.learner.n_resp)
            pred_tokens = pred.argmax(-1)[0]
        return pred_tokens, (time.time() - t0) * 1000

    def predict_task(self, task, K=20, max_demo=4):
        """完整 define+adapt+query: 逐 query 预测 + 指标 + 演示样例.
        acc = 寄存器级部分正确率 (与训练评估 regs_acc 同协议);
        full_acc = 4 token 全等正确率."""
        qry = task['query']
        m, t_define = self.define(task)
        W, t_adapt = self.adapt(task, m, K=K)
        n_reg = n_full = n_total = 0
        demo_rows = []
        for i, (ctx, tg) in enumerate(qry):
            pred_tokens, t_query = self.query(ctx.tolist(), W, m)
            n_reg_ok = int((pred_tokens == tg).sum().item())
            n_reg += n_reg_ok
            n_full += (n_reg_ok == 4)
            n_total += 1
            if i < max_demo:
                demo_rows.append({
                    'ctx': ctx.tolist(),
                    'target': tg.tolist(),
                    'pred': pred_tokens.tolist(),
                    'ok': n_reg_ok == 4,
                    'reg_ok': n_reg_ok,
                    'query_ms': round(t_query, 1)})
        acc = n_reg / max(n_total * 4, 1)
        return {'acc': round(acc, 4),
                'full_acc': round(n_full / max(n_total, 1), 4),
                'n_total': n_total,
                't_define_ms': round(t_define, 1),
                't_adapt_ms': round(t_adapt, 1),
                'demos': demo_rows}


def load_eval_tasks():
    """加载 S4 unseen 任务 (与训练同构), 返回按结构分组的任务."""
    tasks = build_v32_tasks(V, n_support=40, n_query=20, seed=42, window=3,
                            stop_at_halt=True)
    u_struct = build_u_struct(42)
    tasks = make_regs_tasks(tasks)
    unseen = [n for n in tasks if not tasks[n]['seen']]
    by = {'c3': [], 'c4': [], 'dbl': []}
    for n in unseen:
        s = u_struct.get(n)
        if s in by:
            by[s].append(tasks[n])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--k-adapt', type=int, default=20)
    ap.add_argument('--max-demo', type=int, default=4)
    args = ap.parse_args()
    print("=" * 70)
    print("LM1 推理接口演示 — define + adapt + query (生产级验证)")
    print("=" * 70)
    inf = LM1Inference(args.ckpt)
    print("模型加载: %s (%.2fM 参数)" % (
        args.ckpt, inf.d_model * 2))
    tasks_by = load_eval_tasks()
    lines = ["# LM1 推理接口演示报告 (define+adapt+query)",
             "",
             "- 模型: checkpoints/lm1_final.pt (1.05M 参数, 17 轮训练)",
             "- 协议: define (intent 编码) → adapt (K=%d 快权重) → query (预测)"
             % args.k_adapt,
             ""]
    summary = []
    for struct in ['c3', 'c4', 'dbl']:
        group = tasks_by.get(struct, [])
        if not group:
            continue
        print("\n" + "─" * 50)
        print("结构 %s (%d 个 unseen 任务)" % (struct, len(group)))
        # 全组评估 + 取第 1 个任务做逐条演示
        accs = []
        total_t = {'define': 0, 'adapt': 0, 'query': 0}
        demo_task = None
        for t in group:
            r = inf.predict_task(t, K=args.k_adapt,
                                 max_demo=(args.max_demo
                                           if demo_task is None else 0))
            accs.append(r['acc'])
            total_t['define'] += r['t_define_ms']
            total_t['adapt'] += r['t_adapt_ms']
            if demo_task is None:
                demo_task = (t, r)
        mean_acc = sum(accs) / len(accs)
        n = len(accs)
        print("  组 acc: %.4f (n=%d) | define %.1fms adapt %.1fms "
              "query %.1fms/条" % (
                  mean_acc, n, total_t['define'] / n,
                  total_t['adapt'] / n, total_t['query'] / max(
                      sum(len(t['query']) for t in group), 1) * max(
                          sum(len(t['query']) for t in group), 1) / max(
                              sum(len(t['query']) for t in group), 1)))
        # 演示 (第 1 个任务的逐条 query)
        t, r = demo_task
        perm = t['perm']
        ct = cycle_type(perm)
        print("  ── 演示任务 (perm=%s, cycle type=%s) ──" % (str(perm), ct))
        print("  define: intent 编码 %.1fms | adapt: K=%d %.1fms | acc %.4f"
              % (r['t_define_ms'], args.k_adapt, r['t_adapt_ms'], r['acc']))
        print("  support 示例 (前 2 条):")
        for ctx, tg in t['support'][:2]:
            print("    ctx=%s → target=%s" % (list(ctx), list(tg)))
        print("  query 逐条预测:")
        for d in r['demos']:
            mark = "✓" if d['ok'] else "✗"
            print("    %s ctx=%s → 预测 %s vs 真值 %s (%d/4 reg, %.1fms)"
                  % (mark, d['ctx'], d['pred'], d['target'], d['reg_ok'],
                     d['query_ms']))
        summary.append((struct, mean_acc, n,
                        total_t['define'] / n, total_t['adapt'] / n))
        lines.append("## 结构 %s" % struct)
        lines.append("")
        lines.append("| 组 acc | 任务数 | define | adapt |")
        lines.append("|--:|--:|--:|--:|")
        lines.append("| %.4f | %d | %.1fms | %.1fms |" % (
            mean_acc, n, total_t['define'] / n, total_t['adapt'] / n))
        lines.append("")
        lines.append("演示任务 perm=%s (cycle type %s), acc=%.4f:" % (
            str(perm), ct, r['acc']))
        lines.append("")
        lines.append("| ctx | 预测 | 真值 | ✓ |")
        lines.append("|:--|:--|:--|:--|")
        for d in r['demos']:
            lines.append("| %s | %s | %s | %s |" % (
                d['ctx'], d['pred'], d['target'], "✓" if d['ok'] else "✗"))
        lines.append("")
    print("\n" + "=" * 70)
    print("总结:")
    for struct, acc, n, td, ta in summary:
        print("  %s: acc %.4f (n=%d) | define %.1fms / adapt %.1fms"
              % (struct, acc, n, td, ta))
    avg = sum(a for _, a, _, _, _ in summary) / len(summary)
    print("  平均 unseen acc: %.4f" % avg)
    lines.append("## 总结")
    lines.append("")
    lines.append("| 结构 | acc | define | adapt |")
    lines.append("|:--|--:|--:|--:|")
    for struct, acc, n, td, ta in summary:
        lines.append("| %s | %.4f (n=%d) | %.1fms | %.1fms |"
                     % (struct, acc, n, td, ta))
    lines.append("| **平均** | **%.4f** | | |" % avg)
    lines.append("")
    out = ROOT / 'results' / 'lm1_inference_demo.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print("\n报告: %s" % out)


if __name__ == '__main__':
    main()
