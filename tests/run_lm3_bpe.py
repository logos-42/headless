"""LM3 - BPE + cross-domain continual learning + iid baseline
Data: en(wikitext-2) -> zh(chinese novel) -> code(python)"""
import os, sys, time, json, math, random, argparse, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_v31_meta_learning import V31_RLN

DATA = Path("/tmp/lm3data")
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
CHUNK = 256


class BPE:
    """char-level BPE: base = all unique chars, merge top pairs."""
    def __init__(self):
        self.base = {}
        self.merges = {}

    def train(self, texts, num_merges=1000, sample=300000):
        rng = random.Random(42)
        concat = ""
        for t in texts:
            concat += t[:sample] if len(t) > sample else t
        chars = sorted(set(concat))
        self.base = {c: i for i, c in enumerate(chars)}
        nxt = len(chars)
        ids = [self.base[c] for c in concat]
        for m in range(num_merges):
            pairs = collections.Counter(zip(ids, ids[1:]))
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            self.merges[(a, b)] = nxt
            nxt += 1
            new_ids = []
            i = 0
            while i < len(ids):
                if (i < len(ids) - 1 and (ids[i], ids[i+1]) in self.merges):
                    new_ids.append(self.merges[(ids[i], ids[i+1])])
                    i += 2
                else:
                    new_ids.append(ids[i]); i += 1
            ids = new_ids
        self.vocab = nxt
        print("BPE vocab: %d (base %d + %d merges)" % (nxt, len(chars), num_merges))

    def encode(self, s):
        ids = [self.base[c] for c in s if c in self.base]
        if not ids:
            return []
        # 线性左→右归并: head 与后续 token 合并后不前进 (等价于原 del 版, O(n))
        head = ids[0]
        out = []
        for nxt in ids[1:]:
            k = (head, nxt)
            if k in self.merges:
                head = self.merges[k]
            else:
                out.append(head)
                head = nxt
        out.append(head)
        return out

class LM3System:
    def __init__(self, d_model=192, d_state=12, n_layers=2,
                 steps_per_domain=300, mode="sequential", seed=42,
                 method="base", bpe_cache=None):
        self.d_model = d_model; self.d_state = d_state
        self.n_layers = n_layers; self.steps_per_domain = steps_per_domain
        self.mode = mode; self.method = method; self.seed = seed
        self.stage = 0; self.history = []
        # domains: en(2M) zh(2.5M) code(2M)
        raw_en = open(DATA / "train.txt").read()[:2000000]
        raw_zh = open(DATA / "zh.txt").read()[:2500000]
        raw_code = open(DATA / "code.txt").read()[:2000000]
        self.domains = {"en": raw_en, "zh": raw_zh, "code": raw_code}
        self.bpe = BPE()
        if bpe_cache and os.path.exists(bpe_cache):
            import json as _json
            ck = _json.load(open(bpe_cache))
            self.bpe.base = ck['base']
            self.bpe.merges = {
                tuple(int(x) for x in k.split('_')): v
                for k, v in ck['merges'].items()}
            self.bpe.vocab = ck['vocab']
            print("BPE loaded from cache: vocab %d" % self.bpe.vocab)
        else:
            self.bpe.train(list(self.domains.values()), num_merges=800)
            if bpe_cache:
                import json as _json
                ck = {'base': self.bpe.base,
                      'merges': {"%d_%d" % k: v
                                 for k, v in self.bpe.merges.items()},
                      'vocab': self.bpe.vocab}
                _json.dump(ck, open(bpe_cache, 'w'))
                print("BPE cached to %s" % bpe_cache)
        # 预编码: 每个域只 encode 一次 (BPE 是主要开销)
        self.domain_ids = {d: self.bpe.encode(t)
                           for d, t in self.domains.items()}
        self._build_model()

    def _build_model(self):
        random.seed(self.seed); torch.manual_seed(self.seed)
        self.rln = V31_RLN(self.bpe.vocab + 2, d_model=self.d_model,
                           d_state=self.d_state, n_layers=self.n_layers)
        self.lm_head = nn.Linear(self.d_model, self.bpe.vocab)
        self.opt = torch.optim.Adam(
            list(self.rln.parameters()) + list(self.lm_head.parameters()),
            lr=1e-3)
        # OML 双层: 快权重 (head, 高 lr) + 慢权重 (rln, 低 lr)
        self.opt_outer = torch.optim.Adam(self.rln.parameters(), lr=1e-4)
        self.opt_inner = torch.optim.Adam(self.lm_head.parameters(), lr=1e-2)
        self._n_params = (sum(p.numel() for p in self.rln.parameters())
                          + sum(p.numel() for p in self.lm_head.parameters()))
        # replay buffer: {domain: [chunks]}
        self.replay = {}
        self.replay_size = 60

    def _chunks(self, dname, rng):
        ids = self.domain_ids[dname]
        n = len(ids) - CHUNK - 1
        if n <= 0: return []
        idxs = list(range(0, n, CHUNK // 2))
        rng.shuffle(idxs)
        for i in idxs:
            x = ids[i:i + CHUNK]; y = ids[i + 1:i + CHUNK + 1]
            if len(x) == CHUNK and len(y) == CHUNK:
                yield torch.tensor(x), torch.tensor(y)

    def _eval_chunks(self, dname, n=100):
        ids = self.domain_ids[dname]
        n_avail = len(ids) - CHUNK - 1
        if n_avail <= 0: return []
        rng = random.Random(0)
        idxs = rng.sample(range(0, n_avail, max(1, n_avail // (n * 2))),
                          min(n, n_avail // 2 + 1))
        for i in idxs:
            x = ids[i:i + CHUNK]; y = ids[i + 1:i + CHUNK + 1]
            if len(x) == CHUNK and len(y) == CHUNK:
                yield torch.tensor(x), torch.tensor(y)

    def _stream(self, steps, rng):
        """yield (domain_name, x, y) for training steps."""
        if self.mode == "iid":
            # mixed: all domains, shuffled chunks
            iters = {}
            for d in self.domains:
                iters[d] = self._chunks(d, rng)
            for _ in range(steps):
                d = rng.choice(list(self.domains))
                try:
                    x, y = next(iters[d])
                except StopIteration:
                    iters[d] = self._chunks(d, rng)
                    x, y = next(iters[d])
                yield d, x, y
        else:
            # sequential: current domain only
            d = self.stage_domain
            it = self._chunks(d, rng)
            for _ in range(steps):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = self._chunks(d, rng); x, y = next(it)
                yield self.stage_domain, x, y

    def train_stage(self, steps=None):
        steps = steps or self.steps_per_domain
        rng = random.Random(self.seed + self.stage * 7)
        t0 = time.time(); losses = []
        qit = None
        for s, (d, x, y) in enumerate(self._stream(steps, rng)):
            self.rln.train()
            if self.method == "oml":
                self.opt_inner.zero_grad()
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                loss.backward(); self.opt_inner.step()
                # 外循环: 低频慢更新表示 (opt_outer 只含 rln 参数)
                if s % 10 == 0:
                    self.opt_outer.zero_grad()
                    h2 = self.rln(x.unsqueeze(0))[0]
                    logits2 = self.lm_head(h2[-CHUNK:])
                    loss2 = F.cross_entropy(logits2, y)
                    loss2.backward(); self.opt_outer.step()
            elif self.method == "oml2":
                # 正确 OML: per-domain meta-step
                # 内循环: 克隆 head, 在 support chunk 上适应 K 步
                # 外循环: 用 fast head 在 query chunk 上的 loss 更新全模型
                import copy
                support_x, support_y = x, y
                qd = d if self.mode == "iid" else self.stage_domain
                if qit is None:
                    qit = self._chunks(qd, rng)
                try:
                    query_x, query_y = next(qit)
                except StopIteration:
                    qit = self._chunks(qd, rng)
                    query_x, query_y = next(qit)
                # 快照 head (state_dict 拷贝, 快于 deepcopy)
                fast_state = {k: v.detach().clone()
                              for k, v in self.lm_head.state_dict().items()}
                with torch.no_grad():
                    h_f = self.rln(support_x.unsqueeze(0))[0]
                    h_fd = h_f[-CHUNK:].detach()
                # 内循环: 手动 SGD (K=2), 只动 head 快照
                for _k in range(2):
                    w = fast_state['weight'].detach().requires_grad_(True)
                    b = fast_state['bias'].detach().requires_grad_(True)
                    logits = h_fd @ w.t() + b
                    l_f = F.cross_entropy(logits, support_y)
                    g_w, g_b = torch.autograd.grad(l_f, (w, b))
                    with torch.no_grad():
                        fast_state['weight'] = w - 0.1 * g_w
                        fast_state['bias'] = b - 0.1 * g_b
                # 外循环: fast head 在 query 上的 meta-loss
                self.opt.zero_grad()
                h_q = self.rln(query_x.unsqueeze(0))[0]
                logits_q = h_q[-CHUNK:] @ fast_state['weight'].t() + fast_state['bias']
                l_q = F.cross_entropy(logits_q, query_y)
                l_q.backward()
                self.opt.step()
                # 周期性合并 fast head 回原 head (OML 权重整合)
                if s % 10 == 0:
                    self.lm_head.load_state_dict(
                        {k: v.detach().clone()
                         for k, v in fast_state.items()})
                loss = l_q.item()
                # OML 双循环 + replay 混入 (组合实验)
                self.opt_inner.zero_grad()
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                for rd, chunk_list in self.replay.items():
                    for rx, ry in rng.sample(
                            chunk_list, min(2, len(chunk_list))):
                        loss = loss + 0.5 * F.cross_entropy(
                            self.lm_head(
                                self.rln(rx.unsqueeze(0))[0][-CHUNK:]),
                            ry)
                loss.backward(); self.opt_inner.step()
                if s % 10 == 0:
                    self.opt_outer.zero_grad()
                    h2 = self.rln(x.unsqueeze(0))[0]
                    logits2 = self.lm_head(h2[-CHUNK:])
                    loss2 = F.cross_entropy(logits2, y)
                    loss2.backward(); self.opt_outer.step()
            elif self.method == "replay":
                self.opt.zero_grad()
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                # 混入 replay buffer (旧域样本, 每域抽样 2 条)
                for rd, chunk_list in self.replay.items():
                    for rx, ry in rng.sample(
                            chunk_list, min(2, len(chunk_list))):
                        loss = loss + 0.5 * F.cross_entropy(
                            self.lm_head(
                                self.rln(rx.unsqueeze(0))[0][-CHUNK:]),
                            ry)
                loss.backward(); self.opt.step()
            else:
                self.opt.zero_grad()
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                loss.backward(); self.opt.step()
            losses.append(loss.item())
        return sum(losses) / len(losses), time.time() - t0

    def evaluate_domain(self, name):
        self.rln.eval()
        total_ce = total_acc = n = 0
        with torch.no_grad():
            for x, y in self._eval_chunks(name):
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()
                total_ce += loss.item(); total_acc += acc; n += 1
        return round(total_ce / n, 4), round(total_acc / n, 4)

    def save(self, tag):
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / ("lm3_%s_%s_%s.pt" % (self.method, self.mode, tag))
        torch.save({"stage": self.stage, "mode": self.mode,
                    "d_model": self.d_model, "d_state": self.d_state,
                    "n_layers": self.n_layers, "vocab": self.bpe.vocab,
                    "merges": self.bpe.merges, "base": self.bpe.base,
                    "rln": self.rln.state_dict(),
                    "lm_head": self.lm_head.state_dict(),
                    "history": self.history}, path)
        return path

    def resume(self, path):
        ck = torch.load(path)
        self.stage = ck["stage"]; self.mode = ck["mode"]
        self.history = ck["history"]
        self.bpe.merges = ck["merges"]; self.bpe.base = ck["base"]
        self.bpe.vocab = ck["vocab"]
        self.rln.load_state_dict(ck["rln"])
        self.lm_head.load_state_dict(ck["lm_head"])
        return self.stage

    def run(self, domains=("en", "zh", "code")):
        print("=" * 70)
        print("LM3 BPE cross-domain %s: %.2fM params" % (self.mode, self._n_params / 1e6))
        print("domains: %s" % list(domains))
        print("=" * 70)
        n_stages = 1 if self.mode == "iid" else len(domains)
        for s in range(n_stages):
            self.stage += 1
            self.stage_domain = domains[s] if self.mode == "sequential" else None
            loss, t = self.train_stage()
            if self.method == "replay" and self.mode == "sequential":
                # 存当前域代表性 chunks 进 buffer
                rng2 = random.Random(self.stage)
                buf = []
                it = self._chunks(self.stage_domain, rng2)
                for _ in range(self.replay_size):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        break
                    buf.append((x, y))
                self.replay[self.stage_domain] = buf
            row = {"stage": self.stage, "loss": round(loss, 4), "train_s": round(t)}
            ev = {}
            for d in domains:
                ce, acc = self.evaluate_domain(d)
                ev[d] = {"ce": ce, "acc": acc}
            row["eval"] = ev
            self.history.append(row)
            print("[%s] loss=%.4f | en(%.4f/%.4f) zh(%.4f/%.4f) code(%.4f/%.4f) (%.0fs)"
                  % (self.stage_domain or "mixed", loss,
                     ev["en"]["ce"], ev["en"]["acc"],
                     ev["zh"]["ce"], ev["zh"]["acc"],
                     ev["code"]["ce"], ev["code"]["acc"], t), flush=True)
            self.save("latest")
        self._write_report(domains)
        return self.stage

    def _write_report(self, domains):
        lines = ["# LM3 BPE cross-domain report (%s / %s)" % (self.method, self.mode), "",
                 "- model: %.2fM (d=%d, s=%d, layers=%d), BPE vocab=%d"
                 % (self._n_params / 1e6, self.d_model, self.d_state,
                    self.n_layers, self.bpe.vocab),
                 "- domains: en(wikitext-2 2M) -> zh(chinese novel 2.5M) -> code(python 2M)",
                 "- mode: %s" % self.mode, "",
                 "| stage | loss | en ce/acc | zh ce/acc | code ce/acc |",
                 "|:--|--:|:--|:--|:--|"]
        for row in self.history:
            ev = row["eval"]
            lines.append("| %s | %.4f | %.4f/%.4f | %.4f/%.4f | %.4f/%.4f |" % (
                row["stage"], row["loss"],
                ev["en"]["ce"], ev["en"]["acc"],
                ev["zh"]["ce"], ev["zh"]["acc"],
                ev["code"]["ce"], ev["code"]["acc"]))
        lines.append("")
        lines.append("## forgetting (en self-acc: first -> last)")
        first_en = self.history[0]["eval"]["en"]["acc"]
        last_en = self.history[-1]["eval"]["en"]["acc"]
        lines.append("- en: %.4f -> %.4f, forget %.4f" % (
            first_en, last_en, round(first_en - last_en, 4)))
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / ("lm3_%s_%s_report.md" % (self.method, self.mode))).write_text(
            "\n".join(lines), encoding="utf-8")
        print("report: %s" % (RESULT_DIR / ("lm3_%s_%s_report.md" % (self.method, self.mode))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--d-state", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--steps-per-domain", type=int, default=300)
    ap.add_argument("--mode", default="sequential", choices=["sequential", "iid"])
    ap.add_argument("--method", default="base",
                    choices=["base", "replay", "oml", "combo", "oml2"])
    ap.add_argument("--bpe-cache", type=str, default=None)
    args = ap.parse_args()
    sys = LM3System(d_model=args.d_model, d_state=args.d_state,
                    n_layers=args.n_layers,
                    steps_per_domain=args.steps_per_domain,
                    mode=args.mode, method=args.method,
                    bpe_cache=args.bpe_cache)
    done = sys.run()
    print("done: %s, %d stages" % (args.mode, done))


if __name__ == '__main__':
    main()
