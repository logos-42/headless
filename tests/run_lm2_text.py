"""LM2 docstring"""
import os, sys, time, json, math, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_v31_meta_learning import V31_RLN
DATA = Path("/tmp/lm2data")
CKPT_DIR = ROOT / "checkpoints"
RESULT_DIR = ROOT / "results"
CHUNK = 256

class CharTokenizer:
    """char tokenizer from corpus"""
    def __init__(self, texts):
        chars = set(''.join(texts[:200]))
        self.chars = sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab = len(self.chars)
        print("vocab: %d chars" % self.vocab)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]


class LM2System:
    def __init__(self, d_model=192, d_state=12, n_layers=2,
                 steps_per_domain=500, seed=42):
        self.d_model = d_model; self.d_state = d_state
        self.n_layers = n_layers; self.steps_per_domain = steps_per_domain
        self.seed = seed; self.round = 0; self.history = []
        raw = {d: open(DATA / (d + ".txt")).read()
               for d in ["train", "validation", "test"]}
        self.domains = {"A": raw["train"][:2000000],
                        "B": raw["validation"], "C": raw["test"]}
        self.tok = CharTokenizer(list(self.domains.values()))
        self._build_model()
    def _build_model(self):
        random.seed(self.seed); torch.manual_seed(self.seed)
        self.rln = V31_RLN(self.tok.vocab + 2, d_model=self.d_model,
                           d_state=self.d_state, n_layers=self.n_layers)
        self.lm_head = nn.Linear(self.d_model, self.tok.vocab)
        self.opt = torch.optim.Adam(
            list(self.rln.parameters()) + list(self.lm_head.parameters()),
            lr=1e-3)
        self._n_params = (sum(p.numel() for p in self.rln.parameters())
                          + sum(p.numel() for p in self.lm_head.parameters()))

    def _chunks(self, text, rng):
        ids = self.tok.encode(text)
        n = len(ids) - CHUNK - 1
        if n <= 0: return []
        idxs = list(range(0, n, CHUNK // 2))
        rng.shuffle(idxs)
        for i in idxs:
            x = ids[i:i + CHUNK]; y = ids[i + 1:i + CHUNK + 1]
            if len(x) == CHUNK and len(y) == CHUNK:
                yield torch.tensor(x), torch.tensor(y)

    def _eval_chunks(self, text, n=100):
        ids = self.tok.encode(text)
        n_avail = len(ids) - CHUNK - 1
        if n_avail <= 0: return []
        rng = random.Random(0)
        idxs = rng.sample(range(0, n_avail, max(1, n_avail // (n * 2))),
                          min(n, n_avail // 2 + 1))
        for i in idxs:
            x = ids[i:i + CHUNK]; y = ids[i + 1:i + CHUNK + 1]
            if len(x) == CHUNK and len(y) == CHUNK:
                yield torch.tensor(x), torch.tensor(y)
    def train_domain(self, name, steps=None):
        steps = steps or self.steps_per_domain
        text = self.domains[name]
        rng = random.Random(self.seed + self.round * 7)
        it = self._chunks(text, rng)
        t0 = time.time(); losses = []
        for s in range(steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = self._chunks(text, rng); x, y = next(it)
            self.rln.train(); self.opt.zero_grad()
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
            for x, y in self._eval_chunks(self.domains[name]):
                h = self.rln(x.unsqueeze(0))[0]
                logits = self.lm_head(h[-CHUNK:])
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()
                total_ce += loss.item(); total_acc += acc; n += 1
        return round(total_ce / n, 4), round(total_acc / n, 4)
    def save(self, tag):
        CKPT_DIR.mkdir(exist_ok=True)
        path = CKPT_DIR / ("lm2_" + tag + ".pt")
        torch.save({"round": self.round, "d_model": self.d_model,
                    "d_state": self.d_state, "n_layers": self.n_layers,
                    "vocab": self.tok.vocab,
                    "rln": self.rln.state_dict(),
                    "lm_head": self.lm_head.state_dict(),
                    "history": self.history}, path)
        return path

    def resume(self, path):
        ck = torch.load(path)
        self.round = ck["round"]; self.history = ck["history"]
        self.rln.load_state_dict(ck["rln"])
        self.lm_head.load_state_dict(ck["lm_head"])
        return self.round
    def run(self, domains=("A", "B", "C")):
        print("=" * 70)
        print("LM2 real-text continual learning: %.2fM params, domains %s" % (
            self._n_params / 1e6, list(domains)))
        print("=" * 70)
        for name in domains:
            self.round += 1
            loss, t = self.train_domain(name)
            row = {'domain': name, 'loss': round(loss, 4), 'train_s': round(t)}
            ev = {}
            for d in domains:
                ce, acc = self.evaluate_domain(d)
                ev[d] = {'ce': ce, 'acc': acc}
            row['eval'] = ev
            self.history.append(row)
            print("[%s] loss=%.4f | A(%.4f/%.4f) B(%.4f/%.4f) C(%.4f/%.4f) (%.0fs)"
                  % (name, loss, ev["A"]["ce"], ev["A"]["acc"],
                     ev["B"]["ce"], ev["B"]["acc"],
                     ev["C"]["ce"], ev["C"]["acc"], t), flush=True)
            self.save("latest")
        self._write_report()
        return self.round
    def _write_report(self):
        lines = ["# LM2 real-text continual learning report (wikitext-2 3 domains)",
                 "",
                 "- model: %.2fM (d=%d, s=%d, layers=%d), char vocab=%d"
                 % (self._n_params / 1e6, self.d_model, self.d_state,
                    self.n_layers, self.tok.vocab),
                 "- protocol: domain-sequential streaming -> eval all domains",
                 "",
                 "## forget matrix (row=trained domain, col=eval domain)",
                 "",
                 "| trained | A ce/acc | B ce/acc | C ce/acc |",
                 "|:--|:--|:--|:--|"]
        for row in self.history:
            ev = row['eval']
            lines.append("| %s | %.4f/%.4f | %.4f/%.4f | %.4f/%.4f |" % (
                row['domain'], ev['A']['ce'], ev['A']['acc'],
                ev['B']['ce'], ev['B']['acc'],
                ev['C']['ce'], ev['C']['acc']))
        lines.append("")
        lines.append("## forgetting (self-acc after train -> final)")
        first = {}
        for row in self.history:
            d = row['domain']; first[d] = row['eval'][d]['acc']
        last = self.history[-1]['eval']
        for d in first:
            forget = round(first[d] - last[d]["acc"], 4)
            lines.append("- %s: %.4f -> %.4f, forget %.4f" % (
                d, first[d], last[d]["acc"], forget))
        RESULT_DIR.mkdir(exist_ok=True)
        (RESULT_DIR / 'lm2_report.md').write_text('\n'.join(lines),
                                          encoding='utf-8')
        print('report: %s' % (RESULT_DIR / 'lm2_report.md'))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d-model', type=int, default=192)
    ap.add_argument('--d-state', type=int, default=12)
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--steps-per-domain', type=int, default=500)
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()
    sys = LM2System(d_model=args.d_model, d_state=args.d_state,
                    n_layers=args.n_layers,
                    steps_per_domain=args.steps_per_domain)
    if args.resume:
        sys.resume(args.resume)
        print("resumed from %s (round %d)" % (args.resume, sys.round))
    done = sys.run()
    print("done: %d domains" % done)


if __name__ == '__main__':
    main()
