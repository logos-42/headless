"""V35.4: FractalRLN4 — 4 层分形 RLN + 端到端分形 PLN.

按 lm-principle T1/T2 (prefix_allocation_optimal / connection_density_strict_anti):
  连接密度 (1-d)^(D-1) 幂律递减 → 浅层密集深层稀疏.
  层宽: d_i = max(8, d0 · (1 - i/(n-1))^(D-1)), D=1.5, d0=128
    层 0: 128, 层 1: 104, 层 2: 74, 层 3: 42   (2 层时 128/90)
  层间 Linear 投影 (d_i → d_{i+1}).
  残差门控 (T3/T5): y' = y + gate(y), 零初始化.

PLN 分形: 隐藏宽 = RLN 顶层宽度 (42), 输入 3×42 (意图注入 h;m;h⊙m).
  → 整条管线参数集中前缀 (embed + 浅层宽, 深层 + PLN 窄).

Stage1 结构变化 → 必须重训 (FractalRLN4 + LM head 在 token 流上).
"""
import os, sys, time, math, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from hibs_lnn.ssm_v30_3 import SSM_Layer_V30_3
from tests.run_v31_2_long_stream import build_v32_tasks
from tests.run_v31_meta_learning import make_world_token_stream


def fractal_widths(d0, n, D=1.5, min_w=16, mode='geo'):
    """层宽幂律/等比递减 (分形: 浅层密集深层稀疏).

    mode='geo'  (默认, 匹配 v35-plan §1 的 256/181/128/90):
      等比 ratio = 2^(-1/3) ≈ 0.794 (Murray 定律 T15 的 0.79)
      → d_i = max(min_w, round(d0 · ratio^i))
    mode='power': d_i = max(min_w, d0·(1 - i/(n-1))^(D-1))
    """
    if mode == 'geo':
        ratio = 2 ** (-1 / 3)
        return [max(min_w, int(round(d0 * ratio ** i)))
                for i in range(n)]
    ws = []
    for i in range(n):
        depth = i / max(n - 1, 1)
        ws.append(max(min_w, int(d0 * (1 - depth) ** (D - 1))))
    return ws


class FractalRLN4(nn.Module):
    """4 层分形 RLN: 层宽幂律递减 + 层间投影 + 残差门控 (零初始化)."""

    def __init__(self, vocab_size, d0=128, n_layers=4, d_state=8, D=1.5,
                 use_gate=True):
        super().__init__()
        widths = fractal_widths(d0, n_layers, D=D)
        self.widths = widths
        self.d_model = widths[-1]        # 顶层宽度 = PLN 输入维 (V32 兼容)
        self.d0 = d0
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        # embed 输出 = 顶层宽 (V32 契约: embed 输出宽度 == d_model),
        # 内部先宽后窄 (分形): Embedding(d0) → 投影到 d_model → 升维到 layer0
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, widths[0]),
            nn.Linear(widths[0], widths[-1]))
        self.up = nn.Linear(widths[-1], widths[0])
        self.layers = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(SSM_Layer_V30_3(
                widths[i], d_state=d_state, layer_idx=i, ent_mode='none'))
            if i < n_layers - 1:
                self.projs.append(nn.Linear(widths[i], widths[i + 1]))
            if use_gate:
                g = nn.Sequential(
                    nn.Linear(widths[i], max(8, widths[i] // 2)),
                    nn.Tanh(),
                    nn.Linear(max(8, widths[i] // 2), widths[i]))
                with torch.no_grad():
                    g[2].weight.zero_(); g[2].bias.zero_()
                self.gates.append(g)
        self.use_gate = use_gate

    def forward(self, ids, cond=None):
        x = self.up(self.embed(ids))     # (B, L, widths[0])
        for i, layer in enumerate(self.layers):
            y, _ = layer(x, cond=cond)
            if self.use_gate:
                y = y + self.gates[i](y)
            x = y
            if i < len(self.projs):
                x = self.projs[i](x)
        return x  # (B, L, widths[-1])


class FractalRLN4_LM(nn.Module):
    """Stage1 用: FractalRLN4 + LM head."""

    def __init__(self, vocab_size, d0=128, n_layers=4, d_state=8, D=1.5,
                 use_gate=True):
        super().__init__()
        self.rln = FractalRLN4(vocab_size, d0=d0, n_layers=n_layers,
                               d_state=d_state, D=D, use_gate=use_gate)
        top = self.rln.widths[-1]
        self.head = nn.Linear(top, vocab_size)

    def forward(self, ids):
        h = self.rln(ids)
        return self.head(h), 0.0  # (logits, kl)


def stage1_pretrain_fractal4(tasks, train_rules, V, n_epochs=2, L=64,
                             seed=42, d0=128, n_layers=4, D=1.5):
    """在 token 流上预训练 FractalRLN4 (分形结构)."""
    random.seed(seed); torch.manual_seed(seed)
    ids = make_world_token_stream(tasks, train_rules, V)
    model = FractalRLN4_LM(V + 2, d0=d0, n_layers=n_layers, D=D)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    n_seq = max(1, (len(ids) - 1) // (L + 1))
    t0 = time.time()
    for ep in range(n_epochs):
        epoch_nll = 0.0
        for i in range(n_seq):
            inp = ids[i * (L + 1):i * (L + 1) + L].unsqueeze(0)
            tgt = ids[i * (L + 1) + 1:i * (L + 1) + L + 1].unsqueeze(0)
            logits, _ = model(inp)
            nll = F.cross_entropy(logits.reshape(-1, V + 2), tgt.reshape(-1))
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            epoch_nll += nll.item()
        avg = epoch_nll / max(n_seq, 1)
        print(f"    Stage1F4 Ep{ep+1}/{n_epochs}: nll={avg:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(min(n_seq, 50)):
            inp = ids[i * (L + 1):i * (L + 1) + L].unsqueeze(0)
            tgt = ids[i * (L + 1) + 1:i * (L + 1) + L + 1].unsqueeze(0)
            logits, _ = model(inp)
            losses.append(F.cross_entropy(
                logits.reshape(-1, V + 2), tgt.reshape(-1)).item())
    ppl = math.exp(sum(losses) / max(len(losses), 1))
    print(f"    Stage1F4 Eval PPL: {ppl:.2f}", flush=True)
    model.train()
    return ppl, model


if __name__ == "__main__":
    V = 200
    tasks = build_v32_tasks(V, n_support=40, n_query=20, seed=42)
    train_rules = [n for n in tasks if tasks[n]["seen"]]
    ppl, m = stage1_pretrain_fractal4(tasks, train_rules, V, n_epochs=2,
                                      d0=128, n_layers=4, D=1.5)
    torch.save({k: v.cpu().clone() for k, v in m.state_dict().items()},
               "/tmp/v32_s1_f4_state.pt")
    print(f"cached /tmp/v32_s1_f4_state.pt (widths={m.rln.widths}, ppl={ppl:.2f})")
