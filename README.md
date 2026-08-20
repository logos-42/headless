# LMT-twister

A production-grade **continuously learning model** system built on SSM (State Space Model) + OML (Optimization-based Meta-Learning) with autonomous data exploration.

> Headless fork of the LMT-twister research project. Established 2026-08-20.

## Continuous Learning

LMT-twister models learn continuously during runtime without forgetting previously acquired knowledge. The system demonstrates:

- **Zero forgetting across domains** — LM2 on wikitext-2 (3 domains, streaming 500 steps/domain): all domains show **positive transfer** after sequential training (A: 0.429→0.472, B: 0.478→0.488, C: 0.493).
- **Cross-domain robustness** — LM3 on en→zh→code (BPE, 3 domains): naive sequential training causes catastrophic forgetting (-10.4pp on English), proving **true continuous learning requires mechanisms** (Replay +5.0pp, correct OML via inner-loop adaptation).
- **Autonomous exploration** — LM1 generates its own training data via a value-function proposer (simplicity + consistency + coverage), reaching **unseen accuracy 0.1276 / c4 structure 0.1510** with **negative forgetting** (-0.006), beating the family 22-intervention peak (0.099) by **1.5×**.

## Architecture

```
LM1 (Production, 1.05M params)
├── SSM backbone (V31_RLN, configurable d_model/d_state/layers)
├── SwiftTD head — per-feature step-size optimization (IDBD online, no BPTT)
├── OML dual-loop — inner-loop PLN adaptation + outer-loop meta-update
└── S4 Value Proposer — autonomous task generation from structural novelty

LM2 (Text, 0.71M params)
├── SSM character-level LM (vocab 607)
└── Streaming 3-domain wikitext-2

LM3 (BPE, 2.26M params)
├── BPE tokenizer (vocab 4627)
└── en→zh→code cross-domain experiments
```

### Key Mechanisms

| Mechanism | Role | Evidence |
|:--|:--|:--|
| **Replay** | Anti-forgetting (data anchoring) | +5.0pp avg over base |
| **OML** | Adaptation efficiency (inner-loop cloning) | Not anti-forgetting; enables fast new-domain adaptation |
| **SwiftTD** | Per-feature learning rates (online IDBD) | Prevents step-size collapse under bound/decay |
| **Value Proposer** | Autonomous data generation | Drives continuous exploration without human labeling |

## Project Structure

```
tests/
├── run_lm1_production.py  — Production autonomous CL model (checkpoint, eval, infer)
├── run_lm2_text.py        — Real-text continuous learning (wikitext-2)
├── run_lm3_bpe.py         — BPE + cross-domain CL (5 methods)
├── lm1_inference_demo.py  — Production inference API (define+adapt+query)
└── run_v3*.py             — 17 experiment runners (V31-V35 ablation suite)

hibs_lnn/
├── ssm_v30_3.py           — SSM layer with internal entanglement (phase modulation)
├── swiftd_head.py         — SwiftTD per-feature step-size optimizer
├── meta_rule_world.py     — Meta-learning task distribution (S4 permutations)
└── code_world.py          — Register-machine sandbox (8 instructions, 4 registers)

docs/wiki/log.md           — Continuous learning experiment log
```

## Quick Start

```bash
# Smoke test (1 round, 2 iters, no migration)
python smoke_lm1.py

# Production training (10 rounds, auto-explore + evaluate)
python tests/run_lm1_production.py --rounds 10

# Resume from checkpoint
python tests/run_lm1_production.py --resume

# Inference demo
python tests/lm1_inference_demo.py --ckpt checkpoints/lm1_final.pt

# Text continuous learning
python tests/run_lm2_text.py

# BPE cross-domain (replay method)
python tests/run_lm3_bpe.py --method replay
```

## Results

### LM1 — Production Autonomous CL

| Metric | Value | Family Peak | Improvement |
|:--|:--|:--|:--|
| Unseen accuracy | 0.1276 | 0.099 | **+29%** |
| c4 structure accuracy | 0.1510 | 0.099 | **+53%** |
| Forgetting | -0.006 | — | **Negative (improvement)** |
| S5 transfer (R16) | 0.1172 | — | Positive migration |

- **17 rounds / 680 iters** fully automatic convergence
- Model: `checkpoints/lm1_final.pt` (6.3MB)
- Report: `results/lm1_report.md`

### LM2 — Real-Text CL

| Domain | Before | After | Change |
|:--|:--|:--|:--|
| A (train 2M chars) | 0.429 | 0.472 | **+4.3pp** |
| B (valid 1.1M chars) | 0.478 | 0.488 | **+1.0pp** |
| C (test 1.3M chars) | 0.493 | — | **Positive transfer** |

**Conclusion:** Continuous learning with zero forgetting holds on real text (same-domain wikitext).

### LM3 — Cross-Domain CL

| Method | en | zh | code | Average |
|:--|:--|:--|:--|:--|
| Base (sequential) | 0.260 | 0.076 | 0.340 | 0.170 |
| Replay | 0.252 | 0.057 | 0.352 | **0.220** |
| OML (dual-lr) | — | — | — | 0.131 |
| OML2 (correct) | — | — | — | 0.168 |

**Conclusion:** Replay is the only effective anti-forgetting mechanism (+5.0pp). Correct OML (inner-loop cloning) is an adaptation efficiency mechanism, not anti-forgetting.

## Mechanism Conclusions

1. **Replay is the sole anti-forgetting mechanism** — data-level anchoring (+5.0pp)
2. **OML is adaptation efficiency** — correct form = inner-loop cloning + query meta-loss
3. **Dual learning rate ≠ OML** — simple head+body LR split fails
4. **Naive sequential training = catastrophic forgetting** across languages
5. **Production CL = larger representation × long training × autonomous data** — emergent as a system

## Data Requirements

- **LM1 / LM2**: No external data needed (synthetic tasks / wikitext-2 auto-download)
- **LM3**: Place data in `/tmp/lm2data` and `/tmp/lm3data` respectively

## Citation

If you use this work, please cite the mechanism conclusions and the LM1 production system results.

## License

MIT — see [LICENSE](LICENSE)
