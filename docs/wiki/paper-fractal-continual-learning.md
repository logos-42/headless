---
title: 分形持续学习论文
source: session
created: 2026-10-10
last_confirmed: 2026-10-10
audience: public
stage: draft
tags: [paper, continual-learning, fractal, structural-growth]
---

# 分形持续学习论文（draft）

**Fractal Continual Learning: Recursive Structural Growth, Long-Horizon Adaptation, and the Boundary of Structural Advantage**

论文位于 `paper/`（`main.tex` / `main.bib` / `main.pdf` + `README.md`），
编译命令 `tectonic main.tex`（已验证 6 页、6 条参考文献、无 undefined cite）。

## 论文主张（三层, 全部实测支持）

1. **递归结构塑性可行** — 密集根环 + 稀疏后裔环（更新频率 / D^depth）+ 凸组合
   门控跨层沟通 + 相对衰减生长判据 + 干净-loss 过滤。树自动长到 7 节点后稳定。
2. **长期持续学习成立** — 72,335 轮非平稳 μ-控制环境: 两种结构均从 ~130-176 步
   持续学到 ~9 步（cold 随机 ~32 步, **3.5× 增益**）, 长程不衰减。
3. **结构优势边界（核心, 诚实）** — 小模型 regime 下分形无统计优势:
   收敛后配对差 < 0.3 步, 越界率 5.9% vs 5.8%, 3-seed 与 701 条真实数据
   5-fold 均不显示优势。**分形优势需要更大规模 + 明显收益递减的 regime。**

## 数据与复现（在 HUSHFUSION Forge 仓, 非本仓）

- 长跑守护进程: `scripts/design_daemon_fractal.py` + `oml_supervisor_fractal.sh`
- 72,335 轮配对趋势: 服务器 `artifacts/oml_daemon_trend_fractal_*.jsonl`（活数据）
- 分析: `scripts/analyze_fractal.py`
- 全部 spike: `scripts/spike_fractal_*.py`, `scripts/spike_ag_*.py`
- 证据链: Forge `docs/wiki/plan-antigravity-control.md` §2e-2k
- 理论基础: lm-principle 仓 `LmPrinciple/Fractal.lean`（分形分配定理）

## 状态

- [x] 初稿（结构与全部实测数字）
- [x] 参考文献（Murray 1926 / Finn 2017 / Li 2017 / Kirkpatrick 2017 / 自仓 ×2）
- [x] tectonic 编译 + PDF 验证（6 页）
- [x] README + 本 wiki 页
- [ ] 中文镜像版（zh mirror, 待 leo 确认是否需要）
- [ ] 对抗审稿轮次（adversary referee, 待 leo 启动）

## 诚实边界

- 控制环境是确定性模型, 不是实验装置; λ 是上游第二输入缺口
- 论文是边界结果（机制可行 + 优势边界测量）, 不主张「分形更优」
- 72,335 轮是该协议下的结论; 更大模型的 regime 分形优势留待后续