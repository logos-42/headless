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

## 论文主张（审计版, 2026-10-11 第1轮退稿后重写）

**第1轮双审稿人（NeurIPS级 + CNS级）发现致命实现缺陷, 已退稿重写为审计版:**

1. **审计发现（核心）** — 72,335 轮主实验代码里, 分形树的子节点从不被训练:
   children 不在任何优化器参数表, update_freq 从未参与调度, 7 节点是硬 cap 不是自主收敛。
   论文标题改为 "Before Fractal Continual Learning: An Implementation Audit of
   Recursive Structural Growth and Long-Horizon Evaluation"。
2. **仍然成立** — 根节点长程适应真实: warm 130→9 步 (cold 随机 ~32 步);
   相对衰减生长检测器工作: 6 次生长 1→7 节点 (cap 停)。
3. **撤回** — 「分形结构无优势」改为「未定论 (undecidable)」:
   原跑测的是「根门控未训练随机子网」, 不是分形学习器。
4. **修复已提交** — Forge 仓 10ba961: daemon 递归训练子节点 + 稀疏调度生效 + --seed 可复现。

## 论文主张（原稿, 已撤回）

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