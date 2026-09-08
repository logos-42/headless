---
title: Sovereign AI 当前状态
source: session
created: 2026-09-06
last_confirmed: 2026-09-08
audience: reader
stage: draft
tags: [status]
status: current
---

## 最近更新

- 2026-09-08：LM1 GPU 训练支持 + 8M 档首次跑测（服务器 2×A100-40GB）
  - 完成 lm1 的 GPU 化改造（6 个 commit，见 git log）：`--scale` 规模档位（0.9M~2B）、`--device cuda`、SwiftTDHead `.to()`、swifttd f_b1/f_b2 硬编码 CPU 修复、`--tag` 隔离、migrate_eval GPU 搬移、`--outer-lr`
  - **关键发现 1（线程）**：torch 2.13 CPU 小算子任务 48 线程慢 750x（4-8 线程最优）
  - **关键发现 2（lr）**：8M 档用默认 lr=1e-3 震荡发散（R10 loss 2.57）；降 lr=3e-4 后收敛（R10 loss 0.99）
  - **结果**：8M 档（9.02M 参数）降 lr 版 10 轮即达自动达标线 — unseen 0.1250 / c4 0.1302 / S5迁移 0.123 / 遗忘 ±0.02 区间。c4 0.13 远超家族历史锁死带（0.075-0.096）
  - 对比原始 LM1（0.9M，17 轮 unseen 0.1276）：8M 版 10 轮 unseen 0.1250，接近但略低；c4 0.130 vs 原始 0.151
  - 报告：`results/gpu_lm1/lm1_8m_lr3e4_REPORT.md`；checkpoint：`checkpoints/lm1_8m_round9.pt`
- 2026-09-06：在 `tests/run_lm1_production.py` 完成 Replay + OML + 因果学习整合
  - 新增 `ReplayBuffer` 抗遗忘机制，支持 `--replay/--no-replay` 和 `--replay-ratio`
  - 集成 `CausalProposer`（因果价值函数：生成元新颖性 + do-干预 + backdoor + 反事实 + 真实反馈）与 `CausalRLAgent`（因果 RL：do-counterfactual credit assignment）
  - 实现评估反馈闭环：每轮对提议任务做轻量评估，将真实泛化 acc 回传给因果提议器
  - CLI 新增 `--proposer {value,causal,causal_rl}` 和 `--causal-rl-lr`
  - Checkpoint 增强：持久化 replay buffer、proposer 状态、replay task names
  - 三种提议器 smoke test 均通过，代码已 push 到 `origin/master`

## 进行中的实验

- 待办：24M 档（d768/s24/l6, 27M 参数）在 GPU1（31GB 空闲）BPTT OOM —— 需等 GPU0 邻居释放、释放 ollama 显存、或换 PYTORCH_CUDA_ALLOC_CONF / 减 batch
- 待办：8M 档续跑到 50 轮确认达标连续性（当前仅 10 轮单轮满足自动达标线，checkpoint 在 round9）
- 待用户决定：是否跑 short ablation（3 rounds × 20 iters）对比 value/causal/causal_rl + replay 开关的定量效果
- 待修复：replay buffer 对提议任务的持久化回放（当前提议任务名 `__p*` 在评估后清理，样本无法长期回放）