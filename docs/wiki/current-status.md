---
title: Sovereign AI 当前状态
source: session
created: 2026-09-06
last_confirmed: 2026-09-06
audience: reader
stage: draft
tags: [status]
status: current
---

## 最近更新

- 2026-09-06：在 `tests/run_lm1_production.py` 完成 Replay + OML + 因果学习整合
  - 新增 `ReplayBuffer` 抗遗忘机制，支持 `--replay/--no-replay` 和 `--replay-ratio`
  - 集成 `CausalProposer`（因果价值函数：生成元新颖性 + do-干预 + backdoor + 反事实 + 真实反馈）与 `CausalRLAgent`（因果 RL：do-counterfactual credit assignment）
  - 实现评估反馈闭环：每轮对提议任务做轻量评估，将真实泛化 acc 回传给因果提议器
  - CLI 新增 `--proposer {value,causal,causal_rl}` 和 `--causal-rl-lr`
  - Checkpoint 增强：持久化 replay buffer、proposer 状态、replay task names
  - 三种提议器 smoke test 均通过，代码已 push 到 `origin/master`

## 进行中的实验

- 待用户决定：是否跑 short ablation（3 rounds × 20 iters）对比 value/causal/causal_rl + replay 开关的定量效果
- 待修复：replay buffer 对提议任务的持久化回放（当前提议任务名 `__p*` 在评估后清理，样本无法长期回放）