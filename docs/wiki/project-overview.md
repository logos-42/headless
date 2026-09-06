---
title: Sovereign AI 项目概览
source: session
created: 2026-09-06
last_confirmed: 2026-09-06
audience: reader
stage: draft
tags: [overview]
status: draft
---

这里写项目的一句话定义、主线目标、交付边界。

## 一句话定义

headless 是一个基于 SSM + OML 的**生产级持续学习模型系统**，模型在运行中持续学习新任务而不遗忘旧知识，并通过因果驱动自主探索生成训练数据。

## 主线目标

1. **持续学习不遗忘**：通过 Replay buffer 和 OML 内循环克隆实现零/负遗忘
2. **自主探索数据**：模型自己生成训练数据，无需人工标注
3. **因果驱动提议**：用因果推理（do-干预、反事实、backdoor 调整）优化探索策略，比启发式价值函数更精准地定位模型盲区

## 交付边界

- `tests/run_lm1_production.py`：LM1 生产级系统，支持三种提议器（value/causal/causal_rl）和 replay 开关
- `hibs_lnn/causal/`：因果技术栈（SCM、DAG、因果发现、因果提议器、因果 RL）
- `tests/run_v31_meta_learning.py` 等：17+ 个实验运行脚本，覆盖 OML 消融、跨域持续学习、BPE 等
- `docs/wiki/`：项目文档、实验日志、当前状态

## 当前系统架构

```
LM1System
├── SSM backbone (V31_RLN, d_model/d_state/layers 可配)
├── OML dual-loop（内循环 PLN 适应 + 外循环合并）
├── ReplayBuffer（抗遗忘：已见任务样本循环回放）
├── Proposer（三选一）
│   ├── S4ValueProposer：简约 + 自洽 + 覆盖启发式
│   ├── CausalProposer：因果价值函数（5 组件）
│   └── CausalRLAgent：因果 RL + credit assignment
└── 评估反馈闭环：提议任务 → 轻量评估 → 真实 acc 回传 proposer
```