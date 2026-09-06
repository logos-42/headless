# Sovereign AI Wiki 日志

>> 本日志从 LMT-twister 主仓库 `docs/wiki/log.md` 精简而来,只保留持续学习
>> (LM 系列 / OML / Replay) 相关条目。完整历史见主仓库。

| 日期 | 事件 | 说明 |
|:-----|:-----|:-----|
| 2026-09-06 | 初始化知识系统 | Bootstrap 创建的持续学习模型日志 |
| 2026-09-06 | LM1 整合 Replay + OML + 因果学习 | 在 `tests/run_lm1_production.py` 中完成三大机制整合：1) 新增 `ReplayBuffer` 抗遗忘机制，每轮按比例回放已见任务；2) 集成 `CausalProposer`（因果价值函数：生成元新颖性 + do干预 + backdoor + 反事实 + 真实反馈）与 `CausalRLAgent`（因果RL：do-counterfactual credit assignment）；3) 实现评估反馈闭环：每轮对提议任务做 `_eval_task_acc`，将真实泛化表现回传给因果提议器；4) CLI 新增 `--proposer {value,causal,causal_rl}`、`--replay/--no-replay`、`--replay-ratio`、`--causal-rl-lr`。已验证三种提议器均可运行。 |