# Sovereign AI Wiki 日志

>> 本日志从 LMT-twister 主仓库 `docs/wiki/log.md` 精简而来,只保留持续学习
>> (LM 系列 / OML / Replay) 相关条目。完整历史见主仓库。

| 日期 | 事件 | 说明 |
|:-----|:-----|:-----|
| 2026-09-06 | 初始化知识系统 | Bootstrap 创建的持续学习模型日志 |
| 2026-09-06 | LM1 整合 Replay + OML + 因果学习 | 在 `tests/run_lm1_production.py` 中完成三大机制整合：1) 新增 `ReplayBuffer` 抗遗忘机制，每轮按比例回放已见任务；2) 集成 `CausalProposer`（因果价值函数：生成元新颖性 + do干预 + backdoor + 反事实 + 真实反馈）与 `CausalRLAgent`（因果RL：do-counterfactual credit assignment）；3) 实现评估反馈闭环：每轮对提议任务做 `_eval_task_acc`，将真实泛化表现回传给因果提议器；4) CLI 新增 `--proposer {value,causal,causal_rl}`、`--replay/--no-replay`、`--replay-ratio`、`--causal-rl-lr`。已验证三种提议器均可运行。 |
| 2026-09-08 | LM1 GPU 训练支持 + 8M 档跑测 | 服务器(2×A100-40GB, NGC 容器)GPU 直通后, 完成 lm1 的 GPU 化改造并在 GPU1 跑 8M 档(9.02M 参数)。系列 commit: `ff070ad`(--scale 档位 0.9M~2B + --threads) → `d595b37`(--device cuda + SwiftTDHead .to()) → `88baf80`(swifttd f_b1/f_b2 硬编码 CPU 修复) → `812aa51`(--tag 隔离 checkpoint/报告) → `124ad82`(migrate_eval GPU 下 extend_head_n 新模块搬移) → `7f55370`(--outer-lr 可配)。**结果**: 降 lr=3e-4 版 10 轮即达 lm1 自动达标线 — unseen 0.1250 / c4 0.1302 / S5迁移 0.123 / loss 收敛 ~0.99 (vs 1e-3 版 R10 发散 loss 2.57)。c4 0.13 远超家族历史锁死带(0.075-0.096)。报告: `results/gpu_lm1/lm1_8m_lr3e4_REPORT.md`。详见 git log + docs/wiki/current-status.md。 |