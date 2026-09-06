# headless Wiki 日志

>> 本日志从 LMT-twister 主仓库 `docs/wiki/log.md` 精简而来,只保留持续学习
>> (LM 系列 / OML / Replay) 相关条目。完整历史见主仓库。

| 日期 | 事件 | 说明 |
|:-----|:-----|:-----|
| 2026-09-06 | 初始化知识系统 | Bootstrap 创建的持续学习模型日志 |
| 2026-09-06 | **LM2 headless 仓库复跑 — ✅ 复现同源三域正迁移零遗忘** | CPU 约束缩规模 (d96/ds8/l1 0.18M, 120 步/域, eval 100 chunks; 原全量 d192 0.71M/500 步/域 见 08-16)。wikitext-2-raw 同源三域 (A=train 2M/B=valid/C=test) 顺序流式。**遗忘矩阵: A 0.3110→0.3735 (+6.3pp) / B 0.3634→0.3764 (+1.3pp) / C 0.3812 — 三域全正迁移零遗忘, 定性复现 08-16**。复现: `python3 tests/run_lm2_text.py --d-model 96 --d-state 8 --n-layers 1 --steps-per-domain 120` → `results/lm2_report.md` (gitignored) |
| 2026-09-06 | **LM3 headless 仓库复跑 base vs replay — ✅ 复现 Replay 独档防遗忘** | d96/ds8/l1 0.25M, BPE vocab 986 (当前 /tmp/lm3data: en=wikitext 2M / zh=重复句语料 250K 字符 / code=200K — 与 08-16 fineweb-edu/instructions 数据不同, 数字不可直接比), 100 步/域 en→zh→code。**base: en 0.1730→0.0361 (灾难遗忘 -13.7pp)**; **replay: en 0.1730→0.2014 (+2.8pp 零遗忘, 比 base 保留多 +16.5pp)** — 定性一致: 朴素顺序跨域灾难遗忘, replay 数据级锚定有效。zh/code acc ≈1.0 是语料重复性假象 (诚实标注)。**修复: BPE encode 原 `del` 版 O(n²) 卡死 (2M 字符 >7min) → 线性左→右归并 0.4s 且输出逐位等价**; checkpoint/报告文件名加 method 防 base/replay 互相覆盖。复现: `python3 tests/run_lm3_bpe.py --d-model 96 --d-state 8 --n-layers 1 --steps-per-domain 100 --method replay --bpe-cache /tmp/lm3_bpe_cache.json` |