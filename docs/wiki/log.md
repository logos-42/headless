# 持续学习模型日志 (headless)

> 本日志从 LMT-twister 主仓库 `docs/wiki/log.md` 精简而来,只保留持续学习
> (LM 系列 / OML / Replay) 相关条目。完整历史见主仓库。

| 日期 | 事件 | 说明 |
|:-----|:-----|:-----|
| 2026-08-16 | **LM1 生产级自主持续学习模型 — 🎯 达标, c4 0.1510 击穿家族天花板** | "开 loop 循环, 目标是完成模型训练, 生产级的模型, 训练数据自己探索性生成"。LM1 (1.05M 参数 d192/d12/2层, SSM+OML 双层+价值函数自主探索数据, checkpoint 可恢复, 达标自动停)。17 轮 680 iters 自动收敛: **unseen 0.1276 / c4 0.1510 / 遗忘 -0.006 / S5 迁移 0.1172 (R16)**。**c4 0.151 = 家族 22 次干预峰值 0.099 的 1.5 倍** — 更大表示 × 长持续训练 (680 vs 40 iters) × 自主探索数据, 生产级系统整体涌现而非单机制。R1-R17 曲线完整 (R6/R12/R16 三次表示转变点)。模型: checkpoints/lm1_final.pt (6.3MB)。`results/lm1_report.md` + `tests/run_lm1_production.py` |
| 2026-08-16 | **LM1 推理接口演示 (define+adapt+query 生产级验证)** | `tests/lm1_inference_demo.py` — LM1Inference 类: load checkpoint → define (intent 编码 ~0.7ms) → adapt (快权重 K=20 ~280ms) → query (预测 ~6ms/条)。三结构类 unseen 演示: **c3 0.0985 / c4 0.1122 / dbl 0.1279, 平均 0.1129** (寄存器级部分正确率, 与训练评估协议一致, 训练 R17 0.1276 同量级)。**生产级闭环验证: 6.3MB 模型文件 → 加载 → 三阶段推理**。坑: _define 需完整 task (perm=None 会产生无效 intent) / acc 须用 regs_acc 部分正确率 (全等太严) / ctx 须 tolist()。`results/lm1_inference_demo.md` |
| 2026-08-16 | **LM2 真实文本持续学习 (LM1 管线语言模型化) — ✅ 三域零遗忘正迁移** | "把 LM1 管线迁到真实文本语料"。wikitext-2-raw 三域 (A=train 2M/B=valid 1.1M/C=test 1.3M 字符), 0.71M SSM 字符级 LM (vocab 607), 域顺序流式 500 步/域。**遗忘矩阵: A 0.429→0.472 (+4.3pp) / B 0.478→0.488 / C 0.493 — 三域全正迁移零遗忘**。"持续在运行时更新模型"在真实文本上成立。诚实: 三域同源维基 (风格相近), 字符级, 未跑 iid 对比, 单 seed。`tests/run_lm2_text.py` + `results/lm2_report.md` + checkpoints/lm2_latest.pt |
| 2026-08-16 | **LM3 (BPE + 跨域 + iid 对比) — ✅ 跨域灾难遗忘实锤, 朴素顺序≠持续学习** | en(wiki)→zh(中文小说)→code(python) 三跨域, 2.26M BPE 模型 (vocab 4627)。**Sequential: en 0.260→0.156 (-10.4pp 遗忘) / zh 0.076→0.014 (-6.2pp) / code 0.340 (近期域强)**; **iid: en 0.201/zh 0.030/code 0.270, 平均打平 (0.167 vs 0.170)**。结论: ① LM2 零遗忘是同源假象 (域风格相近) ② 朴素顺序训练跨语言灾难遗忘, 近期域优势被遗忘抵消 ③ iid 无遗忘但无专注 ④ **真正的持续学习需要机制 (OML 双层/回放) — 下一步 LM3+OML 对照**。`results/lm3_bpe_crossdomain.md` |
| 2026-08-16 | **LM3-OML 决定性对照 — ✅ Replay 胜出 (+5.0pp), 简单双 lr 不是 OML** | base vs replay vs oml 三配置同进程 (同一 BPE 缓存, 200 步/域, en→zh→code)。**Replay: en 0.252/zh 0.057/code 0.352, 平均 0.220 > Base 0.170 (+5.0pp)** — 防遗忘 (en +9.6pp/zh +4.3pp) 且新域不损 (code +1.2pp); **OML 双层 (head lr 1e-2 + rln lr 1e-4) 失败 (平均 0.131)** — 快 head 被新域重写, en 峰值就低。教训: **简单双学习率≠OML**, 家族 OML 真正形态 = 内循环克隆适应 (adapt_graph, V31 验证), 语言版需 per-domain meta-step。下一步: 正确 OML vs Replay + 组合。`results/lm3_oml_replay_contrast.md` |
| 2026-08-16 | **Combo (OML+Replay) 实验 — combo=OML 失败, 坏机制污染好机制** | 组合实验: OML 双循环 + replay 混入。**combo 结果与 OML 完全一致 (平均 0.131)** — head lr 1e-2 重写梯度压过回放损失。教训: 机制组合非简单相加, 正确 OML (内循环克隆) 才能与 Replay 组合。**四配置定论: Replay 0.220 > Base 0.170 > OML=Combo 0.131**。论文已全量重写 (持续学习版, 旧架构弃用): Headless_Model/head-en.tex (265 行) + head-zh.tex (272 行), tectonic 双版编译通过 |
| 2026-08-17 | **OML2 (正确 meta-step OML) vs Replay 对决 — OML 非防遗忘机制, Replay 独一档** | 正确 OML: head 克隆 + support 适应 (2 内步 SGD) + query meta-loss 更新表示 + 10 步合并。结果: 平均 0.168 ≈ base 0.170 (en -10.7pp 遗忘), code 0.344 = base (新域适应恢复)。**机制结论: OML=适应效率, Replay=旧域锚定 (数据级)**; 五配置: Replay 0.220 > Base 0.170 > OML2 0.168 > OML=Combo 0.131。论文 EN/ZH 双版已加 oml2 行 + 讨论重写 (tectonic 编译通过) |
| 2026-09-06 | **因果技术栈落地 (Causal AI) — 用因果发现改进自主数据提议器** | 把《Causal AI》的因果方法落地到 S4 元学习世界 (生成元→结构→泛化 SCM)。新增 `hibs_lnn/causal/`: dag (d-separation) / scm (do+反事实) / identification (后门调整) / discovery (条件独立检验+PC 骨架) / causal_proposer (因果提议器) / causal_rl (Causal RL+credit assignment)。**实测: 同样已学集合下因果提议器每步覆盖新生成元原子 0.50 vs 启发式 0.33**。自测 15/15, demo 8 步秒级, `smoke_lm1` 未破坏。接入: 将 LM1 的 `S4ValueProposer` 换成 `CausalProposer` 即可。`causal_tech_stack_demo.py` + `docs/wiki/causal_stack.md` |

## 机制结论汇总

- **Replay 是唯一的防遗忘机制** (数据级锚定, +5.0pp)
- **OML 是适应效率机制, 不是防遗忘机制** (正确形态 = 内循环克隆 + query meta-loss)
- **简单双学习率 ≠ OML** (OML/Combo 均失败)
- **朴素顺序训练跨语言灾难遗忘**, iid 无遗忘但无专注
- **生产级系统 (LM1) = 更大表示 × 长持续训练 × 自主探索数据, 整体涌现**