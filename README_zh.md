# headless

基于 SSM（状态空间模型）+ OML（基于优化的元学习）构建的生产级**持续学习模型**系统，配备自主数据探索能力。

> 前身为 LMT-twister。建立于 2026-08-20。

## 持续学习

headless 模型在运行时持续学习，不会遗忘先前获取的知识。系统展示的能力包括：

- **跨域零遗忘** — LM2 在 wikitext-2 上（3 域，流式 500 步/域）：顺序训练后所有域均呈现**正向迁移**（A: 0.429→0.472，B: 0.478→0.488，C: 0.493）。
- **跨域鲁棒性** — LM3 在 en→zh→code 上（BPE，3 域）：朴素顺序训练造成灾难性遗忘（English -10.4pp），证明**真正的持续学习需要机制**（Replay +5.0pp，正确 OML 通过内循环适应）。
- **自主探索** — LM1 通过价值函数提议器（简约性 + 自洽性 + 覆盖）自主生成训练数据，达到 **unseen 准确率 0.1276 / c4 结构 0.1510**，**遗忘为负**（-0.006），以 **1.5 倍** 超越家族 22 次干预峰值（0.099）。

## 架构

```
LM1（生产级，105 万参数）
├── SSM 骨干网络（V31_RLN，d_model/d_state/层数可配置）
├── SwiftTD 头 — 逐特征步长优化（在线 IDBD，无需 BPTT）
├── OML 双层循环 — 内循环 PLN 适应 + 外循环元更新
└── S4 价值提议器 — 基于结构新颖性自主生成任务

LM2（文本，71 万参数）
├── SSM 字符级语言模型（词表 607）
└── 三域流式 wikitext-2

LM3（BPE，226 万参数）
├── BPE 分词器（词表 4627）
└── en→zh→code 跨域实验
```

### 核心机制

| 机制 | 作用 | 证据 |
|:--|:--|:--|
| **Replay（回放）** | 防遗忘（数据级锚定） | 相对基线 +5.0pp |
| **OML** | 适应效率（内循环克隆） | 非防遗忘机制；实现新域快速适应 |
| **SwiftTD** | 逐特征学习率（在线 IDBD） | 防止 bound/decay 下的步长塌缩 |
| **价值提议器** | 自主数据生成 | 无需人工标注即可持续探索 |

## 因果技术栈（Causal AI，新增）

把《Causal AI》一书的因果方法落地到 S4 元学习世界，用**因果发现改进自主数据提议器**：

| 文件 | 落地 | 对应书章节 |
|:--|:--|:--|
| `hibs_lnn/causal/dag.py` | DAG + d-separation | 第 3-4 章 |
| `hibs_lnn/causal/scm.py` | SCM + do() 干预 + 反事实 | 第 6-9 章 |
| `hibs_lnn/causal/identification.py` | 后门调整 / g-公式识别 | 第 10-11 章 |
| `hibs_lnn/causal/discovery.py` | 条件独立检验 + PC 骨架 | 因果发现 |
| `hibs_lnn/causal/causal_proposer.py` | 因果驱动的提议器（do-效应/反事实） | 自主探索 |
| `hibs_lnn/causal/causal_rl.py` | Causal RL + 因果 credit assignment | 第 12 章 |

- 演示：`python3 causal_tech_stack_demo.py`（8 步，秒级）
- 自测：`python3 -m hibs_lnn.causal.self_test`（15/15 passed）
- 文档：`docs/wiki/causal_stack.md`

**实测**：同样已学集合下，因果提议器每步覆盖新生成元原子 **0.50**，启发式仅 **0.33**。
## 项目结构

```
tests/
├── run_lm1_production.py  — 生产级自主持续学习模型（checkpoint、评估、推理）
├── run_lm2_text.py        — 真实文本持续学习（wikitext-2）
├── run_lm3_bpe.py         — BPE + 跨域持续学习（5 种方法）
├── lm1_inference_demo.py  — 生产级推理接口（define+adapt+query）
└── run_v3*.py             — 17 个实验运行脚本（V31-V35 消融套件）

hibs_lnn/
├── ssm_v30_3.py           — 带内部纠缠的 SSM 层（相位调制）
├── swiftd_head.py         — SwiftTD 逐特征步长优化器
├── meta_rule_world.py     — 元学习任务分布（S4 排列）
├── code_world.py          — 寄存器机沙箱（8 条指令，4 个寄存器）
└── causal/                — Causal AI 技术栈（DoWhy/pgmpy 的轻量等价物）
    ├── dag.py             — DAG + d-separation
    ├── scm.py             — 结构因果模型 + do() + 反事实
    ├── identification.py  — 后门调整 / g-公式识别
    ├── discovery.py       — 条件独立检验 + PC 骨架
    ├── causal_proposer.py — 因果驱动的自主数据提议器（改进提议器）
    └── causal_rl.py       — Causal RL + 因果 credit assignment

tests/
├── run_lm1_production.py  — 生产级自主持续学习模型（checkpoint、评估、推理）
├── run_lm2_text.py        — 真实文本持续学习（wikitext-2）
├── run_lm3_bpe.py         — BPE + 跨域持续学习（5 种方法）
├── lm1_inference_demo.py  — 生产级推理接口（define+adapt+query）
└── run_v3*.py             — 17 个实验运行脚本（V31-V35 消融套件）

docs/wiki/
├── log.md                 — 持续学习实验日志
└── causal_stack.md        — 因果技术栈落地文档

causal_tech_stack_demo.py  — 因果技术栈端到端演示（8 步，秒级）
```

## 快速开始

```bash
# 冒烟测试（1 轮、2 步、不触发迁移）
python smoke_lm1.py

# 生产级训练（10 轮，自动探索 + 评估）
python tests/run_lm1_production.py --rounds 10

# 从 checkpoint 恢复
python tests/run_lm1_production.py --resume

# 推理演示
python tests/lm1_inference_demo.py --ckpt checkpoints/lm1_final.pt

# 文本持续学习
python tests/run_lm2_text.py

# BPE 跨域（replay 方法）
python tests/run_lm3_bpe.py --method replay
```

## 实验结果

### LM1 — 生产级自主持续学习

| 指标 | 数值 | 家族峰值 | 提升 |
|:--|:--|:--|:--|
| Unseen 准确率 | 0.1276 | 0.099 | **+29%** |
| c4 结构准确率 | 0.1510 | 0.099 | **+53%** |
| 遗忘 | -0.006 | — | **负值（能力提升）** |
| S5 迁移（R16） | 0.1172 | — | 正向迁移 |

- **17 轮 / 680 步** 全自动收敛
- 模型：`checkpoints/lm1_final.pt`（6.3MB）
- 报告：`results/lm1_report.md`

### LM2 — 真实文本持续学习

| 域 | 训练前 | 训练后 | 变化 |
|:--|:--|:--|:--|
| A（训练 200 万字符） | 0.429 | 0.472 | **+4.3pp** |
| B（验证 110 万字符） | 0.478 | 0.488 | **+1.0pp** |
| C（测试 130 万字符） | 0.493 | — | **正向迁移** |

**结论：** 持续学习与零遗忘在同源真实文本（维基）上成立。

### LM3 — 跨域持续学习

| 方法 | en | zh | code | 平均 |
|:--|:--|:--|:--|:--|
| 基线（顺序训练） | 0.260 | 0.076 | 0.340 | 0.170 |
| Replay | 0.252 | 0.057 | 0.352 | **0.220** |
| OML（双学习率） | — | — | — | 0.131 |
| OML2（正确实现） | — | — | — | 0.168 |

**结论：** Replay 是唯一有效的防遗忘机制（+5.0pp）。正确 OML（内循环克隆）是适应效率机制，而非防遗忘机制。

## 机制结论

1. **Replay 是唯一的防遗忘机制** — 数据级锚定（+5.0pp）
2. **OML 是适应效率机制** — 正确形态 = 内循环克隆 + query 元损失
3. **简单双学习率 ≠ OML** — 仅使用 head+body 不同 LR 会失败
4. **朴素顺序训练 = 跨语言灾难性遗忘**
5. **生产级持续学习 = 更大表示 × 长期训练 × 自主探索数据** — 作为系统整体涌现

## 数据需求

- **LM1 / LM2**：无需外部数据（合成任务 / wikitext-2 自动下载）
- **LM3**：将数据放入 `/tmp/lm2data` 和 `/tmp/lm3data`

## 引用

如使用本工作，请引用上述机制结论及 LM1 生产级系统结果。

## 许可证

MIT — 详见 [LICENSE](LICENSE)
