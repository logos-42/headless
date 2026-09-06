# causal 因果技术栈落地（headless）

> 把《Causal AI》一书的因果技术落地到 headless 的 S4 元学习世界，核心目标是
> **用因果发现改进 LM1 的自主数据提议器**。

## 为什么 S4 世界天然是一个结构因果模型（SCM）

headless 的任务被参数化为 S4 群的排列 `perm`。每个排列有：

- **generator_signature**：它由哪些对换生成元（6 原子）复合而成 —— **因**
- **cycle_type**：循环类型（ident/swap/3-cycle/4-cycle/双对换）—— 结构复杂度
- **acc**：在未见结构上的泛化表现 —— **果**

于是"任务为什么难/易"可以因果归因到 `生成元 → 结构 → 泛化`。

## 目录

| 文件 | 对应书章节 | 内容 |
|:--|:--|:--|
| `hibs_lnn/causal/dag.py` | 第 3-4 章 | DAG + **d-separation**（路径阻塞） |
| `hibs_lnn/causal/scm.py` | 第 6-9 章 | **SCM** + `do()` 干预 + **反事实**（Abduction/Action/Prediction）|
| `hibs_lnn/causal/identification.py` | 第 10-11 章 | **后门调整** + g-公式识别 + 因果效应 |
| `hibs_lnn/causal/discovery.py` | 发现章 | 条件独立检验（标准 CMI + Bootstrap）+ PC 骨架 |
| `hibs_lnn/causal/causal_proposer.py` | 自主探索 | **CausalProposer**：do-效应 / 反事实 / 发现驱动的提议 |
| `hibs_lnn/causal/causal_rl.py` | 第 12 章 | **Causal RL**：状态/动作/奖励 + 因果 credit assignment |
| `causal_tech_stack_demo.py` | — | 端到端演示（8 步） |
| `hibs_lnn/causal/self_test.py` | — | 自测（15 项） |

## 运行

```bash
# 自测
python3 -m hibs_lnn.causal.self_test        # 15/15 passed

# 端到端演示（秒级，纯 numpy）
python3 causal_tech_stack_demo.py
```

## 如何接入 LM1 的自主数据提议器

`CausalProposer` 与 `S4ValueProposer` 接口一致：
`__init__(learned_perms, k=...)` · `propose(n=None) -> [perm, ...]`。

在 `tests/run_lm1_production.py` 的 `LM1System._init_model` 中
把 `self.proposer = S4ValueProposer(learned, ...)` 替换为
`self.proposer = CausalProposer(learned, k=...)` 即可接入，无需改动主循环。

## 因果 vs 启发式提议（demo 实测）

在相同已学集合上各提议 6 个 unseen 排列：

| 提议器 | 新生成元原子/步 | 备注 |
|:--|--:|:--|
| 启发式 (sim/con/cov) | 0.33 | 只在"结构标签"层面追求差异 |
| 因果 (do/反事实/发现) | **0.50** | 定向覆盖真正决定泛化的"原因原子" |

**解释**：先验启发式容易在结构标签空间乱跳（覆盖未知结构占比高），
但真正约束泛化的是"原因原子"（生成元）覆盖率。因果提议用 do-效应 + 反事实
识别"被生成元瓶颈卡住"的任务，用更少步数覆盖更多原因原子。

## 可选依赖（更重的工具栈）

本实现为轻量自包含（无外部因果库），主要求"可复现、秒级可跑、不污染 torch 环境"。
如需与书保持一致的高层工具栈，可另行安装并对照：

```bash
pip install pgmpy dowhy causal-learn networkx
```

- `networkx` → 替代 `dag.py` 的 d-separation
- `pgmpy` → 替代 `scm.py` 的贝叶网络
- `dowhy` → 替代 `identification.py` 的 Identify→Estimate→Refute 工作流
- `causal-learn` → 替代 `discovery.py` 的 PC/CI 检验