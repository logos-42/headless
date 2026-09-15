# lm4 benchmark 的时间结构测量 —— 为什么 E9/E10 无法检验 Options

**结论(用户 2026-09-14 的判断被证实)**:lm4 里**不存在可供 option 抽象的时间结构**。
因此「Options 有害」这个结论**不能**用来否定 Options —— 它测的是 **benchmark 的性质**。

---

## 1. 用户设的硬门槛(我接受)

> **不能因为当前 Option 实验负收益,就得出"Option 不适合持续学习"的结论。**
> 目前最多只能说:我们当前实现的 Option 机制,在当前样本量、benchmark、
> 执行方式和任务上**没有表现出收益**。

要区分 5 个可能性:

| 假设 | 状态 |
|:--|:--|
| H1 Option 本身不适合持续学习 | ❌ 不能排除 |
| H2 样本量不足,没学出来 | ❌ 未验证 |
| H3 **benchmark 不适合体现长期收益** | ✅ **本轮证实** |
| H4 当前执行方式有问题 | ⚠️ D1 部分支持 |
| H5 只在非平稳/长期任务中才体现优势 | ❌ 未验证 |

## 2. 我实现里的错误(用户指出,我认)

用户的原文:

> "Option = [a1,a2,...,ak] 其实**不是一个 adaptive 的 Option**,它更像
> '我发现了一段过去有效的动作脚本,现在把这段脚本重新播放'。"
> "**抽象的是目标,不是动作。**"

我的原实现:
```
Option = (I_o, [a_1..a_k], β_o)      <- 固定动作序列
执行   = a_1 → a_2 → ... → a_k        <- 状态变化不触发动作重算
```
用户指出的正确形式:
```
Option = (I_o, g_o, π_o, β_o)         <- 目标是抽象出来的东西
执行   = a_t = π_o(s_t)               <- **每一步**按当前状态重算
```
且第一版**不要给 option 自己的新 policy**,直接用 `a_t = π_base(s_t, g_o)`。

**这正是 D1「伤害随使用量放大 (r=−0.80)」的机制解释。**

## 3. 真正的实验:先问 lm4 有没有可被抽象的结构

在拿负面结论说服任何人之前,先排除最致命的一种可能:

> 若动作之间**可交换**,则「先 a 后 b」≈「a 的增益 + b 的增益」
> → **不存在多步规划空间** → option 无论怎么实现都不可能有用。

### 测量方法

在与 E9/E10 **同构**的模型上 (`StatMLP`, `norm=fixed`, 同一数据与细区间划分):

```
I(a,b) = acc_b(先训a再训b) − acc_b(只训b)
A(a,b) = I(a,b) − I(b,a)          <- 只有不对称性才需要多步规划
```

**只看 `I ≠ 0` 不够** —— 普通迁移不需要 option,只要一个正确的课程。
**只有 `A ≠ 0` 才说明"顺序有意义"。**

### 结果(n=12 对 × 3 seed = 36 次测量,`results/temporal_structure.json`)

| 量 | 均值 | 说明 |
|:--|:--|:--|
| b 零训练 acc | 0.2017 | 随机 = 0.1667 |
| **只训 b 后 acc** | **1.0000** | 区间**平凡可分** |
| **纯迁移(只训 a 测 b)** | **0.0000** | 训练 a 让 b **完全不可预测** |
| 迁移增益 Δ | −0.2017 ± 0.3556 | 负 = 干扰 |
| **不对称 A** | **0.0000 ± 0.0000** | bootstrap 95% CI = **[0, 0]** |
| \|A\| > 0.05 的比例 | **0.0%** | — |

### 三层含义

1. **子任务平凡**:单个细区间训 20 步就到 100%(密度分位数本来就在特征里)
2. **唯一动力学是纯干扰**:训练 a 让 b 的准确率从 0.20 变成 **0.0000**
   —— 不是降到随机水平,而是模型**自信地答错**
3. **干扰完全对称**:`A ≡ 0` 精确为零 → **顺序完全不重要**

**→ lm4 的动作完全可交换:没有组合、没有前置、没有顺序依赖。
不存在任何可供 option 抽象的时间结构。**

## 4. 这对我们既有结论的影响

| 结论 | 是否需要修正 |
|:--|:--|
| **E9/E10:Options 显著有害** | ✅ **必须重新表述** —— 它测的是 benchmark 的性质 |
| D1:伤害随使用量放大 (r=−0.80) | 🟡 仍有效,但**归因要改**:是 open-loop 执行 + 无可抽象结构 |
| D2:模型新鲜度是次级因素 | ✅ 仍有效 |
| 步长 E 表:四算法无差别 | ✅ 不受影响(与 option 无关) |

**正确的表述(用户建议的措辞)**:

> "Open-loop option execution degraded performance as option usage increased,
> while stale-model refresh produced a secondary but consistent effect.
> This suggests that the primary limitation was not option discovery itself,
> but the loss of closed-loop adaptation during option execution."

## 5. 下一步:按用户给的矩阵建**能回答这个问题**的 benchmark

### K1 矩阵(已在跑,`tests/oak_k1.sh`)

唯一变量 = Option 的**执行方式**:

| 臂 | `--opt-mode` | 机制 |
|:--|:--|:--|
| **E9** | (无 option) | 基线 |
| **O1** | `fixed` | 固定动作序列 = **open-loop**(D1 的病灶) |
| **O2** | `goal` | 目标条件 + **每步重算** `a_t = π_o(s_t)` |
| **O3** | `goal_term` | O2 + 自适应终止 `β_o(s)`(目标达成 / 风险 / 停滞) |
| **O4** | `goal_term_override` | O3 + **不确定性抢占**(option 从属于当前证据) |

**判据(用户给的)**:若 `O2 > O1` → 问题不是 temporal abstraction,
而是 **open-loop 执行**;若 `O3/O4` 继续提升 → option 的价值需要闭环执行。

**四档机制的单元级验证(已通过)**:
```
mode                 option步  重算步  启动   抢占  终止原因
fixed                16       0      15     0     {}                              <- open-loop ✓
goal                 105      105    1      0     {}                              <- 每步重算 ✓
goal_term            102      102    4      0     {'goal_reached': 3}             <- 自适应终止 ✓
goal_term_override   34       34     5      3     {'goal_reached': 2, 'override': 3} <- 抢占 ✓
```

### B0–B9 证伪矩阵(待建)

| 实验 | Option | 环境 | 关键测量 |
|:--|:--|:--|:--|
| B0/B1 | ❌/✅ | stationary | performance |
| **B2/B3** | ❌/✅ | **A→B** | **T_adapt**(恢复性能需要多少样本) |
| **B4/B5** | ❌/✅ | **A→B→C** | 反复适应 |
| **B6/B7** | ❌/✅ | **A→B→A** | **知识复用**(T_A^(3) < T_A^(2) < T_A^(1)) |
| B8/B9 | ❌/✅ | 多 regime | **sample efficiency** R(N) |

外加:duration ablation (1/2/4/8/16)、usage ablation、model freshness。

**关键点**:这些 benchmark 必须**本身具有时序结构**(用户指出:option 的价值
更可能是"学会了一个**可复用的行为结构**,环境变化后仍可快速重组和调用",
而不是"这一小段 trajectory 比 primitive action 做得好")。

## 6. 放弃 Options 的硬门槛(用户设定,我接受)

在下列**全部**做完之前,不允许得出「Option 对持续学习没有价值」:

1. 足够大的样本量 2. 多随机 seed 3. stationary benchmark
4. regime-shift benchmark 5. repeated-regime benchmark 6. sample-efficiency benchmark
7. transfer/reuse benchmark 8. **closed-loop Option** 9. **adaptive termination**
10. **uncertainty interrupt** 11. 与 primitive policy 的**公平计算预算比较**
12. 统计显著性 13. confidence interval 14. discovery 与 execution 的分离 ablation

且必须在**全部**指标(最终性能 / adaptation speed / sample efficiency /
transfer / reuse / forgetting / stability)上**都没有优势**。

**→ 主线保持:`Experience → Prediction → Knowledge → Option → Policy → Action`**
