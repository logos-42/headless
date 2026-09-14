# OaK 路线与我们项目的对齐与差距

**日期**: 2026-09-14 · **状态**: 设计文档 (实施中)

来源材料 (本轮读的一手文献):
- Sutton et al. — **OaK Architecture** 三大特征: ①所有组件持续学习 ②每个 weight 有独立
  步长、由 online cross-validation 元学 ③state/time 抽象按五步递进
- Degris, Javed, Sharifnassab, Liu, **Sutton** (2024) *Step-size Optimization for
  Continual Learning* — arXiv 2401.17401
- Mahmood, Sutton, Degris, Pilarski (2012) *Tuning-free Step-size Adaptation* — Autostep
- White, Modayil, Sutton (2012) *Scaling Life-long Off-policy Learning* — Horde/GVF
- Degrave et al. (2022) DeepMind **TCV** 磁控制; Tracey et al. (2023) arXiv 2307.11546
  *Towards practical RL for tokamak magnetic control* (65% shape / **3× 新任务学习时间**)
- Wu et al. (2024) arXiv 2409.09238 **HL-3** *High-Fidelity Data-Driven Dynamics Model*
  (EFITNN; 400ms/1kHz; 700kA 外推 zero-shot 差 152.6kA, +2 shots 降 MAE 90%)

---

## 一、OaK 第③条的四个缺失件 —— 现在填了哪些

| OaK 组件 | 状态 | 实现位置 |
|:--|:--|:--|
| **Action → Env → Reward → 参数更新** | ✅ 已建 | `hibs_lnn/rl_proposer.py` `RLProposer` |
| **步长为主导的 policy 积累** (IDBD) | ✅ 已建 + 已修 | 同上, `algo="idbd"` |
| **免调参步长** (Autostep) | ✅ 已建 | 同上, `algo="autostep"` (默认) |
| **探索 policy 跨 redefine 积累** | ✅ 已建 | `RLProposer.redefine()` 按描述子继承 |
| **Transition model** | ✅ 已建 | `hibs_lnn/transition_model.py` |
| **Planning** | ⚠️ 已建但**不可信** | `TransitionModel.plan()` — 见第四节 |
| **Options / 时间抽象** | ❌ 空白 | — |
| **GVF / Horde (多预测并行)** | ❌ 空白 | — |

---

## 二、实测:IDBD 与 Autostep 的免调参性 (weight-flipping, 论文原始基准)

20 维输入, 前 15 维目标恒 0、后 5 维每 20 步翻转 ±1。合格算法须自发给翻转权重更大步长。

| (μ, α₀) | IDBD 分化比 | Autostep 分化比 |
|:--|--:|--:|
| (0.001, 0.1) | 4.37 | 1.58 ✗ |
| (0.01, 0.1) | 22.84 | **39.99** |
| (0.01, 1e-3) | 164.56 | **16.63** |
| (0.01, 0.01) | 32.27 | **38.79** |
| (0.1, 0.1) | 2.65 (MSE **1.6e9 发散**) | 5.4e10 |
| (0.001, 1e-3) | 2.06 ✗ | 1.19 ✗ |
| (0.05, 0.01) | 0.39 (MSE **1.6e9 发散**) | 3.9e6 |
| (0.5, 0.1) | 0.11 (MSE **1.6e9 发散**) | 3.4e7 |

**结论**: IDBD **3/8 档直接发散** (MSE 1.6e9), 分化成功 4/8;
Autostep **0 档发散** (MSE 全在 2.1~4.9), 分化成功 6/8。
**与论文所述完全一致**: *"IDBD 的好区间窄, 且紧邻发散点"* / *"Autostep 过冲不可能"*。

### ★ 为什么 IDBD 在我们的任务上需要 α₀=0.2 才活

Autostep 论文的单位分析给出精确解释: 指数项 `δ·x·h` 的量纲是 **y²**,
故最优 μ 的量纲是 **1/y²** —— **目标方差一变, 最优 μ 就跨数量级漂移**。

我们的 reward = Δ(any-time 准确率) ≈ 0.02 → y² 极小 → 所需 μ 比 O(1) 目标大 ~2500×。
这解释了实测现象: `α₀=0.01/0.05` 时 `alpha_std ≈ 1e-6` (**步长死亡**),
`α₀=0.2` 才分化。**这是同一现象在两个场景的独立复现** (V35.2/3 的 `β_std=1.6e-05`)。

### 修过的 bug (自查 + 自身测试抓出)

| bug | 症状 | 修法 |
|:--|:--|:--|
| IDBD 步长更新式**错删了 x** | 曾误以为官方是 `exp(μδh)`; 论文式(3) 是 `exp(μδxh)` | 恢复 `phi` |
| 指数 runaway | `alpha.mean=0.886, max=1.0` 撞 clamp | 换 Autostep 归一化 + 指数截断 |
| `--rl-algo/mu/alpha0` **未接调用点** | 四个臂全报 `algo=autostep`、不同超参给出**逐位相同**结果 | 调用点补 3 个参数 |
| JSON 结果**非原子写** + 无 numpy 兜底 | `TypeError: int64 not JSON serializable`, 留下**损坏半截文件** (random-matched 三 seed 全坏) | 兜底序列化器 (`tolist` 须在 `item` **之前**) + 临时文件 `os.replace` |

---

## 三、Q15 的研究设计 vs 我们的实际条件

| Q15 的要求 | 我们 |
|:--|:--|
| `δ^pred = x_{t+1} − x̂_{t+1}` 学环境 | ✅ `TransitionModel` 就是学的动力学; 且我们有**真实跨年漂移** (2015→2016) |
| `δ^RL = r_t + γV(s_{t+1}) − V(s_t)` 学价值 | ✅ `RLProposer` 的 `δ = r − V` (bandit 形式, γ=1) |
| **两个信号不要混成一个** | ⚠️ **我们现在混着** — `RLProposer` 的 reward 用 Δ(any-time), 没有独立的预测误差通路 |
| **必须主动制造非平稳** | ✅ 有 (`--stream nonstationary` + 跨年数据) |
| 对照组 A–E 阶梯 | ⚠️ 有部分, 但**没有这条阶梯** |
| 成功标准含 **Adaptation Time** | ✅ 有 `efficiency` / 达标步数 |
| 成功标准含 **Catastrophic Forgetting** | ✅ 有 最差遗忘界 |

### ★ 必须直说的差距:**我们没有 actuator**

Q15 设 `u_t = V_coil` —— 线圈电压。**我们的数据是 RBSP-A 卫星对磁层的被动观测,
没有任何可施加的控制量。** lm4 的 `do()` 是对**数据**做因果干预, 不是对**系统**施加动作。

所以:
- **做不了** "磁通控制" 的闭环 (需要 FGE/RZIP 级别的等离子体仿真器, 我们没有任何一个)
- **能做** Q15 明确建议的**第一阶段**: `x̂_{t+1} = f_w(x_t, u_t)` 的**预测性知识**
- 我们的 `u_t` 只能取"已知外部条件"(太阳风驱动 / 磁地方时 / L-shell) ——
  **我们改不了它, 但可以条件化它**, 并问反事实 (这正是 `causal_intervene.py` 做的)

**这不是退而求其次, 而是 Q15 自己排的顺序: "第一阶段建议: plasma dynamics /
predictive model, 而不是一开始就让 policy 大幅变化。**"

---

## 四、Planning 不可信 —— Q8 的理论判决 (附实证)

`TransitionModel.plan()` 在 `stream=fixed` 数据上给出:

```
fixed 顺序 (0,1,2,3,4,5 循环)   预测末端准确率 0.5802
planned (beam search)           预测末端准确率 0.9582   ← 序列 = [2,2,2,2,2,2,2,2,2,2,0]
random (20 次平均)              预测末端准确率 0.4279
```

**planned 序列是连训 11 次同一个域** —— 而训练数据里每个域最多出现 2 次,
这个 state-action 对**零覆盖**。

**Q8 给出理论判决**: *"Off-policy ≠ 无数据也能预测 … 目标策略必须在现有经验中得到
足够的信息支持, 否则: 没有数据覆盖 → 预测不可靠。"*

Q1 结果 (转移可学): copy 0.3348 / constant 0.3468 / ridge **0.2755** (打赢 copy −17.7%)。
**转移模型本身可学, 但规划器在无覆盖区域的外推是幻觉。** 不再执行那个序列。

---

## 五、下一步 (按 Q15 的顺序)

1. **分离两条学习信号** — `RLProposer` 拆出独立的 `δ^pred` 通路 (预测转移)
   与 `δ^RL` 通路 (价值), 不再混用 Δ(any-time)
2. **A–E 阶梯** (在无 actuator 前提下改写成):
   - A 固定转移模型 + 固定调度 (baseline)
   - B 在线学转移模型 + 固定调度
   - C 固定转移模型 + 在线学调度 (`RLProposer`)
   - D 在线转移模型 + 在线调度
   - E D + IDBD/Autostep 步长
   → 才能分清 "持续学习起作用" vs "单纯 RL 起作用"
3. **头号指标改为 Adaptation Time** (环境变化后恢复性能需要多久) —— 这是 Q15 点名的
   最关键指标, 也正是我们"达标步数"该被提升为头条的地方
4. 前述全部: 多 seed (可靠标准 std < mean/2)

## 六、当前在跑

- `tests/rl_driver.sh` (GPU0): **idbd / autostep(论文默认) / autostep(α₀=0.2) / value** × 3 seed
  —— 修复参数传递后重跑, 用于回答"步长分化到底带不带来收益"
- `tests/rm_rerun.sh` (GPU1): `random-matched` × 3 seed —— 那个**唯一干净的对照臂**,
  之前因 JSON 序列化 bug 全部作废
- `bm2` 已完成 (`results/BM2_DONE`): value 0.7783 / random 0.7872 / value-nofb 0.7768
  → **修正后的价值函数没有打赢 random**, 待 random-matched 补齐后做完整统计
