# OaK 结构件: Options / 反事实 rollout / 不确定性门控

按用户指点:**停止往系统里堆 RL 算法**(PPO/SAC/TD3 会变成"算法动物园"),
转而补 OaK 的**结构性组件**:

```
Prediction → Abstraction → Options → World Model → Planning
```

本轮实现三个模块,并把它们从"单元测试通过"推进到"可接主回路"。

---

## 1. 三个模块

### 1.1 `hibs_lnn/uncertainty_gate.py` (153 行)

| 组件 | 作用 |
|:--|:--|
| `TransitionEnsemble` | **bootstrap 集成**(K 个模型),`predict(x,u)` 返回 (集成均值, **分歧 U_T**) |
| `UncertaintyGate` | `allow(s,a) = [C(s,a) > τ_C] ∧ [U_T(s,a) < τ_U]` |
| `rollout(x0, actions, gate)` | `T^k(x0, a_{t:t+k})`,**逐步做可信度检查,一遇不可信即中止**(不硬外推) |

`U_T` 用**集成分歧**(bootstrap 重采样的预测方差)估计 —— 这是有原则的做法,
比"到最近训练点的距离"更能反映真实的学习不确定性。

### 1.2 `hibs_lnn/option_manager.py` (236 行)

Option 三要素按用户定义:`o = (I_o, π_o, β_o)`。

| 部分 | 实现 |
|:--|:--|
| `I_o` initiation | `initiable(s, radius)` —— 状态落在 option 起始区域附近才可启动 |
| `π_o` policy | **一串子动作** `actions = [a_1..a_k]`,不是单个动作 |
| `β_o` termination | **学习式**:`β_o(s) = σ(w·[s, g−s, unc, 1])`,`learn_termination()` 从数据学 `w` |

**关键:option 是"发现"的,不是手工写死的。**

```
状态聚类 (k-means, n_regions)
   ↓
建转移图: (region r, action u) --T--> next_region, 边权 = U_T
   ↓
筛可靠边: 只留 U_T < τ_U 的边
   ↓
找多步路径: 长度 L=2..max_len 的路径 = 一个 option 的 π_o
   ↓
I_o = 路径起点区域, β_o 从真实数据学
```

这和 `transition_model.py` 直接接上:**Option 由可预测的状态转移结构产生**,
而不是首先由 reward 定义。

---

## 2. 验证结果 (`tests/test_oak_structure.py`,四测项全过)

```
① Option 发现
   发现 6 个 option, 转移图 20 条边
     actions=[0, 0]      长度=2
     actions=[1, 0]      长度=2
     actions=[0, 0, 0]   长度=3
     ...
   长度 ≥2 的 option: 6 个  ✓ 形成了时间抽象
   用到的动作集合 = [0, 1]   (动作2 噪声=0.60)
   ✓ 高噪声动作被 τ_U 筛掉 —— 可靠路径过滤起作用

①b 预测正确性 (含**远离数据质心**的探针 —— 回归守卫)
   探针 = 区域中心 R0 = [-0.07]
     a0: Δ=[0.599] 真值=[0.6]  误差=0.0010 ✓
     a1: Δ=[0.249] 真值=[0.25] 误差=0.0009 ✓
     a2: Δ=[0.208] 真值=[0.2]  误差=0.0078 ✓
     a3: Δ=[0.029] 真值=[0.03] 误差=0.0010 ✓
   -> 4/4 正确

② 门控挡 OOD
   动作0 有覆盖 (visits=3134): allow=(True,'ok')
   动作0 零覆盖 (复现 [2,2,...]): allow=(False,'low_coverage')  ✓
   动作0 U_T=0.00143  vs  动作2 U_T=0.00524   -> 高噪声被识别 ✓

③ rollout 中止
   不限门控 [0,0,1]: 4 步, 中止=None
   零覆盖下 [0,0,0] (门控开): 1 步, 中止=(0,'low_coverage')  ✓

④ 学习式终止 β_o
   option actions=[0,0], 目标区域中心=[2.114]
   β_o 权重 = [-0.0051 -2.1048 0.0 -1.5986]
   距目标 -1.0 -> P(terminate)=0.1152
   距目标 -0.5 -> P(terminate)=0.2712
   距目标 -0.2 -> P(terminate)=0.4112
   距目标 +0.0 -> P(terminate)=0.5153
   距目标 +0.2 -> P(terminate)=0.6180
   ✓ 单调上升 (从数据学出, 不是硬阈值)
```

`β_o` 在 `g − s` 上的权重 = **−2.1048**(负号正确:离目标越远越不终止)。

---

## 3. 过程中修掉的 4 个真 bug(每条都有回读验证)

### 3.1 特征里冗余常数列 → 条件数 1.24e16

`feat` 写成 `concat([x, onehot(u), [mean(visits)]])`,而 `mean(visits) ≡ 1`
与截距**完全共线**。后果是双向的:

```
条件数 = 1.235e16 (近乎奇异)
Δ 估计 = 35.6 / 32.5 / 10.3 / 5.7     真值应 ~0.3        ← 预测全错
U_T   对所有动作都 ≈ 3.27                                ← 完全不区分噪声
```

去掉后:`Δ = [0.30, 0, 0, 0]` ✓,`U_T` 动作0=0.0018 vs 动作2=0.0399 → **22× 区分** ✓

### 3.2 onehot 与截距共线 → 远离质心时预测炸

`onehot(u)` 各列之和恒等于 1,所以再加一列 `ones` **也是完全共线**。
条件数仍 8.9e15;解只在**数据质心附近碰巧正确**,远离即炸:

```
在区域中心 R0 = [4.95, 4.55, 1.36, 1.22] (远离质心):
  Δ = [92.7, 96.4, 25.8, 23.7]      真值应为 [0.30, 0, 0, 0]
```

去掉冗余截距后条件数 **12.8**,远离质心也正确。**这就是 ①b 那条回归测试守的东西。**

### 3.3 fit/predict 对「Δ vs 绝对状态」的约定不一致

测试把**绝对下一状态**当 Δ 喂进来。在质心处 `x=0` 使两者**巧合相等**,
完全掩盖了这个 bug;一旦远离质心就露出 `Δ = x + fx(u)` 的错误。

修法:**接口改成自洽的**(fit/predict 都用绝对下一状态,内部学 Δ 再补回 x),
从设计上消除误用可能。

### 3.4 测试夹具三次尺度错配(不是模块的问题)

| 尝试 | 病 |
|:--|:--|
| 状态生成成 i.i.d. 噪声 | 状态根本不演化 → 不存在多步结构 → 0 option |
| 动作效果 (0.3) << 区域间距(数个单位) | `center+effect` 的最近区域永远是它自己 → 无"前进的边" → 0 option |
| 纯随机游走**无界漂移** | 区域中心跑到 ~12 → 同一个病 |

可行解 = **低维链式环境**(`dim_x=1`:状态沿一条线排开,单步跨相邻区域,
多步序列跨多个区域)= option 的经典形态。

**真实 lm4 的对应关系**:状态 = [0,1] 的精度画像(**有界**),单次训练改变
0.1~0.3,与 6 个域的区域间距**同量级** —— 天生满足条件。**所以夹具的病不会
出现在真实数据上。**

---

## 4. 与 OaK 的对照(更新)

| OaK 组件 | 状态 | 说明 |
|:--|:--|:--|
| Action → Env → Reward → 参数更新 | ✅ | `rl_proposer.py` |
| 多步长 policy accumulation | ⚠️ 无收益 | **negative result**(n=11 后 p=0.564) |
| Policy 跨 redefine accumulation | ✅ | `redefine()` 探索继承 |
| δ^pred / δ^RL 分离 | ✅ | `dual_proposer.py`,逐位隔离 |
| Coverage gating | ✅ | **升级为** coverage + **uncertainty** |
| Transition model | ✅ | **加 rollout + 不确定性** |
| **Options** | ✅ **本轮** | `option_manager.py`,**从转移结构发现** |
| **Temporal abstraction** | ✅ **本轮** | 多步动作序列 |
| **Learned termination** | ✅ **本轮** | `β_o` 逻辑回归,非硬阈值 |
| Planning | ⚠️ 隐含 | 待用 transition model + options 显式实现 |
| Real actuator | ❌ | **不强行解决**(被动观测) |
| Offline/model-based control | 🟡 半 | `rollout` 已就绪,待接目标 |
| Safety layer | ✅ | coverage + uncertainty + action bounds |

---

## 5. 下一步(按用户给的 E 表)

三个模块已在**单元层面**验证。剩下的关键是把它们**接进 lm4 主回路**:

1. 把 `DualSignalProposer` + `OptionManager` 接进 `run_lm4_wave.py`
2. 用 `transition_model` + `rollout` 做**反事实离线控制**(替代缺失的 actuator)
3. 跑 regime-shift 实验(`D_1 → D_2 → D_3`,模拟环境变化)
4. 实验矩阵:E1 Fixed LR / E2 IDBD / E5 continual plasticity / E6 prediction only /
   E7 +RL / E8 +transition model / E9 +coverage gating / **E10 +Options** /
   E11 regime shift / E12 repeated regime shift

**要证明的命题**:

```
Prediction → Abstraction → Option → Planning → Control
        比
state → RL policy → action
更适合**持续变化的磁通动力学**
```

**实验边界(必须明说)**:没有真实 actuator,所以只能声称"完成了基于**被动观测**
数据的 predictive/control-policy learning,并进行了**离线/模型内反事实**控制验证",
不能声称完成了真实磁通控制实验。


---

## 6. InternalKnowledge: 内部知识层 (`hibs_lnn/knowledge.py`)

用户给的立场是 **Knowledge ≠ Parameter** —— 参数只是**承载**知识的一种方式。
所以这一层**不暴露 θ/W/α**,只暴露**可计算的知识查询**。

### 6.1 四层 + 三类元知识

| 层 | 类 | 回答 |
|:--|:--|:--|
| Level 1 预测 `K_P` | `GVFBank`(Horde/GVF) | 未来会发生什么 |
| Level 2 转移 `K_T` | `TransitionEnsemble` | 我做什么会导致什么 |
| Level 3 价值 `K_V` | `register_value_fn` | 什么结果更好 |
| Level 4 抽象 `K_A` | `OptionManager` | 哪些过程是一个整体技能 |
| 元-不确定 | `UncertaintyGate` | 我对自己的预测有多确定 |
| 元-覆盖 | `Coverage` | 我对这个状态/动作到底有没有经验 |
| 元-可塑性 | `Plasticity` | 我应该**多快**改变这个知识 (接 IDBD) |

`α = knowledge about plasticity` —— 它**不是**知识本身,而是"关于知识该如何改变"
的元知识;`β` 是"学习速度本身该如何变化"的元状态。这与 IDBD 主线直接接上。

### 6.2 核心接口: `knowledge_of(s, a)`

返回**内容**而不是参数,并给出三种明确判定:

```
a=0 -> V=0.600  覆盖=436  可信=True   新=False  判定=**我知道**
a=2 -> V=1.200  覆盖=463  可信=True   新=False  判定=**不太确定**
a=7 -> V=None   覆盖=  0  可信=False  新=True   判定=**我不知道**
```

第三行是整层的立命之处:**agent 明确区分「我知道」与「我不知道」,
而不是对没见过的 (s,a) 编一个数出来。**

### 6.3 知识能被留下来,增长可测

`to_dict()` 冻结快照 → 继续积累经验 → `growth(snap)` **逐项指出变化量**:

```
coverage_total_delta       1800
gvf_weight_delta_mean      0.88514      预测知识变了多少
dyn_weight_delta_mean      0.13397      转移知识变了多少
plasticity_beta_delta      [0.00992 ...] 元知识变了多少
```

这才是"**持续构建内部知识**"的可判定形式 —— 不是"模型又训了一轮",
而是能具体指出知识变了什么。

### 6.4 ★ 实测抓到的两件事(都是真问题)

**(a) 知识层对未知输入必须返回"不知道",不能崩。**
测试对从未出现过的动作 7 查询,`value_fn` 抛 `KeyError` 一路炸穿 `knowledge_of`。
修:`value()` 捕获异常返回 NaN,`knowledge_of()` 增加 `known` / `verdict` 字段。
**"承认不知道"是这一层的核心语义,不是容错细节。**

**(b) ★★ 把 α 的更新从它要优化的学习里孤立出来 —— 测试方法错误。**
第一版测试直接跑 `pl.update(phi, delta)` 而 `delta ≡ 1 + noise`,结果**8 种
稳定化手段(降 μ / δ RMS 归一化 / 保持 αφ²≤0.99 / h 上界)全部撞界**。
根因是结构性的:

```
g = δ·φ·h,  迹的稳态 h* = α·φ·δ/(α·φ²) = δ/φ
⟹  g ≈ δ²/φ  **恒为正**  ⟹  β += μ·δ²/φ  单向漂移, 永不停
```

**β 能停下来的唯一原因是「δ 随 w 收敛而趋 0」。** 孤立地跑元更新,
就永远没有收敛信号 —— 这不是模块的 bug,是**测试问错了问题**。

改用**闭环**(`w += α⊙δ̂⊙x` 真的在学 + 稀疏非平稳回归:只有 w[0] 可预测):

```
步      α[0]      α[1:]均值     |δ|
   0   0.0500      0.0500     0.3772
 500   0.0875      0.0502     0.0000      <- δ 收敛
1000   0.0655      0.0423     0.0026
 ...
3500   0.0298      0.0244     0.0001
α[0] (真特征) > α[1:] (纯噪声)  ✓
β 撞界 0/6  ✓ 全程有界
```

**诚实标注**:分化是**真实但弱**的(ratio ≈ 1.22)。这与本项目此前两条独立
负面结论一致 —— per-weight 步长元学在我们这类任务上**至多是小幅效应**。
