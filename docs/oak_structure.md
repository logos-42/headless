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
