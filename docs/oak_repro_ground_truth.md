# Ground truth:在论文 gridworld 上复现 STOMP 的正反例

> 论文:Sutton et al., **Reward-Respecting Subtasks for Model-Based RL**, arXiv 2202.03466, Fig.1 / §6
> 代码:`hibs_lnn/gridworld.py` + `hibs_lnn/subtask_options.py` + `tests/benchmark_oak_repro.py`

## 1. 为什么需要这个

本项目在 lm4 上测出「Options 显著有害」,但同一批测量证明 **lm4 的不对称性 `A ≡ 0`**
——动作完全可交换、唯一的动力学是对称纯干扰,**不存在可供时间抽象利用的顺序依赖**。
在那种环境上任何类别的 option 都不可能有收益,所以那个负结果**测的是基准的性质**。

论文的两房间 gridworld 则有**明确的正反例**(Fig.1):

```
只用 primitive actions                —— 基线
+ shortest-path (bottleneck) option   —— 论文实测**比基线还慢**
+ reward-respecting option            —— 论文实测**明显更快**
```

在这样一个有已知答案的环境上跑,才能回答一个可判定的问题:
**我的 option + option model + planning 管线到底写对没有?**

## 2. 环境(逐条对齐论文)

`hibs_lnn/gridworld.py` 实现论文规格:4 动作、撞墙不动、到达 goal 拿 +1 且 episode 结束、
**落点**在灰色区每步 −1、γ = 0.99。

布局用**自检函数**保证三条性质同时成立(否则复现不了论文):

```
起点 (1,1) -> 走廊 (4,6)  [介数 314]
到达走廊: True   最短路长度 8
最短路上的灰色格数: 1        <- 存在"必须穿过的负奖励"
存在绕开灰色区的路径: True  (长度 10 vs 最短 8)   <- 存在"绕路"
三条性质全满足: ✓
```

第 2 条与第 3 条缺一不可:没有"必须穿过"就没有 shortest-path 的代价,
没有"绕路"就没有 reward-respecting 的收益。

## 3. 结果

```
条件 A  只用 primitive actions          look-ahead = 1716   (基线)
条件 B  + shortest-path option          look-ahead = 2145   相对 A = 1.250  ✓ 更慢
条件 C  + reward-respecting option      look-ahead =  195   相对 A = 0.114  ✓ 快 8.8x

判据(事前写死):
  ① shortest-path 不比 primitive 更快 : ✓  (比值 1.250, 与论文一致)
  ② reward-respecting 明显更快        : ✓  (比值 0.114, 与论文一致)
```

**论文 §6 的 bonus 权重扫描也复现了**(论文:大 `w̄` 退化成 shortest-path):

```
w̄ = 0.1   ->  (小 bonus, 见 JSON)
w̄ = 1     ->  195     ratio 0.114   <- 论文主用值, 最好
w̄ = 10    ->  2145    ratio 1.250   <- 已退化成 shortest-path 水平
w̄ = 100   ->  2145    ratio 1.250   <- 完全退化成 shortest-path
```

→ **`w̄ = 1` 最好,`w̄ ≥ 10` 退化成 shortest-path** —— 与论文 §6 的描述逐条吻合。

## 4. 复现过程中被抓到的**两个真 bug**(这就是 ground truth 的作用)

### bug 1:stopping value 恒等于主任务价值 → `β_o` 处处为空

第一版把式(4) 里的 `w_i` 硬写成 `1.0`,而 `w̄_i = 1`,于是

```
z_i(s) = V_main(s) + (w̄_i − w_i)·x_i(s) = V_main(s)     <- bonus 恒为 0
```

后果:子任务近似主任务 → `β_o` 处处为假 → **option 永不停止** →
option model 退化成"走满 max_steps" → reward-respecting 与 shortest-path
给出**完全相同**的曲线(实测 2145 / 2145)。

这正是论文原话警告的情形:

> "The stopping values should not equal the estimated values because then the subtask
> would approximate the main task and solving it would probably add nothing new."

**修法**:`w_i` 是**特征 i 在主任务价值函数里的权重**,不是 1。
对一热特征,`w_i = mean(V_main[feature > 0])`。

### bug 2:planner 只贴现一步,option 被系统性高估 → 价值迭代发散

第一版的 option model 返回**未贴现**的累计奖励,planner 又写成
`r_cum + γ·V(s_term)` —— 只贴现**一步**,而不是 option 实际占用的 `k` 步。

后果:**look-ahead 打满 300000 仍未收敛**(primitive 只需 1716)。

**修法**:SMDP 的正确形式

```
Q(s, o) = E[ r_disc + γ^k · V(s_term) ]        其中 r_disc = Σ_t γ^t r_t
```

### bug 3(相关):goal 不是吸收态 → 价值爆炸

第一版转移模型允许从 goal 继续移动并反复落回 goal 拿 +1,
子任务里"走廊处的继续价值"算成 **98.01**(真值 ≤ 1)。

**修法**:goal 设为吸收态(论文:"+1 is received on reaching the goal state,
**which ends the episode**")。

**这三个 bug 全都只会在一个"有已知正确答案"的环境里暴露出来。**
在 lm4 上它们只会表现为"option 有害"。

## 5. 结论

```
1. 管线是对的 —— option + option model + planning 能复现论文的**两个方向**
   (bottleneck option 更慢 · reward-respecting option 快 8.8x)

2. 本项目在 lm4 上的负结果因此**确认**由三件事造成, 与"option 有没有价值"无关:
     a. 我只实现了论文实测最差的那一类 option (bottleneck / shortest-path)
     b. 基准本身没有可被抽象的结构 (A ≡ 0)
     c. 度量选了 planning efficiency 以外的量

3. 下一步按论文顺序做 (见 docs/oak_diagnosis_from_papers.md §12):
     补 subtask 层 (GVF 的 c/z) -> option model 进 planner -> utility feedback
```

## 6. 未完成(诚实标注)

- **四房间随机版(Fig.6)未复现**:当前 `FOUR_ROOM` 布局的几何不对
  (它既没有经过有意义的 bottleneck,收敛也未到容差),需要重新设计布局并过自检。
- option 的 `π_o` 目前由**表格价值迭代精确解出**,不是论文的 off-policy actor-critic(UWT)。
  这是刻意的第一步:先保证 subtask/model/planning 这条链正确,再换学习器。
- `max_steps` 仍有一个上界(200);论文里 option 由 `β_o` 自然终止。
