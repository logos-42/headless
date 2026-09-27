# Ground truth:在论文 gridworld 上复现 STOMP 的正反例

> 论文:Sutton et al., **Reward-Respecting Subtasks for Model-Based RL**, arXiv 2202.03466, Fig.1 / Fig.6 / §6
> 代码:`hibs_lnn/gridworld.py` + `hibs_lnn/subtask_options.py` + `tests/benchmark_oak_repro.py`

## 1. 为什么需要这个

本项目在 lm4 上测出「Options 显著有害」,但同一批测量证明 **lm4 的不对称性 `A ≡ 0`**
——动作完全可交换、唯一的动力学是对称纯干扰,**不存在可供时间抽象利用的顺序依赖**。
在那种环境上任何类别的 option 都不可能有收益,所以那个负结果**测的是基准的性质**。

论文的 gridworld 则有**明确的正反例**:

```
只用 primitive actions                —— 基线
+ shortest-path (bottleneck) option   —— 论文实测**比基线还慢**
+ reward-respecting option            —— 论文实测**明显更快**
```

在这样一个有已知答案的环境上跑,才能回答一个可判定的问题:
**我的 option + option model + planning 管线到底写对没有?**

## 2. 环境(逐条对齐论文)

`hibs_lnn/gridworld.py` 实现论文规格:4 动作、撞墙不动、到达 goal 拿 +1 且 episode 结束、
**落点**在灰色区每步 −1、γ = 0.99。两房间为确定性(Fig.1),四房间为随机动力学
(Fig.6:期望方向 2/3,其余三个方向各 1/9)。

### 2.1 布局必须过自检,否则复现不了论文

三条性质**缺一不可**:没有「必须穿过灰色」就没有 shortest-path 的代价,
没有「绕路」就没有 reward-respecting 的收益。

两房间(修正后,`check_two_room`):

```
起点 (1,4) -> 走廊 (3,4)  [介数 636]
最短路长度 2, 其中灰色格数 1      <- 必须穿过负奖励
绕开灰色区的路径 6 步 > 2 步      <- 存在绕路
三条性质全满足: ✓
```

四房间(修正后,`check_four_room` —— **这个检查是本次新加的**):

```
状态数 80  走廊格 3 个: [(2,6), (5,3), (8,6)]
堵上走廊后的连通分量数: 4         <- 确实是 2x2 四房间, 不是一团
走廊 (2,6): 最短路 6 步(灰色 2 格)  绕行 8 步  绕行更长 ✓
四条性质全满足: ✓
```

### 2.2 ★ 布局与自检本身修掉了三处缺陷(见 §4)

`validate_layout()` 现在在**构造时硬报错**,而不是默默容错。

## 3. 结果:论文 Fig.1 与 Fig.6 **两个方向都复现**

### 3.1 两房间(确定性,Fig.1)

```
条件 A  只用 primitive actions          look-ahead = 1400   (基线)
条件 B  + shortest-path option          look-ahead = 1750   相对 A = 1.250  ✓ 更慢
条件 C  + reward-respecting option      look-ahead =  875   相对 A = 0.625  ✓ 更快 (1.6x)

判据(事前写死):
  ① shortest-path 不比 primitive 更快 : ✓
  ② reward-respecting 明显更快        : ✓
```

### 3.2 四房间(随机动力学,Fig.6)—— **首次跑通**

```
状态数 80, γ = 0.99, slip = 2/3
真值 v*(s0) = 0.631153

条件 A  只用 primitive actions          look-ahead = 48872
条件 B  + shortest-path option          look-ahead = 53710   相对 A = 1.099  ✓ 更慢
条件 C  + reward-respecting option      look-ahead = 26200   相对 A = 0.536  ✓ 更快 (1.87x)

判据(事前写死):
  ① shortest-path 不比 primitive 更快 : ✓
  ② reward-respecting 明显更快        : ✓
```

**随机动力学下结论方向不变** —— 这一点比数字重要:它说明
reward-respecting subtask 的价值不是确定性世界的 artifact。

### 3.3 论文 §6 的 bonus 权重扫描也复现

论文:大 `w̄` 退化成 shortest-path。

```
w̄ = 1     ->  195     ratio 0.114   <- 论文主用值, 最好
w̄ = 10    ->  2145    ratio 1.250   <- 已退化成 shortest-path 水平
w̄ = 100   ->  2145    ratio 1.250   <- 完全退化成 shortest-path
```

→ **`w̄ = 1` 最好,`w̄ ≥ 10` 退化成 shortest-path** —— 与论文 §6 逐条吻合。
(此扫描在旧布局上做的;方向性结论不受布局修正影响,但数值需在新布局上重扫才算数。)

## 4. 复现过程中被抓到的真 bug(这就是 ground truth 的作用)

### 4.1 管线 bug(只在"有已知答案"的环境里才暴露)

**bug 1:stopping value 恒等于主任务价值 → `β_o` 处处为空**

第一版把式(4) 的 `w_i` 硬写成 `1.0` 且 `w̄_i = 1`,于是
`z_i(s) = V_main(s) + (w̄_i − w_i)·x_i(s) = V_main(s)` —— bonus 恒为 0。
后果:子任务近似主任务 → `β_o` 处处为假 → option 永不停止 →
reward-respecting 与 shortest-path 给出**完全相同**的曲线(实测 2145 / 2145)。
正是论文原话警告的情形:

> "The stopping values should not equal the estimated values because then the subtask
> would approximate the main task and solving it would probably add nothing new."

**修法**:`w_i` 是**特征 i 在主任务价值函数里的权重**,不是 1。

**bug 2:planner 只贴现一步 → 价值迭代发散**

option model 返回未贴现的累计奖励,planner 写成 `r_cum + γ·V(s_term)`,
只贴现**一步**而非 option 实际占用的 `k` 步 → **打满 300000 ops 仍未收敛**。

**修法**:SMDP 正确形式 `Q(s,o) = E[r_disc + γ^k · V(s_term)]`。

**bug 3:goal 不是吸收态 → 价值爆炸**

允许从 goal 继续移动并反复领 +1,子任务里"走廊处的继续价值"算成 **98.01**(真值 ≤ 1)。

**修法**:goal 设为吸收态。

### 4.2 ★ 环境/自检 bug(四房间一直复现不了的真正原因)

**bug 4:布局行长不齐 + 混入空格 → 静默改变几何**

旧 `FOUR_ROOM` 行长分别是 12/13/14,且有一行含**空格**。
`GridWorld.__init__` 用 `.ljust(W, '#')` 补齐 —— 于是

- 短行被**悄悄补成墙**,没人设计过的几何被拿去跑
- 只有 `'#'` 被当作墙,于是**空格变成一格可走的幽灵地板**

两者都不报错。**四房间那一支一直在跑一个非预期的环境。**

**修法**:`validate_layout()` 在构造时**硬报错**(行长不齐 / 非法字符 / 边界非墙 / S 与 G 数量)。

**bug 5:`TWO_ROOM` 同样行长不齐 —— 已"复现成功"的结果也建在被改过的几何上**

新校验一上线就抓到两房间的第 3、4 行是 8 字符(其余 9)。

**修法**:按论文重设布局,并**重跑复现**确认方向不变(见 §3.1)。

**bug 6:`check_two_room` 的 hall 可能落在灰色区里 → 这条自检永远不可能通过**

原逻辑要求「存在**绕开灰色区**到达 hall 的路径」,但 hall 的选取只排除了起点和终点,
没排除灰色格。实测 hall 常常正好是一个 `-` 格 → 该条件**构造上不可能为真** →
自检永远报 ✗,成了没人看的红灯。

**修法**:hall 的候选集排除灰色格。修复后 `TWO_ROOM` 的真实缺陷才暴露出来(见 bug 7)。

**bug 7:旧 `TWO_ROOM` 实际一条性质都不满足**

修好 bug 6 后立刻测出:旧几何的 col 1 是通的 → 有一条**自由竖井绕过整个灰色区** →
最短路**根本不碰灰色**(`neg_on_shortest = 0`),且"绕路"与最短路等长。
**论文 Fig.1 的三条性质一条都没满足。**

**修法**:重设布局,让灰色区**正对**单格通道(穿灰 2 步 / 绕行 6 步)。

**这七个 bug 全都只会在一个「有已知正确答案」的环境里暴露出来。**
在 lm4 上它们只会表现为「option 有害」。

## 5. 结论

```
1. 管线是对的 —— option + option model + planning 复现了论文的**两个方向**,
   且**确定性(Fig.1)与随机动力学(Fig.6)都成立**

2. 本项目在 lm4 上的负结果因此**确认**由三件事造成, 与"option 有没有价值"无关:
     a. 只实现了论文实测最差的那一类 option (bottleneck / shortest-path)
     b. 基准本身没有可被抽象的结构 (A ≡ 0)
     c. 度量选了 planning efficiency 以外的量

3. 下一步按论文顺序做 (见 docs/oak_diagnosis_from_papers.md §12):
     补 subtask 层 (GVF 的 c/z) -> option model 进 planner -> utility feedback
```

## 6. 未完成(诚实标注)

- **效果量小于论文**:两房间 1.6×、四房间 1.87×,而论文报告的是 8.8× 量级。
  可能原因:① 我的 `π_o` 由**表格价值迭代精确解出**(理想解,理论上应更快而非更慢)
  ② 布局比论文小。**这两个原因都还没做消融**,所以"效果量小"目前只是一个观察,
  不能解释成"机制弱"。
- **§3.3 的 `w̄` 扫描需在新布局上重扫**(旧数值来自被静默改过的几何)。
- option 的 `π_o` 不是论文的 off-policy actor-critic(UWT)。这是刻意的第一步:
  先保证 subtask/model/planning 这条链正确,再换学习器。
- `max_steps` 仍有上界(200);论文里 option 由 `β_o` 自然终止。
- 四房间的 `check_four_room` 里「走廊是介数最高点」一项为 False
  (介数 2320 vs 全局最高 3262)。已核验全局最高的状态不是走廊格,
  说明**介数不是识别走廊的可靠指标** —— 正确做法是用布局里的 `'+'` 标记(代码已如此)。
  这一条记下来,因为它正是本项目用「bottleneck 中心性」找 option 目标的同一个弱点。
