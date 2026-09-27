# A′ 规格:一个使 novelty 与 dynamics change 可辨识的最小 benchmark

**日期**: 2026-09-27
**状态**: 规格(先定义,后写代码)
**上游结论**: 在原 `KeyDoorMDP` 上,`H(T|S,A) = 0` ⇒
`prediction error ≈ novelty`、`stability ≈ novelty`,**A 类不可辨识**。
本文件定义"可辨识"到底是什么意思,以及怎么测。

---

## 零、目标改写(重要)

**错误的目标**: "给 KeyDoor 加噪声,让 A 类通过。"
**正确的目标**: "构造一个使 **novelty** 与 **dynamics change** 可辨识的最小持续学习
benchmark,并用四象限实验验证 A 类是否真正捕获动力学变化。"

**判据不是 A 类通过,而是 A 类有机会被证伪。**
无论 A 类成功还是再次被证伪,阶段五都必须变得有意义 —— 所以 A′ 的产出是
**决策记录**,不是"A 类胜出"。

---

## 一、要测的三个量 + 一个控制量(A′-1)

| 量 | 定义(本轮采用) | 理想情况下应对什么敏感 |
|---|---|---|
| **A1 转移预测误差** | 在线表上 `1 − p̂(s' \| s,a)`(更新前取) | **环境变化** -> ↑ |
| **A2 转移不确定性** | 在线表上后继分布的归一化熵 | **随机性** -> ↑ |
| **A3 稳定性/动力学压力** | 落点 `s'` 的出发边平均归一化熵 | **regime shift** -> ↑ |
| **N 新颖度(控制量)** | `1 / (1 + n(s,a))` | 只应对**首次/罕见访问**敏感 |

**控制量的存在是本规格的核心。** A 类只有在与 N **分开**时才有意义:

    A1 ≈ N 且 A2 ≈ N 且 A3 ≈ N   ->  A 类 = 新颖度检测器, 只能当 coverage 辅助
    A1 ≠ N 或 A2 ≠ N 或 A3 ≠ N   ->  该候选捕获了 novelty 之外的东西

**为什么控制量必须走同一套管道**: 若 N 用另一套统计实现, 差异会来自实现而非语义。
因此 `rr_dyn="N"` 与 `rr_dyn="A1|A2|A3"` 共用 `_rr_dval/_rr_dstep/_rr_dacc/_rr_dyn_bad`
与同一个"与全段基线比较"的判据。

---

## 二、可辨识性的**硬条件**(A′-1 的验收)

新 benchmark 必须能人为制造以下两种构型,否则 A 类仍不可辨识:

    构型 I :  Novelty ↑   而   Dynamics error **不变**
    构型 II:  Dynamics error ↑   而   Novelty **不变**

**构型 II 是真正关键的那个**,也是原 KeyDoor 做不到的:
它要求"同一批状态、同一批 (s,a)、同一批访问计数",但 `P(s'|s,a)` 变了。
只加白噪声做不到(I),只多了新状态;只改状态空间也做不到(II)。

---

## 三、三个环境(A′-2)

### E0 —— 原 deterministic(保留,不删)

    动力学: 完全确定, H(T|S,A) = 0
    作用:   **负对照**。A1/A2/A3/A4 应当全部 ≈ N
    预期:   (已验证) 三者与 novelty 不可区分, 且在无故障场景误报率 = 故障场景

### E1 —— Stationary stochastic

    P(s'|s,a): 以 (1−slip) 执行 a, 否则执行随机动作; **slip 全程不变**
    作用:   制造"不确定性高但无 regime shift"
    预期:   A2(uncertainty) ↑ ; A1 可能 ↑ ; **N 不必然 ↑** ; A3 的 shift 分量 = 0
    这验证构型 I 的一半: 不确定性可与 novelty 分离。

### E2 —— Non-stationary stochastic(**关键**)

    t <  T : slip = slip0   (例如 0.10)
    t >= T : slip = slip1   (例如 0.60)
    作用:   制造真正的 regime shift
    预期:   A1 (预测误差) ↑、A3 (稳定性) ↓ ; **N 不变**(状态集合与计数都不变)
    这验证构型 II, 也就是"变"与"新"的分离。

**为什么 E2 是决定性的**: slip 改变的是 `P(s'|s,a)` 的**形状**,
不改变 agent 走过哪些状态。所以 `n(s,a)` 的计数分布基本不变 ⇒ N 不变,
而 `1 − p̂(s'|s,a)` 在更新瞬间必然上升 ⇒ A1 变。**这是 A1 与 N 分家的唯一构造。**

---

## 四、四象限实验(A′-3)

在 E0/E1/E2 三个环境上,按 **(落点是否有新状态) × (动力学是否变了)**
把 option-段归入四格:

| 格 | 条件 | Novelty | Dynamics error | Uncertainty |
|---|---|---|---|---|
| **S1** 熟悉+稳定 | 计数高, 无 shift | 低 | 低 | 低 |
| **S2** 新状态+稳定 | 计数低, 无 shift | **高** | 低 | 低 |
| **S3** 熟悉+动力学变化 | 计数高, **有 shift** | 低 | **高** | **高** |
| **S4** 新状态+动力学变化 | 计数低, **有 shift** | 高 | 高 | 高 |

**真正要回答的问题不是"A1 有没有触发",而是:**

    A1(S2) ≈ A1(S3) ?   -> 仍然只是 novelty detector
    A1(S3) ≫ A1(S2) ?   -> 真的捕获了动力学变化

因此**核心指标是 `score(S3) − score(S2)`(判别度)**,不是任何单格的分数。

---

## 五、指标(A′-4,取代 trigger count)

原 ⑨ 只看 trigger count(`chaos 6/9 / harm 6/9`),太粗。本轮起至少记录:

| 指标 | 定义 |
|---|---|
| `T_shift` | 人为设定 shift 发生的步 |
| `T_detect` | 该 option 的 A 通道首次被判风险(与基线显著不同)的步 |
| **检测延迟** | `T_detect − T_shift` (**最核心指标**) |
| `T_recover` | 判风险后又回到不显著的步(可逆性) |
| **FP** | 无 shift 时 `P(trigger)` |
| **FN** | 有 shift 时 `P(不 trigger)` |
| **AUC** | 以 A 通道分数区分 shift/无 shift 段的 ROC 面积 |
| **判别度** | `score(S3) − score(S2)` |

**为什么 AUC 与判别度必须同时报**: AUC 高只说明"能分出两种段",
判别度才说明"分的是新还是变"。一个纯 novelty 检测器可以在 E2 上得到不低的
AUC(因为 shift 段恰好也常是新状态),但它的判别度会 ≈ 0。

---

## 六、⑨ 的重定义

原判据 ⑨: `chaos 触发 > 0 且 harm 触发 == 0`(trigger count 比较)

**新判据 ⑨′**:

    ⑨′a  构型 I 成立: 存在环境使得 Novelty 不升而 A2 显著升
    ⑨′b  构型 II 成立: 存在环境使得 Novelty 不变而 A1/A3 显著升
    ⑨′c  判别度: 对每个候选, |score(S3) − score(S2)| > 0 且方向正确
    ⑨′d  检测延迟: shift 后 T_detect − T_shift 有限且 < 段长
    ⑨′e  无假阳性: FP 在 E0/E1(无 shift 环境)上 ≈ 0
    ⑨′f  可恢复: T_recover 存在, 即 shift 后能回到不显著

**⑨′b 与 ⑨′e 是本轮的两个决定性判据。** ⑨′b 若失败 ⇒ A 类仍是 novelty 检测器;
⑨′e 若失败 ⇒ A 类会对正常行为报警, 不能用于决策。

---

## 七、阶段五的进入条件(修正)

**阶段五不因为 A 类失败而取消。** 进入条件从"A 类通过"改为:

    A′ benchmark 建立
          ↓
    A 类具有可辨识性?
          │
       ┌──┴──┐
      Yes    No
       ↓      ↓
    正式评估  记录: 当前候选只能做 coverage
       ↓      ↓
       └──┬───┘
          ↓
        阶段五

即:**A 类被证伪也必须留下记录并进入阶段五**,因为"哪些证据不能用来判断"
本身就是 evidence bank 需要知道的事。

---

## 八、阶段五的定义(先答基础问题, 不做性能研究)

**先回答:多个 GVF 能不能产生互补证据?**

    GVF_1 -> prediction
    GVF_2 -> reward
    GVF_3 -> transition/stability
    GVF_4 -> regime-related signal
              ↓
          GVFBank
              ↓
      evidence aggregation
              ↓
        RR three-state
              ↓
      good / unknown / bad
              ↓
            policy

**绝对不要把多个 GVF 简单平均。** 必须保留 `e_1, e_2, e_3, …` 并让系统知道
"哪个证据支持 good / 哪个支持 bad / 哪个缺失"。

> **`InternalKnowledge` 首先是 evidence bank,不是一个新的势函数。**

这与"构造一个不会自欺的持续学习系统"完全一致:自欺的来源正是把
**缺失的证据**与**中性的证据**压成同一个标量。

---

## 九、阶段六指标(先定义, 暂不实现)

    Knowledge Coverage    覆盖了多少状态/动作/转移
    Knowledge Accuracy    预测误差
    Knowledge Reuse       新任务出现时: 探索是否减少 / 恢复时间是否减少 / 样本是否减少
    Knowledge Stability   regime 内是否不再漂移
    Knowledge Recovery    regime shift 后能否恢复

**Knowledge Reuse 是"学到东西"与"模块变多"的分界线。**

---

## 十、LM4 继续冻结(明确)

现在最有价值的资产不是更大的模型,而是这条链:

    Frozen Contract -> Three-state diagnosis -> Survival boundary
      -> A-class falsification -> **Identifiable benchmark**
      -> Evidence aggregation -> Knowledge growth -> LM4

直接进 LM4 会重现已经抓出来的那个问题:
`unknown -> policy -> 行为看起来更聪明 -> 但 diagnosis 更差`,后面所有结果都难解释。

---

## 十一、执行顺序(7 + 1 个 commit)

    C1  A′ benchmark specification          <- 本文件
    C2  Stationary stochastic KeyDoor (E1)
    C3  Non-stationary KeyDoor (E2)
    C4  四象限 identifiability test
    C5  A1/A2/A3 + N 检测指标
    C6  重跑 ⑨′ + 四门
    C7  A′ decision record
    ---
    C8  GVFBank × RR three-state integration (阶段五)
