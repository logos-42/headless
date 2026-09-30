# 阶段五规格 —— 证据库 (evidence bank) × RR three-state

**写下时间**: 2026-09-30,在写任何实现代码之前
**上游**: 用户七阶段路线图的阶段五;`docs/a_prime_decision.md`、`docs/l5_verdict.md`
**一句话**: 把 GVFBank 从「一组预测器的平均」改造成**证据库**,并让它**驱动**
三态判定 —— 不是加一个新势函数。

---

## 0. 当前的实情(动手前先看清)

| 位置 | 现状 |
|:--|:--|
| `hibs_lnn/knowledge.py:107` | `GVFBank.update()` **返回 `np.mean(errs)`** —— **就是一个简单平均** |
| `hibs_lnn/knowledge.py:116` | `stats()["out_norm"]` 用 `np.ones_like(w)` 当输入 —— 假输入,量没有语义 |
| `hibs_lnn/oak_proposer.py:338` | `observe_gvf(phi, cums, phi)` —— `phi_next` 传的是 `phi` 本身 |
| `hibs_lnn/rr_agent.py:770` | `rr_epistemic(o)` 只读 `rr_outcome`(**option 达成率**),与 GVF **零连接** |

⇒ **阶段五的实质工作 = 把「预测知识」与「三态判定」接起来,并且接法是"证据"不是"平均"。**

## 1. 用户给的硬约束(逐条对应实现)

1. **先回答更基础的问题:多个 GVF 能不能产生互补证据?**
   ⇒ 必须先有一个**互补性度量**,并且允许答案是"不能"。
2. `GVF_1 → prediction` / `GVF_2 → reward` / `GVF_3 → transition/stability` /
   `GVF_4 → regime`
   ⇒ 四个 GVF 要有**明确的语义分工**,每个记自己的量。
3. **绝对不要把多个 GVF 简单平均**
   ⇒ 聚合的返回值里**不许出现跨 GVF 的标量均值**;要能指出**哪条证据**支持什么。
4. **保留 e_1, e_2, e_3 …**,让系统知道
   *哪个证据支持 good* / *哪个支持 bad* / *哪个缺失*
   ⇒ 聚合输出 = **引用清单(citations)**,不是分数。
5. **InternalKnowledge 首先是 evidence bank,不是一个新的势函数**
   ⇒ 不新增 `Φ`;证据**驱动**已有的三态,不替换 `rr_pot`。

## 2. 数据结构:每条证据自己说话

```python
class GVFEvidence:
    """一条证据 —— 不是权重, 是"我凭什么这么说"。"""
    name        # 'prediction' | 'reward' | 'transition' | 'regime'
    role        # 'dynamics' | 'outcome'    ← 决定它能不能"升级"
    value       # 当前预测值
    err_ema     # 近期预测误差 (EMA)
    err_ref     # 参考尺度 (该 GVF 自己的历史分位, 不是全局常数)
    n           # 更新次数 = 覆盖度
    last_step   # 最近更新的步号
    regime      # 最近更新时所在 regime
    usable      # 当前是否可用 (见 §3)
```

**每个字段都是"可判定的"**,不留"看起来还行"这种状态。

## 3. 可用性(usable)的判定 —— 缺失的证据要显式缺席

一条证据在下列任一情况下**不可用**,且必须**报为"缺失"而不是按 0 平均进去**:

| 条件 | 理由 |
|:--|:--|
| `n < n_min` | 没有足够经验 —— 这正是"未知"的定义 |
| `step - last_step > stale_hl` | 证据陈旧,不适用于当前时刻 |
| `regime != 当前 regime` | 证据属于另一个 regime,不能迁移过来直接用 |

★ **这一条是本规格的中心**:旧实现把"没有证据"和"证据说没问题"混成同一个数(都落到 0),
于是**未知永远伪装成好**。显式缺席是修这个病的唯一办法。

★ 陈旧阈值 `stale_hl` 沿用项目已扫过的口径(与 `rr_forget_hl` 同族),不新引入超参族。

## 4. 聚合 = 投票 + 引用,不是平均

```
对每条可用证据 e:
    z(e) = e.err_ema / max(e.err_ref, eps)          # 该 GVF 自己的尺度上的相对误差
    if z(e) >= z_bad:      → bad 票     (附来源 e.name 与 z)
    elif z(e) <= z_good:   → healthy 票 (附来源)
    else:                  → 无票       (中性区, 不投票)

对每条不可用证据 e:
    → missing 票 (附原因: low_coverage / stale / wrong_regime)
```

**判定规则(不对称,沿用反熟悉度保证)**

| 条件 | 判定 |
|:--|:--|
| 有 `bad` 票 | **bad** |
| 无 `bad` 票, 且有 `healthy` 票 **且该票来自 `role == "outcome"`** | **good** |
| 无 `bad` 票, 但有 `missing` 票 | **unknown** |
| 其余 | **unknown** |

★★ **动力学类证据(`role == "dynamics"`)只能降级、不能升级** ——
它可以投票 `bad`,但**永远不能**把状态升成 `good`。
理由与 A 类一致:预测误差低只说明"我预测得准",不说明"这件事有价值";
而**新颖度会拉高误差 ⇒ 若允许升级,新颖度就能伪造"好"**。
这条是硬约束,不是可调项。

**输出必须是引用清单**:

```python
{
  "verdict": "bad",
  "bad":     [{"src": "transition", "z": 3.4}, {"src": "regime", "z": 2.1}],
  "healthy": [],
  "missing": [{"src": "reward", "why": "low_coverage", "n": 3}],
  "usable_n": 2, "total_n": 4,
  "sources": {...每条完整记录...}      # e_1, e_2, e_3, e_4 原样保留
}
```

★ 检查:返回值里**没有任何跨 GVF 的标量均值**。`usable_n`/`total_n` 是计数不是平均。

## 5. 互补性度量(回答"多个 GVF 能不能产生互补证据")

**在已知 regime 切换点**上,把每条 GVF 的 `|z|` 当作一个"变化检测器":

| 量 | 定义 |
|:--|:--|
| `AUC_i` | 单条 GVF `i` 检测 regime 变化的 AUC |
| `AUC_any` | 「**任一条**超阈」的并集 AUC(**检测**口径 —— 加检测器不会降 recall) |
| `AUC_consensus` | 「至少两条同时超阈」的并集 AUC(**一致性**口径,合取,天生比单条保守) |
| **`complementarity`** | `AUC_any − max_i AUC_i` |
| `disagreement` | 判定步中"至少一条投 bad 且至少一条投 healthy"的比例 |
| `corr_err` | 各 GVF 误差序列的两两相关矩阵(冗余度) |

> **2026-09-30 修订(实现期发现,在结果出来之前)**:
> 初版只定义了「至少两条同时超阈」一种并集。那是**合取**,天生比任何单条保守 ⇒
> 用它算出来的 `union − best_single` 必然偏负,那是**聚合规则的产物**,
> 不是"证据不互补"的证据。故拆成 `any`(检测)/`consensus`(一致性)**两个都报**。
> 判据里的 `complementarity` 改用 `AUC_any`。
>
> 另需注意:当阈值触发率极低时,并集标志几乎恒为 0,AUC 会被位置抖动主导 ⇒
> **该数值不可解释**。所以伴随判据必须同时报**触发率**;
> `bad` 票为 0 时不得下"没有变化"的结论,只能下"**机制未工作**"。

**判据(事前写死)**

- `complementarity > 0` 且自助法区间下界 > 0 ⇒ **有互补证据**
- `complementarity ≈ 0` 且 `disagreement` 低 ⇒ **冗余** ⇒ 诚实结论是
  「当前这组 GVF 不提供互补证据」,`GVFBank` 退化为"多了一个预测器",**不写成收益**
- 单条 GVF 的 AUC 全部 ≤ 0.5 ⇒ 这组 GVF 连检测都做不到 ⇒ 记 **unknown 机制不成立**

★ 这一条允许(并要求)**证伪阶段五的核心假设**。若结果是冗余,报告就写冗余。

## 6. 与三态的接线契约

    GVFBank(4 条语义分工) → EvidenceBank.evidence() → support() 引用清单
                                                        ↓
                              RRSkillAgent.rr_epistemic(o) 的**输入之一**
                                                        ↓
                                          good / unknown / bad → 策略

**约束**

1. `rr_evidence` 默认 `False` ⇒ **旧行为逐位不变**(与 `rr_three_state` 的纪律一致)。
2. 打开后,证据只**参与**判定,不替换 `rr_pot`(不动势函数)。
3. 判定结果必须**可追溯**:报告里能列出每条判定背后的来源。
4. **`bad` 不得触发永久退休** —— 沿用既有规则(退休必须可逆)。

## 7. 明确不做(留给阶段六)

- 不做 Knowledge Coverage / Accuracy / **Reuse** / Stability / Recovery 五指标
  —— 那是阶段六。本阶段只把证据**接通并可追溯**。
- 不扫 `z_bad` / `z_good` / `n_min` / `stale_hl`(先证明接线正确,再扫参数)。
- 不碰 LM4 主链。
- 不改 `rr_pot` 的任何候选实现。

## 8. 验收(本阶段的完成定义)

1. `tests/test_evidence_bank.py` 全绿,**含一条"不许平均"的不变量测试**:
   把 4 条证据构造成功劳相等但符号相反的极端组合,
   断言输出**不是**它们的均值,且 `bad`/`healthy` 引用清单非空。
2. **缺席 ≠ 健康**:构造"全部证据不可用"的输入,断言判定是 `unknown` 而**不是** `good`。
3. **动力学证据不能升级**:构造"动力学全 healthy、outcome 缺失"的输入,
   断言判定是 `unknown` 而不是 `good`。
4. `rr_evidence=False` 时,既有 `test_rr_agent` / `test_self_calib` / 三态验收台
   **逐位不变**。
5. 互补性度量跑一次真实场景,输出 `complementarity` / `disagreement` / `corr_err`,
   **无论结论是互补还是冗余都照实写。**
