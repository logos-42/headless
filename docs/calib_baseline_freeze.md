# 校准基线冻结契约(阶段一)

**日期**: 2026-09-27
**目的**: 后续每个实验(生存参数扫描、稠密 cumulant、多 GVF、进 LM4)都要有**干净的参照**。
这份文件把参照钉死, 改它必须是一次显式提交并说明理由。

---

## 一、冻结的配置

`tests/benchmark_three_state.py` 的 `FROZEN` 字典是唯一权威来源:

```python
FROZEN = dict(rr_select_rule="calibrated", rr_pot="gvf", rr_retire_rule="advq",
              rr_score="advq", rr_forget_hl=20, rr_forgive_p=0.10)
```

| 旋钮 | 值 | 状态 | 理由 |
|---|---|---|---|
| `rr_pot` | `gvf` | **冻结** | 唯一在线可用、无 oracle 的势函数。`exact` 只是 oracle 对照 |
| `rr_retire_rule` | `advq` | **冻结** | 主任务信号判据(SMDP advantage) |
| `rr_score` | `advq` | **冻结** | 选择分数与判据同源 |
| `rr_forget_hl` | `20` | **冻结** | 遗忘必须开(不遗忘时旧账把故障盖住) |
| `rr_forgive_p` | `0.10` | **冻结** | 退休必须可逆 |
| `rr_three_state` | 消融轴 | 见下 | 三态本身是一条待测轴 |
| `rr_pot="learned"` | — | **不再作为候选** | 实测初值 artifact(`V(s0)` 逐位 1.0000) |
| `rr_pot="exact"` | — | **只作对照** | 用了真转移模型 = oracle |

未扫、明确留待阶段三: `rr_lr_v` / `rr_starve_steps` / `rr_unk_p` /
`rr_min_launches` / `rr_ucb_c` / regime 切换速度。

## 二、三个固定场景

| 场景 | regime 链 | 故障 | 施加 | 期望 |
|---|---|---|---|---|
| `normal` | R0,R1,R2 | 无 | — | 目标 option 判**好** |
| `inert` | R0..R2 ×2 | `inert`(上限压到 1 步) | 第 3 段起 | **不误杀**(达不成子目标但每步是好的) |
| `harm` | R0..R2 ×2 | `harm`(反转 π_o) | 第 3 段起 | 检出为**坏**;且 agent 能自救 |

两个臂:
- `b2-nostate` —— 冻结基线但 `rr_three_state=False`(三态只报不用)
- `b2-3state`  —— 同上 + `rr_three_state=True`

> **为什么三态也要当消融臂**: 否则"三态有没有用"不可归因 ——
> 这和上一轮 `gain` 规则被关掉是同一类错误的反面。

## 三、八项固定指标

    ① harm 检测延迟        ② inert 误杀率(非目标 option 被判坏)
    ③ 退休后恢复次数        ④ fallback 比例(缓存为空时的退回)
    ⑤ 未知状态占比          ⑥ 主任务成功率与步数
    ⑦ 证据是否持续增长      ⑧ **自欺率**(奖励流为空时仍报"好"的比例)

## 四、判据(全部事前写死, 跑前确定)

| # | 判据 | 依据 |
|---|---|---|
| ① | 任何 `starved=True` 的段, `states['good'] == 0` | 自欺门: "没数据"绝不能报成"没问题" |
| ② | harm 场景 `3state` 故障段成功率 > `nostate` 且差 > 0.2 | 三态必须改变**行为**, 不只是报告 |
| ③ | `3state` 误杀 ≤ `nostate` | 未知不得触发退休 |
| ④ | inert 场景 `3state` 累计误杀 == 0 | 未知/次优 ≠ 坏 |
| ⑤ | normal 场景目标 option 最终判"好" | 三态不能退化成一律未知 |
| ⑥ | 每个场景 `advq_n` 跨段单调不减 | 证据必须持续增长 |
| ⑦ | 主任务步数不劣于 `uniform`(增幅 < 15%);`uniform` 在故障段整体失败时由 ② 覆盖 | 安全基线 |

## 五、当前结果(3 seeds × 300 回合)

    11/11 通过

    harm 故障段成功率    nostate 0.000  ->  3state 0.567
    自欺门违约           0 处
    inert 累计误杀       nostate 0 / 3state 0
    normal 末段三态       good >= 1
    advq_n 单调增长       normal / inert / harm 三场景全过
    主任务步数           normal 11.24→11.22 / inert 11.16→11.17(不劣)

**关键**: `b2-nostate` 在 harm 下**整体死亡**(成功率 0.000、`steps_go = nan`、
`starved=True`、`since_rew` 持续增长)且**永不自救**;
`b2-3state` 检出(延迟 43 步)、判坏、并被原谅机制重新验证
(`retired_n`/`recovered_n` 有计数)。

## 六、阶段三(下一步):生存参数扫描

按这张表扫, 目标是**画出机制边界**, 不是找最优数字:

| 参数 | 取值范围 | 要回答的问题 |
|---|---|---|
| `rr_lr_v` | 低/中/高 | 势函数学得太慢 vs 太快(跟随失败而非跟随任务) |
| `rr_forgive_p` | 0 / 0.02 / 0.10 / 0.3 | 到哪里为止是"可逆", 从哪里开始是"无限制放行" |
| `rr_forget_hl` | inf / 200 / 50 / 20 | 陈旧证据 vs 迁移 |
| `rr_min_launches` | 3 / 6 / 12 | 判据的最低证据门槛 |
| `rr_ucb_c` | 0.5 / 1.0 / 2.0 | 保守 vs 激进 |
| `rr_unk_p` | 0 / 0.5 / 1.0 | "有限度探索"的力度 |

每个取值都要报 ①–⑧ 八个指标, 并明确归类到:
**把未知误判成坏 / 让坏 option 长期存活 / 宽恕变成无限制放行 / regime shift 后学不动**。
