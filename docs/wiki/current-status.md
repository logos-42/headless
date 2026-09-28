---
title: Sovereign AI 当前状态
source: session
created: 2026-09-06
last_confirmed: 2026-09-28
audience: reader
stage: draft
tags: [status]
status: current
---

## 最近更新

- 2026-09-28（下午）：**门控 cron 第四次唤醒 —— 只读复核，数字逐位复现；顺带查出「cron 报告从未送达」**
  - 门控 diff 是 `UNREACHABLE → DONE2` = **6100 端口链路抖动**，不是新的完成事件；
    服务器 bm 数据自 **2026-09-14T08:07 UTC** 起零变化（09-15 后无任何 `bm*` 文件写入）
  - 用与 12:10 提交**同一份**（md5 一致）`tests/analyze_benchmark.py` / `analyze_bm2_full.py`
    对活动数据重跑：全部数字**逐位复现**（lm4 `value` 0.7603±0.0088 / `random-matched` 0.7590±0.0047；
    lm5 `value` 0.2565±0.0928 / `random-matched` 0.2839±0.0157；lm5 `value` vs `value-nofb` 逐位相同）
  - ⚠ **运维缺口（新）**：cron 作业 `bff2174e8bef`（`lm4-lm5-benchmark-done-gate`）在
    `~/.hermes/cron/executions.db` 里 **622 次 suppressed / 6 次 `delivery_outcome=failed` / 0 次送达**；
    今天两次非静默输出（12:10 的 DONE2 报告 11.8 KB、13:43 的 UNREACHABLE 说明 6.4 KB）实录 `failed`，
    作业 `delivery` 字段为 `null` ⇒ **历次 DONE2 报告很可能从未真正送到 leo，只留在 `docs/wiki` + git**
  - 详见 `docs/bm_benchmark_results.md` §0
- 2026-09-28：**BM 修正版结算复核（第三次）—— 服务器数据未变，数字逐位复现，结论不变**
  - 门控 cron 由 `UNREACHABLE → DONE2` 唤醒；对活动数据重跑 `tests/analyze_benchmark.py` / `analyze_bm2_full.py`，
    §1/§2/§3 全部数字**逐位复现**（lm4 `value` 0.7603±0.0088 / `random-matched` 0.7590±0.0047；
    lm5 `value` 0.2565±0.0928 / `random-matched` 0.2839±0.0157）
  - `results/bm2_*` / `bm5b_*` / `bm_bins_*` 最新文件时间戳仍为 **2026-09-14T08:19 UTC**，09-15 后无 bm 相关写入
  - 两个核心问题仍全部不显著：① `value` vs `random-matched`（lm4 +0.0013 / lm5 −0.0274）
    ② `value` vs `value-nofb`（lm5 三 seed 矩阵逐位相同 ⇒ λ_fb 等价死项）
  - 详见 `docs/bm_benchmark_results.md` §0

- 2026-09-27（晚·续七）：**A′ 可辨识 benchmark —— A 类从「不可辨识」到「可辨识」，但 FP 未达标**
  - **目标改写（按 leo）**：不是「给 KeyDoor 加噪声让 A 类通过」，而是
    「构造一个使 novelty 与 dynamics change 可辨识的最小 benchmark」。
    核心原则：**不是为了让 A 类通过，而是为了让 A 类有机会被证伪**
  - **C1 规格**（`docs/a_prime_spec.md`）：给 A1/A2/A3 与**控制量 N** 明确数学定义；
    要求写成可判定的——必须能制造「Novelty ↑ 而 Error 不变」或反向
  - **C2/C3 三环境**（`hibs_lnn/stochastic_keydoor.py`，`eba0b81`）：
    E0 确定性（负对照）/ E1 平稳随机 / **E2 非平稳（slip↑）可用** / FlipKeyDoor 反例。
    机制只写一份（`_perturb`），`step()` 与诊断用 `true_transition()` 共用
  - **★ 我的先验推理被测量推翻**：原以为「翻转分布、支撑集不变」最干净，
    实测相反 —— 翻转版 `N 3.62×`（novelty 拉爆），slip↑ 版 `N 0.88× / A1 3.10×`。
    根因：**决定 novelty 的不是转移的支撑集，而是最终的「去向分布」**；
    agent 行为是状态依赖的，改动力学就改轨迹，改轨迹就改访问分布
  - **顺带修规格偏差**：`rr_agent.py` 的 A1/A2 与 leo 规格**写反了**
    （旧 A1 = 熵，旧 A2 = 1−p̂）→ 对齐为 **A1 = 预测误差 / A2 = 不确定性**；
    新增 **`rr_dyn="N"`（新颖度控制量）**，与 A 族走**同一套管道**
  - **C4/C5 四象限 + 检测指标**（`tests/benchmark_a_prime.py`，`5e160e2`）：
    **主判据必须 n 对齐（S1 → S3，同为熟悉状态）**——不能用 S2 vs S3，
    因为 `n<2 → 返回满值 1.0` 会把 S2 顶到 0.7~0.8 必然压过 S3（饱和分支指纹）
  - **实测（E2，n 对齐 S1 → S3）**：`A1 0.1046 → 0.4550（+0.3504，4.4×）`；
    `N 0.0487 → 0.0291（平坦）`；`A2 +0.1409`（弱于 A1）
    ⇒ **A 类候选第一次在同一环境上表现出不同的量**
  - **检测指标**：`A1 AUC 0.737 / T_detect 400 / 延迟 0 / FN 0`（★ 零延迟）；
    `N AUC 0.232 / T_detect None / FN 1`（检测不到）；**AUC 与「能否触发」不可互替**
  - **三处测量缺陷（我自己写坏的，已修并留指纹）**：阈值用 95 分位 → 被饱和步
    撑到 1.0000 = 指标上限 → **阈值不可达**（指纹：所有通道阈值都是 1.0000、
    `T_detect` 全 `None`）；改 μ+2σ 后仍污染（A1 阈值 1.0680 > 1.0）→ 正解 =
    只从**非饱和步**（`n≥2`）估计并夹到 `[0,1]`
  - **裁定**：门 ①–⑦ **全过**，**门 ⑧ 假阳性 ✘ FP=0.23** ⇒
    **可辨识性成立，但未达可用标准**。按 leo 的 gating 改写进入阶段五，
    **A 类以「辅助证据」身份进入**（`InternalKnowledge 首先是 evidence bank`；
    多 GVF **绝不简单平均**）
  - **未完成**：FP=0.23（唯一未过门）；单 seed（需多 seed）；`T_recover` 未测；
    E1 上的假阳性未单独跑；`n<2 → 1.0` 饱和分支是独立设计问题；**LM4 继续冻结**
  - 报告 `docs/a_prime_decision.md`、`docs/a_prime_spec.md`
  - **另**：**L3 批 25/25 完成**（09:51），机制闸门通过
    `term_reasons = {goal_reached: 95, override: 6, expired: 35}`（L2 为 `1/57`）
  - **容器体检**：SSH `:6100` 通；`/work` 2.3T 可用；**GPU 我这侧全空闲**
    （`--query-compute-apps` 为空，占用者是别的容器）；容器内 CPU 全闲，
    `load 49.86` 是**宿主**的；僵尸进程 PPID=1（`tail -f /dev/null` 永不回收，
    无害）；服务器 `ff070ad` 是本机 HEAD 的**祖先**（不分叉）

- 2026-09-27（晚·续六）：**阶段三生存边界扫描 + 阶段四 A 类否证实验（负结果）**
  - **阶段三**（单因素扫描，7 参数 × 3 场景 × 300 回合，冻结契约不动）：
    七个默认值**全部落在安全区间内**。`rr_forgive_p` 是唯一**双侧窗口**
    （0.0 危险 → 0.02/0.1 安全 → 0.3 危险）；`rr_forget_hl` 是唯一**单调剂量响应**
    （inf 检测 406 → 20 检测 33）；`rr_starve_steps` **不是 horizon 刀口**
    （0.5×–33× 全安全，危险只在 0.17×，下界由「最长正常无奖励间隔」决定）；
    `rr_ucb_c` 无危险区（不敏感旋钮）
  - **阶段四 A：三个候选全部否定**。判据⑨（leo 点名的反熟悉度否证实验）
    实测：A1 `chaos 6/9 / harm 6/9`、A2 `5/9 · 5/9` —— **对两个镜像故障
    触发次数逐位相同**。补充探针取 normal（无故障）误报率：A1 `3/3`、
    A2 `2/3` —— 与故障场景同量级 ⇒ **无场景分辨力**
  - **根因是结构性的**：`KeyDoorMDP.step` **完全确定性**（无随机源）⇒
    实测所有 `n≥2` 的 `(s,a)` **最大熵逐位为 0.000000** ⇒
    转移不确定性(A1)/转移预测误差(A2)/稳定区域(A3) **在数学上坍缩成同一个量
    = 新颖度计数**。leo 预判的「只测熟悉度」被测量证实，而且更强：
    在此环境下**必然**如此，不是「可能」
  - 修了两个真 bug：**A3 方向写反**（返回熟悉度而非压力，读数恒 1.000）；
    A3 硬编码只扫动作 0/1（漏 GRAB/OPEN）
  - **裁定（按 leo 的 §5/§6）**：A 类**不作为主任务 cumulant**（门 6 失败：
    无任何冷启动指标改善）；**不进入阶段五多 GVF 合并**（§5 进入条件未满足）。
    唯一可操作产出：**A 类的可测性前提是一个随机动力学测试台** ——
    确定性环境下 A1/A2/A3 是同一个量，换判据也无用
  - 报告 `docs/rr_survival_boundaries.md`、`docs/dense_cumulant_a.md`

- 2026-09-27（晚·续五）：**三态 epistemic state（好/坏/未知）+ 校准基线冻结契约**
  - **主线重定位（采纳 leo 的判断）**：真正的主线已从「找好势函数」转为
    「**构造一个不会自欺的持续学习系统**」
  - **核心诊断**：`advq == 0` 有两种来源 ——（a）证据充分真无差异 → 好；
    （b）**奖励流为空 Φ 学不到**（harm 整臂全灭时 `Q_adv ≡ 0.0000` 精确零，
    `since_rew` 持续涨）→ **未知**。旧代码 `0 → 不 <0 → 当健康`
    ⇒ **"我不知道"被系统性读成"没问题" = 自欺**
  - **实现**：自欺门 `rr_starved()`（连续 3×horizon 步无正奖励）；三态判定
    `rr_epistemic()`；决策层（未知→有限度探索+**永不算坏**；全局奖励流空→
    **退回安全基元策略**）；报告层当前/历史状态分离
    （`retired_n` / `recovered_n` / `advq_lo` / `advq_hi` / `states{}`）；
    `rr_three_state` 开关（默认 False = 旧行为**逐位不变**，⇒ 三态本身是消融轴）
  - **新验收台** `tests/benchmark_three_state.py`：3 场景 × 3 臂 × 8 指标 × 7 判据。
    **3 seeds × 300 回合 11/11 通过**
  - **最锋利的一组数**：harm 下 `b2-nostate` **整体死亡**（成功率 0.000、
    `steps_go = nan`）且**永不自救**；`b2-3state` 检出（延迟 43）→ 判坏 →
    恢复到 0.633/0.633/0.817。判据② **0.000 → 0.567**。
    **差别不在报告，在行为。** 自欺门违约 0 处；inert 累计误杀 0；
    证据 `advq_n` 三场景单调增长
  - **副产品**：目标 option 的 `advq`，`uniform`（随机选）−0.27 vs 校准 ≈ +0.005。
    **注意口径**：`≈0 → good` = "不比基线差"，**不是**"比基线好"
  - **冻结契约** `docs/calib_baseline_freeze.md`：改它必须是一次显式提交
  - 未做（明写）：阶段三生存参数扫描（默认值一个都没扫过）；
    阶段四三类稠密 cumulant 的无 oracle 对照；阶段五多 GVF 与 `knowledge.py` 合并；
    阶段六知识增长的可测定义；阶段七硬门槛
  - 报告 `docs/three_state_epistemic.md`

- 2026-09-27（晚·续四）：**在线可用势函数 = GVF；退休必须可逆**
  - **为什么 `Φ = max_a Q` 不行**：探针量出 `V(s0)` **逐位等于初值 1.0000**
    而 `γ^k·V(s1)=0.717` ⇒ `Q_adv ≡ −0.275` 是初值 artifact。
    根因是结构：乐观初始化下**只要有一个动作没试过 `max_a` 就是初值**；
    且 option 绕过 Q（靠模型 π_o 导航）⇒ **势函数恰在最需要它的地方没学过**
  - **正解 = GVF**：中性初始化（全 0）+ TD 从**每个转移**学（无论谁出的动作）+
    不用转移模型 ⇒ 无 oracle。三源同条件对比：
    `learned` 7 个值 advq −0.275 / `exact`(oracle) 13 个值 advq **+0.008** /
    **`gvf` 15 个值 advq `+0.0495/+0.0356` —— 符号与 oracle 一致**
  - ★ **退休会切断证据来源 ⇒ 任何判据都自我封闭**（判坏→不用→冻结→永远坏）。
    实测 `Q_adv` 冻在 −0.0059、fallback 0.991、两类故障输出**逐位相同**。
    修法不是调阈值，是加**通路**：`rr_forgive_p` → fallback **0.991→0.397/0.276**，
    inert 故障随即被正确放过。**「犯错是可以原谅的」在机制上 = 退休可逆**
  - 判据改为**「上置信界 < 0」（显著更差）**：`Q_adv = −0.0002` 落在噪声内，
    严格 `<0` 会误杀；且不需要手工 ε（尺度由数据方差给出）
  - ⚠ **GVF 冷启动边界**：harm 下整臂全灭时 `Q_adv ≡ 0.0000`（**账压根没记**）。
    反转 π_o → 到不了终点 → **奖励流为空** → Φ 学不到 → 证据门挡掉一切 →
    **判据永远开不了火**。**势函数是预测，预测需要事件发生；
    agent 失败恰恰意味着事件没发生 ⇒ 信号在最该起作用时不可用**
  - 未做（明写）：换**稠密 cumulant**（`rl_proposer.py` 的 `(c,z)` 层是入口，
    但"哪个稠密量同时满足①在线②无 oracle③与主任务相关"**没有答案**）；
    **报告层三态（好/坏/未知）** —— 现在 `advq=0` 被读成"不坏"，真相是"不知道"
  - 代码全部加性，默认路径逐位不变（7/7 + 两自检 + 哨兵全过）
  - 报告 `docs/online_potential_gvf.md`

- 2026-09-27（晚·续三）：**持续自我校准 —— 持久化 × 遗忘 × 主任务信号**
  - 新旋钮（全部加性，`hl=inf` 时逐位复现旧行为）：`rr_persist` / `rr_forget_hl` /
    `rr_pot` / `rr_retire_rule` / `rr_score`
  - **持久化必须配遗忘**：不遗忘时旧账说「达成率 **98.1%**，几乎完美」而真相是 **48.9% 已坏**；
    检测延迟随遗忘单调（`inf:1201 / 200:291 / 50:79 / **20:31**`）
  - **反直觉**：`persist=False`（清零）检测**最快**（6 = 下限）——
    持久化带着前三段的正面先验，必须先忘掉；对「检测一个变化」而言清零击败携带
  - ★ **只按 option 自己的达成率校准是错的，比没有校准更糟**：
    判别器 = **harm**（反转 π_o，真的有害）vs **inert**（上限压到 1 步，看着坏其实无害）。
    harm 下 `uniform` 成功率 **0.000**（全灭）而校准臂 **1.000**；
    inert 下内部判据淘汰它 → 步数 **11.09→12.13**，主任务判据**不开火** → **11.09 逐位一致**
  - **判据分离**：inert 时内部率崩到 **0.158** 而主任务 adv 仍是 **+0.0412** 为正
  - ⚠ **在线势函数不可用**：`learned`（自己的 Q）早期乐观常数 → `adv≈-0.004` 全噪声 →
    全判坏 → fallback 1.000 → 性能崩；`exact`（模型 VI）正常。
    **结构对，缺的是在线可用的势函数** —— 这正是「持续自我校准」最终要自己解决的事
  - 事前判据两类故障各 **6/6**；判据本身被实测修正三次
  - 报告 `docs/self_calibration_persistence.md`

- 2026-09-27（晚·续二）：**自我校准根源诊断 + 最小闭环落地**
  - **诊断**：全仓 `option 层 TD/价值学习 → 0 命中`；`prune/remove/drop/淘汰 → 0 命中`；
    `_rr_settle(self, st, ...)` **收下 `st` 然后完全忽略它**
  - 结论：**整个栈只有一层会因为「自己错了」而改变自己**（基元 Q 表的 TD 误差）——
    现在是「被重新生成」，不是「持续自我校准」
  - **最锋利证据**：L2 的 57 启动 / 56 expired / **1 reached**，而系统无处安放这个事实
  - **信号方向也是反的**：`rr-exact` 9.4 步 vs `rr-zeroV` 9.0 步 —— 价值被接到
    β_o（何时停）而不是「值不值得留」⇒ 用价值去**终止**而不是去**纠正**
  - **补上轴 B**（缺的那条消融轴）：`rr_select_rule="calibrated"` ——
    只用**自己的实测达成率**排序（UCB1），坏的自己点名并排除，全坏则**退回基元层**
  - `tests/test_self_calib.py` **7/7**：`uniform` 坏 option 占比 0.825→0.842（错误持续）；
    `calibrated` 0.107→**0.015**（0.138），主任务 0.840→0.970（不更差），退回基元层 620→1930 次
  - **判据本身被实测修正两次**：第一次**打错了臂**；`reached==0` **连 L2 真案都抓不到**
    （0.018 不是 0）→ 改 `< 0.5`（依据=实测两簇间的空档）
  - **结构性发现**：坏 option 在 8/32 个状态上是**唯一**可启动的 ⇒
    **校准的作用范围受 `initiation_ok` 限制**，规则再聪明也无从避开
  - 报告 `docs/self_calibration_root_cause.md`

- 2026-09-27（晚·续）：**OaK 第一步落地 —— SubTask → Option 层实现 + 三方向消融**
  - **驱动层换回论文的 SubTask**（此前从瓶颈/介数/最短路反推 = 论文点名的最差类别）
  - 新增 `hibs_lnn/rr_options.py` + `hibs_lnn/rr_agent.py`（**子类**，primitive 路径逐字委托 ⇒ 对照逐位干净）
  - **机制存活**：`β_o` 非空（`|β|=33`）、闭环（`recomputes == exec_steps == 210`）、
    终止台账 `{beta_at_feature: 43}` 且 **`expired: 0`**（对照 L2 的 56/57 expired）、lock-in 0.16
  - **规划指标**：`rr-exact` 四 regime 全 ≤ primitive（0.333~0.667）；
    **`bottleneck` 1.250 逐位命中论文的 2145/1716 = 1.25**
  - ⚠ **T_adapt 在 KeyDoor 上饱和**（所有 option 臂贴窗口下限 20，连瓶颈臂也是）——
    真因是 **KeyDoor 没有代价区**（论文负向需要「最短路穿过代价区」+「存在绕行」两个前提）。
    ⇒ **T_adapt 不能作主指标，此前以它为头条的 option 结论受此污染**；改用 steps-to-goal
  - ⚠ **价值函数假设得到反向证据**：`rr-zeroV`（z 不用 V_main）9.0 步 **优于** `rr-exact` 9.4 步 ——
    `V_main` 让 `β_o` 提前触发，把控制权过早交还给尚未训练好的 base policy
  - **剂量响应 + 内部哨兵逐位通过**（`opt_prob=0` ≡ primitive）；ΔS 随剂量单调
  - **三个方向在论文 gridworld**：两房间 ①✓1.250 ②✓0.625 ③✗；四房间 ①✓1.099 ②✓0.536 **③✓0.828**
    ⇒ **“目标选得好”只在结构足够复杂的环境里成立**，两房间这个台子分不开
  - 报告 `docs/rr_subtask_ablation.md`；本轮抓到 4 个新真 bug（含 `trans` 记录用步后状态 —— 哨兵判据的回报）

- 2026-09-27（晚）：**L2 结算（H2 被证实）+ 修复版 B 矩阵结算 + `terminated()` 接入区域判据 + L3 起跑**
  - **① L2（rounds=300，25 臂，闸门全过）**：`e9 0.7827 / fixed 0.7570 / goal 0.7660 / goal_term 0.7652 / override 0.7677`
    - **★ O2（闭环）vs O1（开环）Δ=+0.0090 t=+2.19 p=0.0286 显著** —— K1v2 时是 Δ=+0.0029 **p=0.3287 不显著**
      → **用户假设 H2「样本量不足」被证实**（不是被排除）
    - 但**所有 option 档仍显著差于 e9**（Δ=−0.0149…−0.0257，全部 p≤0.0001）
    - option 档排序**符合理论预测**：`override > goal_term ≈ goal > fixed`（开环最差）
    - `term_reasons` 依旧：`goal` 57 次启动 **56 次 expired、仅 1 次 goal_reached**
  - **② 修复版 B 矩阵结算**（KeyDoor，隔离副本，3 seed × 400 回合）：
    复用 `primitive 1.82x / rediscover 1.17x / macro 1.08x`
    - **修复确有效**：rediscover 复用 **0.98x → 1.17x**，永不收敛段 **10/18 → 6/18**；macro 台账活了（21.3%/7.9%/26.4%）
    - ★ **对照干净**：修复版 `primitive` 臂与老基线**逐位相同** → 改动只影响 option 臂
    - **但无任何 option 模式优于 primitive**（1.82x、0 段不收敛、成功率全 1.0）；macro seed43 甚至 6/6 全灭
  - **③ `terminated()` 接入区域判据（关键）**：上一提交只改了 `SkillAgent._macro_step`，但 **lm4 管线不走那里** ——
    `oak_proposer.py:187` 直接调用 `o.terminated(...)`。L2 因此仍在用旧距离判据（= 57 次启动仅 1 次 reached 的原因）。
    本提交把 `in_goal_region` 接入 `terminated()` **主判据**（距离判据保留兜底，未设 regions 时逐字不变）。
    单元验证逐项通过；真实链路 macro 台账 **50.3%**（修复前 14.4%）
  - **④ L3 批起跑**（`tests/oak_longv3.sh`，25 臂，GPU0）：与 L2 **逐字相同**，唯一变量 = 区域判据。
    启动前硬闸门新增两条（缺 `in_goal_region` / `terminated()` 未接线 → **拒绝启动**）
    - 事前判据：① `goal_reached` 显著 > 1 次 ② 若 Δ 仍显著为负 → **终止判据被排除** ③ 剩余症结只能来自**目标选取/发现方法**

- 2026-09-27：**GPU 容器故障闭环 + option 三处结构性缺陷（死账／够不着的目标／错的瓶颈统计量）**
  - **① 容器设备故障已解除。** 运行期对主设备号 195 整段拒绝（`/dev/null/zero/random/ptmx` 放行、仅 nvidia 全 EPERM；
    自建 major 195 minor 0–255 **全 EPERM** = cgroup eBPF 设备白名单特征）；容器内无 `docker.sock`、有 MKNOD 无 SYS_ADMIN/BPF
    → **无法自救**。宿主侧 `docker restart`（非重建：保留可写层，`/root`、SSH key 全在）后验证 `torch.cuda.is_available()=True`、
    `device_count=2`、数据 22M 全在。诊断链两处自我更正：✗「驱动库缺失」查错了路径；✗「没带 `--gpus`」被 PID1 environ 推翻。
  - **② B 矩阵校准（KeyDoorMDP，全 CPU）：400 回合修好「0 成功率」。** primitive 3 seed × 6 段**全部收敛、成功率 1.0**；
    而 **`rediscover` 三段永不收敛（T_adapt 打满 400）、另两段成功率仅 0.2/0.25** → 43 个 option 的技能库在三段上直接摧毁学习。
  - **③ option 三处结构性缺陷（已修，`d1be29d`）**
    - **(a) 死账**：`Option.visits/success` **从来没有任何地方更新过** → `select()` 的 `rate=success/visits` 恒取默认 0.5
      → 54~60 个 option 在启动集内**同分**，选择等于抛硬币
    - **(b) 目标用 k-means 质心**：质心可能不对应任何真实状态 → `π_o` 朝「够不着」的点走
      （实测 macro 执行成功率仅 **14.4%** = 85.6% 到不了自己的目标）
    - **(c) 瓶颈检测用介数中心性**：走廊的语义是「去掉就断开」= **割点**，不是「最短路径经过次数」
      （FM-17 实测：介数最高的格**不是**走廊格）
  - **修复**：新增 `macro` 模式（option 真正当**多步宏动作**执行到 `β_o`，每步闭环重算 `π_o`，终止时结算成败）
    + `_snap()`（起点/目标吸附到最近真实状态）+ `_articulation_points()`（Tarjan；无割点时退回「出边最多」）
    + `term_eps` 改取 `om.init_radius`（**第 5 次「阈值与状态空间尺度不匹配」同型错误**）
  - **验证**：割点检测单元测试 ✓（链图→{1,2,3}／环图→{}／星图→{0}）；三判据自检全过；
    效果（单 seed）：macro 成功率 14.4%→**24.2%**、macro 复用 0.83x→**1.29x**、rediscover 复用 1.28x→**1.52x**
  - **诚实标注**：目前**没有任何模式优于 primitive**（1.34x）的证据；multi-seed 对照待 B 矩阵校准跑完后进行。
    两条实验线在跑：L2 批（GPU，rounds=300，25 臂 × ~6 分 ≈ 2.5h）、B 矩阵校准（CPU，400 回合 × 3 seed）

- 2026-09-27：**β_o 的终止判据在离散状态空间里退化（K1v2「option 从未到达目标」的同源根因）**
  - **真因（逐次执行诊断）**：`Option.terminated()` 用 `‖s−goal_center‖ < eps`，eps = `init_radius` = **0.754**；
    而 KeyDoor 的 `vec()` 是 one-hot ⊕ 两个 bit，**距离离散化**：0.000／1.000／1.732／2.000
    → `0.754` 恰好卡在 **0 与 1.0 之间 → 只接受精确到达**
  - **2737 次执行实测**：起点距离中位 **1.732**，执行中最小距离中位 **1.000**（可达最小值就是 1.000）；
    **变近 60.2%／没变 39.8%／变远 0.0%**；β_o 因达成触发 **0/2737**（全靠预算耗尽退出）
    → **π_o 是正常工作的，坏的是终止判据**。K1v2 归因给「目标是任意远端状态」**只对了一半**
  - **修复**：`Option` 增加 k-means Voronoi **区域归属** —— `in_goal_region(s)` 作为 β_o 主判据
    （无阈值、覆盖全空间、离散状态空间不退化）；`initiable` 同样优先用区域归属
  - **验证**：三判据自检全过（macro 台账 406/1718）；复用 macro 1.29x→**1.37x**；
    ★ **对照干净** —— 修复版 B 矩阵的 `primitive` 臂与老基线**逐位相同**
  - **诚实标注**：**仍无任何模式优于 primitive 的证据**（macro 1.37x vs primitive 1.34x ≈ 打平）。
    multi-seed 对照 `bm_fixed`（3 seed × 400 回合，隔离副本避免污染运行中的 L2）已起跑
  - **方法论坑**：诊断脚本手写复刻 `run_episode` 循环 → 忘 `q.update`／接收 `done` → 只访问 3 个状态 → **零样本**
    → **诊断必须观测真实路径（子类挂钩），不能复刻**

- 2026-09-16：**K1v2 有效测量（修好管线后重跑）+ 四房间 Fig.6 首次复现 + 环境/自检三处缺陷修复**
  - **① K1v2：option 执行模式的第一次有效测量。** 修 `observe_action` 接线后重跑 25 臂（3.0 分/臂匀速 → 确在 GPU 上完成）。
    闭环档这次真的执行：`goal` 的 `starts` **1→21**、`steps` **0→84**、`replan=84`（旧版是 1 次启动跑满 87% 的锁死）
  - **四个 option 档全部显著差于无 option**（Δ=−0.0072…−0.0111，p=0.0003…0.0055），
    且 **O2 闭环 vs O1 开环 Δ=+0.0029 p=0.3287 不显著 → 排除「问题是开环执行」**
  - **机制在 `term_reasons` 里**：`goal` 21 次启动 **20 次 `expired`**、`steps/starts ≈ 4 = max_opt_len`、
    5 seed 合计 `goal_reached` 只 1 次 → **option 从未到达过目标**，因为目标是瓶颈中心性给的任意远端状态
  - **② 四房间复现不了的真因找到了**：旧 `FOUR_ROOM` 行长 12/13/14 且含**空格** → `.ljust` 静默补墙 +
    空格变**幽灵地板**。新增 `validate_layout()` 构造期硬报错 → 立刻又抓到 **`TWO_ROOM` 同样行长不齐**
    → **已「复现成功」的两房间结果也建在被静默改过的几何上**
  - **③ `check_two_room` 自身有 bug**：hall 候选没排除灰色格 → 「绕开灰色区到达 hall」**构造上不可能为真**
    → 该自检从写出来就永远报 ✗。修掉后测出旧两房间**三条性质一条都没满足**（col 1 是通的，自由竖井绕过整个灰色区）
  - **④ 修复并重验**：两房间重跑 primitive 1400 / shortest 1750 (1.250) / reward-respecting 875 (0.625) —— 方向不变；
    **四房间随机动力学首次跑通** 48872 / 53710 (1.099) / 26200 (0.536) → **Fig.1 与 Fig.6 两个方向都成立**
  - **⑤ GPU 再次掉线（当前阻塞）**：根因 `head -c 16 /dev/nvidiactl → Operation not permitted` =
    **容器设备 cgroup 拒绝访问**，只能由宿主侧重启容器修复
  - 详见 `docs/oak_repro_ground_truth.md`（七处 bug 台账化）

- 2026-09-16：**BM 修正版结算复核 —— 数字逐位复现，结论不变（价值函数仍测不出效应）**
  - 对活动服务器数据重跑 `tests/analyze_benchmark.py` / `analyze_bm2_full.py`：§1/§2/§3 全部数字**逐位复现**
    （lm4 `random-matched` 主口径 0.7635 / 0.7542 / 0.7594 与 `results/bm2.log` 逐位相同；日志同时确认候选池 60）
  - 补全原表遗漏 3 条：LM4 `value` vs `perm` any-time **+4.47 (p=0.012)**、`fixed` vs `perm` replay **+4.74 (p=0.018)**
    → **所有显著的对比指向的都是「在线调度 > 预先定死的任务流」，没有一条指向价值函数**
  - 核心两问仍不显著：① `value` vs `random-matched` ② `value` vs `value-nofb`（lm5 逐位相同，λ_fb ≈ 死项）
  - 详见 `docs/bm_benchmark_results.md` §0/§3

- 2026-09-14：**OaK 正式对照结果 —— Options 显著有害（诚实负面结论，80 臂双 GPU）**
  - **E9 vs E10 唯一变量是时间抽象**：lm4 any_time **−1.6%（t=−5.02, p<0.0001）**、最终 acc **−3.7%**
    （p<0.0001）、naive 臂 **−15.5%**；lm5 长配置 any_time **−1.9%（t=−5.14, p<0.0001）**
  - **机制确实生效**（option_starts 41.9/75.2，steps 62.3/82.8）→ **不能按「机制没触发」免责**
  - regime shift（E11/E12）方向一致但不显著（p=0.22/0.31）—— 对「Options 在非平稳环境才有价值」是反证据
  - 步长 E 表四算法几乎无差别（any_time 差 ~0.4%），与此前三条独立负面结论一致
  - **修两个统计 bug**：`OAKProposer.stats()` 覆盖 base 真 α 统计（值取自未被更新的死对象）；
    `analyze_oak` 只读 lm4 的文件名导致 lm5 的 20 臂完全没被读到
  - 诊断队列（D1 opt_frac 剂量-反应 / D2 refresh 频率 / D3 修好的 α）在跑
  - 详见 `docs/oak_results.md`

- 2026-09-14：**OAKProposer 接入 lm4 主回路 + E9/E10 对照启动（本轮最大缺口已补上）**
  - `hibs_lnn/oak_proposer.py`：state = 粗域能力画像、action = 选哪个细区间训练、reward = Δ(any-time)
    —— **`T(s,a)→s'` 是真可学、数据里天然存在的转移**
  - **E9 vs E10 只差 `--oak-options`**：同一 base、同一门控，唯一变量是时间抽象
  - 修 5 个 bug：签名漏接（`str.replace` 静默 no-op + 缺 assert）、知识层维度混用、
    **门控语义错误（零覆盖重定向导致 `coverage=[0,9,0,0,0,0]` 探索崩塌）**、
    `Option.stats()` 不暴露 actions、option 启动不执行第一个动作
  - 服务器冒烟：`n_trans=72 options_found=6`，序列 `[0,5]/[0,5,1]/[5,1,5]`，acc 0.7325
  - `tests/oak_driver.sh` 5seed×{E9,E10} 已在 GPU0 运行；完成时服务器 watcher 自动跑 `analyze_oak.py`

- 2026-09-14：**InternalKnowledge 内部知识层（Knowledge ≠ Parameter）**
  - 新建 `hibs_lnn/knowledge.py`：四层知识 + 三类元知识；**不暴露 θ/W/α，只暴露知识查询**
  - `knowledge_of(s,a)` 给出三种判定「我知道 / 不太确定 / 我不知道」——
    **对未见的 (s,a) 承认识不知道，而不是编一个数**
  - `to_dict()` + `growth()`：知识能留下来，且增长**逐项可测**（coverage/GVF/dynamics/β 的变化量）
  - **实测两件事**：(a) 知识层必须对未知输入容错（修 `value()` + `verdict` 字段）；
    (b) **把 α 的更新从学习里孤立出来是测试方法错误** —— `g = δ²/φ` 恒正 → β 单向漂移，
    8 种稳定化手段全部撞界；β 能停的唯一原因是 δ 随 w 收敛趋 0，故必须**闭环**测
  - 闭环结果：α[真特征] > α[噪声] ✓、β 全程有界 ✓，但分化**真实而弱**（ratio 1.22）

- 2026-09-14：**OaK 结构件三件套：Option manager + 反事实 rollout + coverage/uncertainty 门控**
  - 按用户指点**停止堆 RL 算法**（会变成"算法动物园"），转而补 OaK 的结构性组件：
    `Prediction → Abstraction → Options → World Model → Planning`
  - 新建 `hibs_lnn/uncertainty_gate.py`：`TransitionEnsemble`（bootstrap 集成分歧估 `U_T`）+
    `UncertaintyGate`（`allow = [C(s,a)>τ_C] ∧ [U_T<τ_U]`）+ `rollout`（**一遇不可信即中止**，不硬外推）
  - 新建 `hibs_lnn/option_manager.py`：`Option=(I_o, π_o, β_o)`，**option 从转移结构发现**
    （聚类→建图→筛可靠边→找多步路径），`β_o` 是**学习式终止**而非硬阈值
  - **`tests/test_oak_structure.py` 四测项全过**：① 发现 6 个长度≥2 的 option 且**自动避开高噪声动作**；
    ①b 远离质心 4/4 预测正确（回归守卫）；② 零覆盖被挡 + U_T 区分高低噪声；③ rollout 遇零覆盖即中止；
    ④ P(终止) 随接近目标单调上升（β 在 `g−s` 上权重 = −2.10，符号正确）
  - **修掉 4 个真 bug**：特征冗余常数列（条件数 1.24e16、预测全错）、onehot 与截距共线
    （远离质心 Δ=[92.7,96.4,25.8,23.7]，真值 0.3/0/0/0）、fit/predict 的「Δ vs 绝对状态」约定不一致
    （质心处巧合相等而掩盖）、夹具三次尺度错配
  - 详见 `docs/oak_structure.md`

- 2026-09-14：**分离 δ^pred 与 δ^RL 两条学习信号 + 覆盖度门控（Q15/Q8）**
  - Q15 要求「两个不要混成一个东西」—— 此前 `RLProposer` 只有 reward=Δ(any-time)，没有独立预测通路
  - 新建 `hibs_lnn/dual_proposer.py` `DualSignalProposer`：
    **通路 P**（world model，`W_p` 只由 `δ^pred` 更新）／**通路 V**（value，`θ_v/α/h` 只由 `δ^RL` 更新，IDBD 步长）；
    预测输出只作为 `φ_v` 的一个分量进入价值通路
  - **覆盖度门控（Q8）**：`g=1/√(1+n)`，预测按 g 缩放；`counterfactual()` 对未覆盖候选报不可信
  - **验证三项全过**：① 信号隔离（只喂一路时另一路参数变化 = **0.000000000000**）
    ② 门控生效（未访问 trustworthy=False）③ **规划幻觉对策**：零覆盖候选的预测值反而最高（0.6450 vs 0.5721），
    门控挡住它 —— 实测对照 0.9582 预测 / 0.4357 真实
  - 提交 `25158a9`

- 2026-09-14：**步长自适应（IDBD / Autostep / Continual-IDBD）—— 最终测不出效应（诚实负面结论）**
  - 建 `hibs_lnn/rl_proposer.py` 五种步长算法 + **内部知识注入**（`--rl-know 14`，φ 9→23 维）
  - **主结论（n=11）**：`idbd-raw` vs `value` 最差遗忘界 Δ=+0.0026（p=0.965）；
    IDBD 系 vs value 系 Δ=−6.2%（**p=0.564**）；any_time p=0.53 —— **全部不显著**
  - **★ 自我纠错**：n=3–5 时看到的「−24% 一致方向」与 **p=0.0049** 是**池化不同代码版本**的假象，
    补 seed 后效应消失，**原结论作废**
  - **分化机制（真知识）**：`h` 在 Autostep 里分化**比 IDBD 更强**（1.16 vs 0.45）→ 不是特征不可分；
    alpha 不分的元凶是**逐分量归一化把指数卡死在 μ**（`|e|max ≡ μ`）
  - **Regime-shift**：autostep 最稳（MSE_B 0.564），我的 cidbd 最差（8.640，α 顶死 + recovery 触发 311 次）
  - **修 6 个 bug**，其中两个是诊断性缺陷：driver 里 `$(date)` 先执行会**重置 `$?`** → 失败的 run 报 `rc=0`；
    以及本轮第二次「参数加了但调用点没接」
  - **alberta-framework 装不了**（要求 Python ≥ 3.13，当前 3.11）
  - 文档 `docs/stepsize_idbd_results.md`

- 2026-09-14：**真 RL 提议器 + IDBD/Autostep 步长自适应 —— 分化成立但无收益（诚实负面结论）**
  - 用户要求「动作→环境→奖励→参数更新」的真闭环 + 「步长为主导的 policy 积累」+「redefine 也要探索 policy 的积累」
  - **新建** `hibs_lnn/rl_proposer.py`（θ 可学；reward = **Δ any-time 准确率**；`h` 信用迹累积；
    **α 本身是累积参数**；`redefine()` 按物理描述子最近邻**继承** explore/α/θ）
    + `hibs_lnn/transition_model.py`（OaK 第③条：`T(a_t,u_t)→Δa` + beam-search planning）
    + `docs/oak_alignment.md`
  - **论文原始基准（weight-flipping）**：IDBD 分化 4/8 且 **3/8 档发散（MSE 1.6e9）**；
    Autostep **6/8 且 0/8 发散** —— 与 Degris 2024 / Mahmood 2012 所述一致
  - **真实回路（lm4，n=3）**：any-time — value **0.7702** > idbd 0.7659 > auto 0.7596 > auto2 0.7575，
    **全部两两 t 检验不显著**（idbd−value p=0.82）→ **步长不是本任务主瓶颈**
  - **反直觉**：真实回路里 IDBD 分化强（α_std 0.2357 / ratio 3.92），**Autostep 几乎不分化**
    （0.0004 / 1.04）← 与 weight-flipping 相反；推测 Autostep 的 `α/=M` 均匀压缩在 fdim=9 下压平了 ratio
  - **唯一有信号的项**：最差遗忘界 — idbd 0.2899±0.0179 vs value 0.3808±0.0606（−24%，t=−2.49，p=0.112），
    签名同 V35.22「稳定器」；**seed 2–6 扩展已在跑**
  - **修 4 个 bug**：① `--rl-algo/mu/alpha0` 调用点未接（四臂逐位相同，对照作废）；② JSON 非原子写 +
    无 numpy 兜底 → **损坏半截文件**；③ IDBD 式误删 `x`；④ `random-matched` 对照臂因此首次全废
  - 文档 `docs/stepsize_adaptation_results.md`、`docs/oak_alignment.md`

- 2026-09-14：**修正版 benchmark 结算 —— 价值函数四项全活后依然不敌频率对齐对照（诚实负面结论）**
  - **先作废一批**：`docs/bm_benchmark_results.md` 原记录的 `bm_*` 结果**全部作废** —— 那批的价值函数是半成品
    （`con` 未实现 / `sim≡1` / `fb≡1`，实跑退化成「均匀化采样器」）
  - **修正版**（`results/bm2_*` lm4 + `results/bm5b_*` lm5，con 连续强度 / sim σ 自适应 / 池 18→60 / 在线逐轮提案，
    `value_cv` 0.089→0.4711，四项 `term_std`>0）× 3 seed：
    - **① `value` vs `random-matched`（唯一干净对照）**：lm4 replay Δ=**+0.0013**（t=+0.23，p=0.84）、
      平均遗忘反而**多 0.028**；lm5 末轮均值 Δ=**−0.0274**（t=−0.50，p=0.66）、any-time Δ=+0.0073（t=+0.35）
      → **两个 benchmark 符号相反 ⇒ 只能判「测不出效应」，不能判「有效」**
    - **② `value` vs `value-nofb`（隔离真实反馈项 λ_fb）**：lm4 提议序列**已发散**（修复生效）但性能
      Δ=−0.0024（t=−0.41）/ any-time Δ=+0.0015（t=+0.09）不显著；lm5 **3 seed 遗忘矩阵逐位相同**
      → `λ_fb` 在 lm5 对调度**零影响**（等价死项）
  - **「方差稳定器」再次被推翻**：lm5 `value` 的 CV **36.2%** 为全场最大（`random-matched` 5.5%、`perm` 7.9%）；
    lm4 `value` CV 1.2% 也没小于 `random-matched` 0.6%
  - **lm5 负 Δ 的具体机制**（不是玄学）：`value` 24 次提议只点了 **2 次** `causal*` 域（seed 1 一次没点），
    因果系四域占末轮均值 4/8 → **覆盖偏置**，`cov=1/(1+freq)` 在 8 域 × 8 轮的池子上不足以强制覆盖
  - 仍能站住的只有 **「在线调度 > 预先定死的任务流」**（lm4 `value` vs `perm` Δ=+0.0210，t=+3.27，p=0.034），
    且 `random` / `random-matched` 同样赢 `perm` → **红利属于「调度」这件事，不属于价值函数的判别力**
  - **顺带挖出真 bug 并恢复数据**：lm4 `random-matched` 3/3 `rc=1` —— `run_lm4_wave.py` 的 `json.dump` 缺
    `default=` 转换器，`int64` 触发 `TypeError` 使 **dump 到一半崩掉**，json 截断成非法文件，
    `any-time` / 最差遗忘界**永久丢失**（`report.md` 在 dump 之前写，故准确率没丢）。
    用 monkeypatch 序列化器重跑 seed 42 得 0.7635，与 driver 日志**逐位相同** → 同修订版内完全可复现
  - **并发干扰（已记录）**：服务器 `tests/run_lm4_wave.py` 被**另一路会话**在 08:14 / 08:19 UTC 改了两次并重跑
    `random-matched` 的 seed 1/7 → 该臂主口径取「与 `value` 同修订版」的日志值，另报敏感性
    （两种读法 Δ=−0.0009 / −0.0054，**都远不显著**，结论不随修订版漂移）
  - 报告 `docs/bm_benchmark_results.md`；脚本 `tests/analyze_benchmark.py`（lm4 全臂）/ `tests/analyze_bm2_full.py`（lm4+lm5）

~~- 2026-09-14：**Benchmark 结算 —— 价值函数未能优于频率对齐对照（负面结论）**
  - lm4 + lm5 新 benchmark 全臂（8 + 7 臂 × 3 seed）结算，报告 `docs/bm_benchmark_results.md`，脚本 `tests/analyze_bm_full.py`
  - **唯一干净对照 `value` vs `random-matched`**：lm4 replay Δ=−0.0045（t=−0.32）/ any-time Δ=+0.0125（t=+0.47）
    —— **不显著且数值略低**；lm4 **最差遗忘界 Δ=+0.1509（t=+2.22）显著更差**；
    lm5 replay Δ=+0.0030（t=+0.07）不显著、any-time Δ=+0.0361（t=+2.06）边缘显著更好、最差遗忘界不显著
    → **lm4 与 lm5 方向互相矛盾，只能判「测不出稳定效应」**
  - **连混淆对照也赢不了**：`value` vs `random`（均匀随机）lm4 replay Δ=−0.0197 → 表现不佳**不是**频率分布差异造成的假象
  - **`λ_fb` 是死代码（真 bug）**：`value-nofb` 与 `value` 逐位相同（三指标 Δ=+0.0000 / t=+0.00，提议频率向量完全一致）
    → 「真实反馈 = 1−acc」对调度**零影响**，此前把 value 臂读作「含反馈的有效性」不成立，必须修
  - **「方差稳定器」说法推翻**：干净配置下 lm4 value replay CV 2.0% 并未小于 random（1.7%）；
    lm5 value CV 25.6% 反而全场最大（random-matched 5.3%）
  - 唯一站得住的是 **「在线调度 > 预先定死的任务流」**（`value` vs `perm`：replay t=+3.44 / any-time t=+4.31 显著），
    但 `random` / `random-matched` 同样具备该优势 → **是「调度」的功劳，不是「价值函数」的功劳**
  - 顺带修 lm4 `random-matched` 首次 3/3 `rc=1` 的 profile 加载 bug（driver 把嵌套结果文件整拷作 profile）
~~（⚠️ **作废**：该批价值函数是半成品，见上条修正版结算）~~
- 2026-09-14：**LM4 骨干替换完成 + 价值函数接入**（本日为最大一次推进）
  - **骨干替换四轴全胜**：StatMLP（窗口统计 + MLP + **冻结归一化**）vs 复值 SSM
    - joint（6 域）0.8104 vs 0.7213（+0.089）
    - replay 8 seed **0.7297±0.0038（CV 0.5%）** vs SSM 0.6126±0.0180（CV 2.9%）
    - 单 run 耗时 ~0.7 min vs ~13 min
    - 3/6/12 域全部追平可达上限（0.9151 / 0.8104 / 0.6579）
  - **真凶 = BatchNorm running statistics 漂移**（持续学习隐性杀手）：batchnorm 0.2617/32% → **fixed 0.7308/90.2%**。这一个 bug **同时伪装成三个结论**（「OML 不适合 CL」/「PLN 头是元凶」/「天花板vs保留率是架构权衡」），全部作废
  - **干净配置 CL 方法矩阵**：PLN 0.7287 / SwiftTD 0.7365 / 线性 0.7368（无差异）；**OML2+PLN 0.7401 最优**；replay 强度几乎不影响
  - **跨年泛化**（2015 训 → 2016 测）：域内 0.7400 → 跨年 **0.591**（naive 0.183 ≈ 随机）
  - **因果线闭环**：发现 20 稳定边（**独立恢复出偶极场定律 `logB—logL`**）+ `do()` 干预 + Pearl 三步反事实
  - **价值函数接入**（`hibs_lnn/value_proposer.py`，从 LMT-twister V35.19 移植）：首轮 fixed 0.7300 > random 0.6360 > value 0.6172（固定序仍胜），但 **naive 臂 value 0.3700 vs fixed 0.1762**（提议器降低域间干扰 +0.20）
  - 文档：`docs/lm4_backbone_final_verdict.md` / `lm4_clean_cl_matrix.md` / `lm4_ssm_bottleneck_verdict.md` / `lm4_unified_cl_design.md`
- 2026-09-14：**LM5 多模态（文本 + 电磁波 + 因果）**
  - `tests/run_lm5_mm.py`：共享主干 + 模态 stem/head，输出跨模态遗忘矩阵；因果域用 `S4WorldSCM` + 观测/干预双标签
  - **核心结论：骨干选择是模态依赖的**（3 seed）—— 文本 SSM 赢（en/code 各 +0.125）/ 电磁波 MLP 赢（+0.065）/ 因果域无差别（差 ≤0.005，两架构撞同一堵墙 = 任务瓶颈）
  - 混合骨干 `MultiModalHybrid`：总平均 0.3133（微胜 SSM 0.3117），但**未达两边最优**（wave 0.4583 < 纯 MLP 0.5256，疑似梯度干扰）
  - **指标修复**：`y_do` 多数类基线 0.789 使原始准确率失真 → 新增平衡准确率（宏平均召回）；SCM 干预饱和已修为均匀覆盖（0.789→0.508）
  - 待办：lm5 的价值函数调度只有冒烟验证，未跑正式对照
- 2026-09-11：LM4 特征升级 + 两天队列结算
  - WFR 波功率谱接入（4→30 维）joint 0.4354 → **0.7213**（救活 D2/D3，原为特征信息不足）
  - 挖出 MAG `-100000` 哨兵污染（非文档化填充值被 `abs()` 洗成正值极值）
  - 18 run 结算：`agg-path` 未拉高天花板但**稳定性显著**（CV 10.6% → 3.0%）
- 2026-09-10：LM4 电磁波持续学习启动（NASA CDAWeb RBSP-A EMFISIS 真实数据）
- 2026-09-08：LM1 GPU 训练支持 + 8M 档首次跑测（2×A100-40GB）
- 2026-09-06：LM1 整合 Replay + OML + 因果学习（`--proposer {value,causal,causal_rl}`）

## 进行中的实验

- **lm4 价值函数 3 臂 × 6 seed**（`vp2_*`，18 run）：验证「提议器降低 naive 遗忘」的 +0.20 效应是否可复现
- **lm4 因果发现可扩展到干预效应估计**：当前 `causal_intervene.py` 的 DAG 为物理先验手工定向，缺太阳风/地磁/MLT 混杂
- **已完成（负面）**：lm5 价值函数正式 3 seed 对照 —— 见上，干净对照下测不出稳定增益
- 待办：**修 lm4 `json.dump` 的 int64 序列化 bug**（`default=lambda o: o.item() if hasattr(o,'item') else str(o)`）—— 已造成一个臂的 `any-time`/最差遗忘界永久丢失，且 `rc=1` 被 driver 当普通失败记录，极易漏看
- 待办：**`λ_fb` 在 lm5 仍是死项**（3 seed 遗忘矩阵逐位相同）；lm4 已发散但无性能贡献 → 反馈闭环整体仍无价值
- 待办：**`random` 臂的提议不随 seed 变**（三 seed replay 0.7731/0.7732/0.7732，CV 0.0%）—— 需查是否漏用 seed 种子化
- 待办：**lm5 `value` 的覆盖偏置**（因果系四域只被点 2/24 次）—— `cov` 项在 8 域 × 8 轮池上不足以强制覆盖
- 待办：提高 seed 数（现 `n=3`，`|t|≈2` 只能算边缘证据，ddof 口径一变结论就翻转）
- 待办：**lm5 补 `random`（均匀随机）臂** —— 现缺该臂，无法在 lm5 上复现混淆对照
- 待办：新 benchmark 全臂开启 `--trace-every`（学习效率/达标步数指标仍缺）
- 待办：lm5 混合骨干 wave 退化归因（是否 SSM 文本通路梯度干扰 MLP 波通路）

## 已知的执行缺口（必须默认开启，不能只在单次 run 里用）

- **cron 交付通道（2026-09-28 查出）**：`lm4-lm5-benchmark-done-gate`（`bff2174e8bef`）的 `delivery` 为 `null`
  → 所有非静默输出记为 `delivery_outcome=failed`（6/6，0 次送达）。**报告只落在 `docs/wiki` + git**；
  修好之前，不要假设 leo 看过任何本作业的报告
- **学习效率曲线**：`--trace-every` 在 105 个 result tag 中仅 1 个开启过 → 后续所有实验**默认开启**
- **类不平衡指标**：任何类不平衡任务必须**同时**报原始准确率 + 平衡准确率
- **push 后必须更新本 wiki**（log.md 追加 + 本文件同步 + `wiki_check.py` / `wiki_lint.py --strict=v2` / `raw_manifest_check.py` 全绿）

## 方法论沉淀（本日新得，已写入 `ml-ablation-methodology` 技能）

1. **一个实现 bug 可以伪装成多个科学结论** —— BN 漂移在全部配置里生效，因此同时污染了方法对比/头对比/架构对比
2. **反直觉的好结果必须当 bug 追** —— 本日三次：naive 跨年 1.000、BN 漂移、跨年指标设计错误
3. **排除一个假说后必须继续找真因** —— 容量扫描否定容量假说后停手，错过了真因是 BN
4. **架构结论不跨模态迁移** —— 应做按模态分派，每模态各自验证
5. **训练预算是头等变量** —— 10× 步数把 wave 从 0.346 拉到 0.5256（CV 6.7%），在宣布「任务不可学」前先乘预算

## OaK / Options 方向的口径修正（2026-09-16，重要）

**「Options 在 lm4 上显著有害」这个结论已作废** —— 它是**十重复合**的产物，没有一条指向「option 有没有价值」：

| # | 混杂 | 证据 |
|:--|:--|:--|
| ① | 实现了论文实测**最差**的那类 option | `discover_subgoals` 用介数中心性找 bottleneck = 论文点名的 `shortest path options based on bottleneck states`；论文 Fig.1 实测其 planning **比 primitive 还慢** |
| ② | option 形式错 | 正确形式 `o=(π_o, β_o)` **不含目标**（论文脚注 elide 掉 initiation set）；目标属于 subtask，以 **stopping value `z_i(s)`** 承载 |
| ③ | 缺 subtask 层 | 论文：所有发现方法的差别**只在 GVF 的 `(c, z)`**；本项目 `GVFBank` 与发现**无接线** |
| ④ | `π_o` 不是学出来的 | 对模型贪心 argmax，无 option 价值函数，无 off-policy 学习(UWT) |
| ⑤ | **option model 不参与 planning** | option 的价值正是「让 planning 更高效」→ **按构造不可能体现收益** |
| ⑥ | 度量错 | 论文用 planning look-ahead 操作数；本项目用稳态 any-time acc |
| ⑦ | 缺 Step 11 utility feedback/删除 | OaK 与 Prototype-AI II 的**唯一区别**就是这套删改反馈；本项目决定「Option 不删」 |
| ⑧ | 跳过 Step 8 | 论文里 option 是第 10 步，前面应先有「无时间抽象的 one-step model-based agent」 |
| ⑨ | 时间抽象被掐死 | `max_opt_len=3`；论文里 option 跑 11–17 步 |
| ⑩ | 基准无结构 | lm4 `A ≡ 0` → 任何类别的 option 都无收益 |

**Ground truth 已建立,且 Fig.1（确定性两房间）与 Fig.6（随机四房间）两个方向都复现**（`tests/benchmark_oak_repro.py`）：

```
两房间(确定性, Fig.1):    primitive 1400 / shortest-path 1750 (1.250, 更慢 ✓)
                          reward-respecting  875 (0.625, 更快 1.6x ✓)
四房间(随机, Fig.6):      primitive 48872 / shortest-path 53710 (1.099, 更慢 ✓)
                          reward-respecting 26200 (0.536, 更快 1.87x ✓)
w̄ 扫描(**两布局都重扫, 规则已修正**):
  两房间 0.1/0.3 -> 175 (0.125) ; 1 -> 875 (0.625) ; 3/10/100 -> 1225 (0.875)  [饱和]
  四房间 0.1/0.3 -> 1271 (0.026) ; 1 -> 26200 (0.536) ; 3 -> 0.751 ; 10 -> 0.858 ; 100 -> 0.938  [逐档单调]
  ✅ 复现: 单调退化 + 小 w̄ 极有效(四房间 0.026 = 快 38 倍)
  ❌ 不复现: 「大 w̄ 退化成 shortest-path 水平」—— 实际趋近 primitive 基线(→1.0),
     即大 w̄ 让 option 变**无用**而非变**有害**
```

★ **注意**：两房间的旧数字（1716 / 2145 / **195**,8.8×）是在一个**被 `.ljust` 静默改过的几何**上得到的
—— 旧布局第 3、4 行是 8 字符而其余是 9,而且该几何实际**一条论文性质都不满足**。
重设布局后方向不变但效果量降到 1.6×。**旧数字已作废,以本表为准。**

**修复**：诊断文档 `docs/oak_diagnosis_from_papers.md`、结果 `docs/oak_repro_ground_truth.md`（七处 bug 台账化）；
代码 `hibs_lnn/gridworld.py`、`hibs_lnn/subtask_options.py`。
本轮修掉**七个真 bug**：三个在 ground truth 暴露（stopping value 恒等于主任务价值 /
planner 只贴现一步致 value iteration 发散 / goal 非吸收态致价值爆炸）、
一个在主回路（**coverage 的唯一写入点 `observe_action()` 主回路从未调用** → 闭环 option 静默空转、轨迹与 E9 逐位相同）、
三个在环境与自检（`FOUR_ROOM` 行长不齐+含空格 → 静默补墙+幽灵地板 / `TWO_ROOM` 同样行长不齐 /
**`check_two_room` 的 hall 候选未排除灰色格 → 该自检构造上永不可能通过**）。

### K1v2：修好管线后的**第一次有效测量**（2026-09-16）

修掉 `observe_action` 接线后重跑 25 臂（`tests/oak_k1v2.sh`，3.0 分/臂匀速 → 确在 GPU 上完成）。
**闭环档这次真的执行**：`goal` 的 `option_starts` **1→21**、`option_steps` **0→84**、`replan_steps=84`
（旧版是「1 次启动跑满 87% 全程」的锁死形态）。

```
e9(无 option)        0.7581 ± 0.0047
fixed                0.7470   Δ=−0.0111  p=0.0003  显著更差
goal                 0.7499   Δ=−0.0082  p=0.0036  显著更差
goal_term            0.7498   Δ=−0.0083  p=0.0033  显著更差
goal_term_override   0.7509   Δ=−0.0072  p=0.0055  显著更差

★ O2(闭环) vs O1(开环):  Δ=+0.0029  p=0.3287  不显著
```

**两个实质结论**：
① **排除了「问题是开环执行」** —— 闭环与开环无差别,所以病灶不在执行模式。
② **机制藏在 `term_reasons` 里**：`goal` 21 次启动中 **20 次 `expired`**、`steps/starts ≈ 4 = max_opt_len`、
5 个 seed 合计 `goal_reached` 只发生 1 次 → **option 从未到达过它的目标**。
因为目标是瓶颈中心性给出的任意远端状态,而非 reward-respecting subtask,它只是花光预算奔向一个到不了的地方 —— 纯机会成本。
（分析器 `tests/analyze_k1v2.py` **先过机制闸门再谈对照**;闸门已用旧 K1 的真实计数做夹具验证,
能准确拒掉 `starts==1`（锁死）与 `start_blocked>0`（空转）两种形态。）

**下一步按论文顺序，不跳步**：补 subtask 层 → option model 进 planner → off-policy 学 `π_o` → utility feedback → 才回到 lm4/lm5（且须先给它们造出有价值结构的状态空间）。

