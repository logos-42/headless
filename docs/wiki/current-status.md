---
title: Sovereign AI 当前状态
source: session
created: 2026-09-06
last_confirmed: 2026-09-14
audience: reader
stage: draft
tags: [status]
status: current
---

## 最近更新

- 2026-09-14：**Benchmark 结算 —— 价值函数未能优于频率对齐对照（负面结论）**
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
- 待办：**查 `λ_fb` 为何对调度零影响**（`value` 与 `value-nofb` 逐位相同）—— 价值函数里的反馈闭环目前是死代码
- 待办：提高 seed 数（现 `n=3`，`|t|≈2` 只能算边缘证据，ddof 口径一变结论就翻转）
- 待办：**lm5 补 `random`（均匀随机）臂** —— 现缺该臂，无法在 lm5 上复现混淆对照
- 待办：新 benchmark 全臂开启 `--trace-every`（学习效率/达标步数指标仍缺）
- 待办：lm5 混合骨干 wave 退化归因（是否 SSM 文本通路梯度干扰 MLP 波通路）

## 已知的执行缺口（必须默认开启，不能只在单次 run 里用）

- **学习效率曲线**：`--trace-every` 在 105 个 result tag 中仅 1 个开启过 → 后续所有实验**默认开启**
- **类不平衡指标**：任何类不平衡任务必须**同时**报原始准确率 + 平衡准确率
- **push 后必须更新本 wiki**（log.md 追加 + 本文件同步 + `wiki_check.py` / `wiki_lint.py --strict=v2` / `raw_manifest_check.py` 全绿）

## 方法论沉淀（本日新得，已写入 `ml-ablation-methodology` 技能）

1. **一个实现 bug 可以伪装成多个科学结论** —— BN 漂移在全部配置里生效，因此同时污染了方法对比/头对比/架构对比
2. **反直觉的好结果必须当 bug 追** —— 本日三次：naive 跨年 1.000、BN 漂移、跨年指标设计错误
3. **排除一个假说后必须继续找真因** —— 容量扫描否定容量假说后停手，错过了真因是 BN
4. **架构结论不跨模态迁移** —— 应做按模态分派，每模态各自验证
5. **训练预算是头等变量** —— 10× 步数把 wave 从 0.346 拉到 0.5256（CV 6.7%），在宣布「任务不可学」前先乘预算
