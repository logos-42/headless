---
title: Sovereign AI 当前状态
source: session
created: 2026-09-06
last_confirmed: 2026-09-16
audience: reader
stage: draft
tags: [status]
status: current
---

## 最近更新

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

- **学习效率曲线**：`--trace-every` 在 105 个 result tag 中仅 1 个开启过 → 后续所有实验**默认开启**
- **类不平衡指标**：任何类不平衡任务必须**同时**报原始准确率 + 平衡准确率
- **push 后必须更新本 wiki**（log.md 追加 + 本文件同步 + `wiki_check.py` / `wiki_lint.py --strict=v2` / `raw_manifest_check.py` 全绿）

## 方法论沉淀（本日新得，已写入 `ml-ablation-methodology` 技能）

1. **一个实现 bug 可以伪装成多个科学结论** —— BN 漂移在全部配置里生效，因此同时污染了方法对比/头对比/架构对比
2. **反直觉的好结果必须当 bug 追** —— 本日三次：naive 跨年 1.000、BN 漂移、跨年指标设计错误
3. **排除一个假说后必须继续找真因** —— 容量扫描否定容量假说后停手，错过了真因是 BN
4. **架构结论不跨模态迁移** —— 应做按模态分派，每模态各自验证
5. **训练预算是头等变量** —— 10× 步数把 wave 从 0.346 拉到 0.5256（CV 6.7%），在宣布「任务不可学」前先乘预算
