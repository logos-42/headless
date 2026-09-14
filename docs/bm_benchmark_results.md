# LM4 / LM5 持续学习 Benchmark 结算 (价值函数调度 vs 频率对齐对照)

- **日期**: 2026-09-14
- **数据来源**: 服务器 `/work/liuyuanjie/headless/results/` (driver `tests/bm_driver.sh`, 2026-09-14 05:14–06:36 UTC, `BM_DONE` 已落地)
- **汇总脚本**: `tests/analyze_bm_full.py` (lm4 + lm5 全臂, Welch t 检验, 输出 `/tmp/bm_summary.json`)
- **配置**: 所有臂 3 seed (42/1/7), `cl-method=replay`, `norm=fixed`, `head=linear`
  - **LM4**: StatMLP 骨干, 6 域 (按 log10 密度分位), 候选池 = 物理描述子 `[log10 密度, log10 偶极 L, sin 磁纬]` 切 18 个细区间, 12 轮
  - **LM5**: `MultiModalHybrid` 骨干, 8 域 (en/zh/code/wave/causal/causal_do/causal_bal/causal_do_bal), 8 轮
- **统计口径**: `mean±std` 用样本标准差 (ddof=1); 两组对比用 **Welch t** (`t=(Δ)/sqrt(s²/n+s²/n)`), 判定阈值 `|t|>=2`
  - ⚠️ `n=3`, 自由度极小, `|t|≈2` 只是**边缘**证据; 同一天平在 ddof=0 下会把 `t=-1.66` 抬到 `t=-2.04` (见下文「口径敏感性」)

---

## 0. 执行状态 (含一次故障与修复)

| 批次 | 结果 |
|:--|:--|
| LM4 stage1 (value ×3 + bins 任务流 ×12) | 15/15 rc=0 |
| LM4 stage2 (value-nofb / random / **random-matched** ×3) | **random-matched 3/3 rc=1 失败** (另 6/6 成功) |
| LM5 stage3 (4 任务流 ×3 seed) | 12/12 rc=0 |
| LM5 stage4 (value ×3 → value-nofb / random-matched ×3) | 9/9 rc=0 |

**lm4 `random-matched` 首次全灭, 根因是一个真 bug (已修并重跑)**:

`tests/bm_driver.sh` 把 `results/bm_value_s42/lm4_wave_results.json` 整文件拷成 `results/freq_profile.json`,
而 lm4 结果文件是**嵌套**结构 (`{naive, replay, _config}`), `proposer_freq` 在 `replay` 之下;
`run_lm4_wave.py` 的加载器写的是平铺取值 `_fp.get("proposer_freq") or _fp`, 取不到就**退化成整个 dict** →
`np.asarray(dict, dtype=float)` 抛 `TypeError: float() argument must be a string or a real number, not 'dict'`,
启动 9 秒即崩。日志伪装成正常: `[freq-matched] 载入频率分布: 3 项` (3 = dict 的顶层键数, 不是候选数)。
lm5 的 `lm5_mm_results.json` 是**扁平**结构, 所以 lm5 同臂 9/9 正常 —— 这也解释了为什么只有 lm4 挂。

修法 (向后兼容, 两种结构都吃, 稀疏 dict 自动转密集向量): `tests/run_lm4_wave.py` 加载器改为
平铺 → `replay.proposer_freq` 回退 → `{idx: freq}` 稠密化。修后重跑 3 seed, `rc=0`, 且日志正确显示
`载入频率分布: 18 项`。**未重启 driver, 只补跑失败的 3 个 run** (日志 `results/bm_stage2_rerun.log`)。

---

## 1. LM4 — 任务流 + 调度臂 (3 seed)

| 臂 | 类型 | replay mean±std | any-time mean±std | 最差遗忘界 | 平均遗忘 |
|:--|:--|:--|:--|:--|:--|
| `fixed`(bins, 启发式域序) | 任务流 | 0.7601±0.0035 | 0.6940±0.0159 | 0.3213 | 0.2123 |
| `perm` (随机排列) | 任务流 | 0.7393±0.0068 | 0.6994±0.0234 | 0.4838 | 0.2401 |
| `revisit` (反复) | 任务流 | 0.7522±0.0090 | 0.7601±0.0208 | 0.4495 | 0.2302 |
| `nonstationary` | 任务流 | 0.7487±0.0134 | 0.7058±0.0346 | 0.4591 | 0.2232 |
| **`value`** (价值函数) | 调度臂 | **0.7734±0.0158** | **0.7699±0.0160** | 0.4133 | 0.1716 |
| `value-nofb` (消融反馈项) | 调度臂 | 0.7734±0.0158 | 0.7699±0.0160 | 0.4133 | 0.1716 |
| `random` (均匀随机) | 调度臂 | 0.7931±0.0132 | 0.7898±0.0222 | 0.2651 | 0.1394 |
| **`random-matched`** (★频率对齐对照) | 调度臂 | **0.7778±0.0186** | **0.7574±0.0431** | **0.2624** | 0.1491 |

逐 seed (replay) 对照, 便于核查离散度:

| 臂 | seed 42 | seed 1 | seed 7 |
|:--|:--|:--|:--|
| `value` | 0.7756 | 0.7879 | 0.7566 |
| `random` | 0.7785 | 0.8041 | 0.7968 |
| `random-matched` | 0.7676 | 0.7993 | 0.7667 |

**读法**: 3 个调度臂全部高于 4 个任务流臂; 但在 3 个调度臂内部, **`value` 是 replay / any-time 最低的那个**,
`random-matched` 居中, `random` 最高。**最差遗忘界上 `value` (0.4133) 明显比 `random-matched`/`random`
(0.2624 / 0.2651) 差**, 与任务流臂同级。

## 2. LM5 — 多模态 (3 seed)

| 臂 | 类型 | replay(=末轮均值) mean±std | any-time mean±std | 最差遗忘界 | 平均遗忘 |
|:--|:--|:--|:--|:--|:--|
| `fixed` | 任务流 | 0.2775±0.0450 | 0.1883±0.0039 | 0.3333 | 0.0916 |
| `perm` | 任务流 | 0.2979±0.0236 | 0.1913±0.0284 | 0.2917 | 0.0576 |
| `revisit` | 任务流 | 0.2866±0.0118 | 0.1915±0.0261 | 0.4167 | 0.0728 |
| `nonstationary` | 任务流 | 0.2875±0.0256 | 0.1800±0.0298 | 0.1378 | 0.0419 |
| **`value`** | 调度臂 | **0.2702±0.0691** | **0.2126±0.0135** | 0.2083 | 0.0758 |
| `value-nofb` | 调度臂 | 0.2702±0.0691 | 0.2126±0.0135 | 0.2083 | 0.0758 |
| **`random-matched`** (★) | 调度臂 | **0.2671±0.0141** | **0.1765±0.0273** | 0.2950 | 0.0693 |

⚠️ **LM5 没有 `random`(均匀随机) 臂**, 因此 lm5 只能做干净对照, 做不了混淆对照。
⚠️ `value` 的 replay 方差极大 (`CV 25.6%`): seed 42 = 0.1921 是离群点, 另两个 seed 0.2947 / 0.3236。
`random-matched` 反倒最稳 (CV 5.3%) —— **「价值函数方差更小」的旧结论在 lm5 replay 上不成立**。

各臂末轮逐域准确率, seed 42 (domains = `en, zh, code, wave, causal, causal_do, causal_bal, causal_do_bal`):

| 臂 | en | zh | code | wave | causal | causal_do | causal_bal | causal_do_bal |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| `fixed` | 0.125 | 0.000 | 0.000 | 0.3575 | 0.3545 | 0.5175 | 0.2774 | 0.3277 |
| `perm` | 0.250 | 0.000 | 0.125 | 0.4948 | 0.3640 | 0.4845 | 0.2754 | 0.3185 |
| `revisit` | 0.125 | 0.000 | 0.250 | 0.3962 | 0.3530 | 0.5470 | 0.2511 | 0.3018 |
| `nonstationary` | 0.250 | 0.125 | 0.250 | 0.1490 | 0.3545 | 0.5180 | 0.2771 | 0.3292 |
| `value` | 0.125 | 0.000 | 0.375 | 0.3462 | 0.3515 | 0.5445 | 0.2815 | 0.3344 |
| `random-matched` | 0.125 | 0.000 | 0.375 | 0.3196 | 0.3640 | 0.4845 | 0.2754 | 0.3185 |

**读法**: 文本三域 (`en/zh/code`, 机会水平 0.125/0.0/0.125 的下一 token 预测) 基本贴地, 全 benchmark 的信号
几乎全部来自 `wave` 与 4 个因果域。因此 lm5 的均值被少数域主导, 臂间差异很难做统计区分。

## 3. 关键两两对比 (Welch t, Δ = 前者 − 后者)

★ **干净对照** = `value` vs `random-matched` (同一条实测提议频率分布驱动, 只差「选哪个区间」的决策规则)。
非干净对照 `value` vs `random` 混入了「复习更均匀」效应, 只能参考。

| 指标 | 对比 | Δ | t | 判定 |
|:--|:--|:--|:--|:--|
| **LM4 replay** | **`value` vs `random-matched`** | **−0.0045** | **−0.32** | **不显著** |
| **LM4 any-time** | **`value` vs `random-matched`** | **+0.0125** | **+0.47** | **不显著** |
| **LM4 最差遗忘界** | **`value` vs `random-matched`** | **+0.1509** | **+2.22** | **显著 (value 更不安全)** |
| LM4 replay | `value` vs `random` (参考) | −0.0197 | −1.66 | 不显著 |
| LM4 any-time | `value` vs `random` (参考) | −0.0199 | −1.26 | 不显著 |
| LM4 最差遗忘界 | `value` vs `random` (参考) | +0.1482 | +1.95 | 不显著 (趋近) |
| LM4 replay | `value` vs `value-nofb` | +0.0000 | +0.00 | 不显著 (**完全无差**) |
| LM4 replay | `value` vs `perm` | +0.0341 | +3.44 | 显著 |
| LM4 any-time | `value` vs `perm` | +0.0706 | +4.31 | 显著 |
| LM4 replay | `value` vs `revisit` | +0.0212 | +2.02 | 显著 (边缘) |
| LM4 any-time | `value` vs `revisit` | +0.0098 | +0.65 | 不显著 |
| LM4 replay | `random-matched` vs `random` | −0.0153 | −1.16 | 不显著 |
| **LM5 replay** | **`value` vs `random-matched`** | **+0.0030** | **+0.07** | **不显著** |
| **LM5 any-time** | **`value` vs `random-matched`** | **+0.0361** | **+2.06** | **显著 (边缘, value 更好)** |
| **LM5 最差遗忘界** | **`value` vs `random-matched`** | **−0.0867** | **−0.75** | **不显著** |
| LM5 replay | `value` vs `value-nofb` | +0.0000 | +0.00 | 不显著 (**完全无差**) |

### 口径敏感性 (必须同报)

`tests/analyze_benchmark.py` (旧脚本) 用 `np.std` 默认 `ddof=0` 且把 std 当总体标准差, 会把 `n=3` 的方差低估,
同一对比 `value vs random` (LM4 replay) 得到 **t=−2.04「显著」**; 本报告改用 `ddof=1` 后为 **t=−1.66「不显著」**。
`n=3` 时这种翻转是常态, 因此 **不要把单个 `|t|≈2` 当结论**。本报告所有判定按 ddof=1 给出。

---

## 4. 结论: 价值函数到底有没有用?

**诚实结论: 在这两个 benchmark 上, 没有证据表明价值函数优于频率对齐对照 —— 干净对照下它不赢, 甚至在一处显著更差。**

1. **核心对照 (`value` vs `random-matched`) 不支持价值函数**
   - LM4: replay `Δ=−0.0045 (t=−0.32)`, any-time `Δ=+0.0125 (t=+0.47)` —— **都没赢**, 数值上还略低。
   - LM4 最差遗忘界: `Δ=+0.1509 (t=+2.22)` —— **显著更差**。这是本次唯一 `|t|>=2` 的核心对照,
     且方向对价值函数**不利**。
   - LM5: replay `Δ=+0.0030 (t=+0.07)` 不显著; any-time `Δ=+0.0361 (t=+2.06)` 边缘显著更好;
     最差遗忘界不显著。**lm4 与 lm5 方向相互矛盾** → 只能判为「测不出稳定效应」, 不能判为「有效」。
2. **连混淆对照都赢不了**: `value` vs `random` (均匀随机) 在 LM4 上 `Δ=−0.0197` —— 即便把「复习更均匀」
   这份不属于价值函数的红利算给 value, 它**仍然低于**均匀随机。换言之, LM4 上价值函数的表现不佳
   **不是**频率分布差异造成的假象。
3. **`value-nofb` 与 `value` 逐位相同** (三种指标 `Δ=+0.0000, t=+0.00`, pre-seed 频率向量也完全相同) →
   注入判别器的那一项 `λ_fb` (**真实反馈 = 1−acc**) 在当前实现里**对调度结果零影响**。
   这一项要么没被 proposal 逻辑读到, 要么被 softmax 温度压平。**这是必须修的真 bug**:
   它意味着「价值函数里的真实反馈闭环」目前是**死代码**, 我们此前把 `value` 臂的结果读作
   「价值函数(含反馈)有效」的推论**不成立** —— 实测到的任何效果都只能来自 `sim/con/cov` 三项。
4. **价值函数确实显著优于「没有调度」**: LM4 上 `value` vs `perm` (`Δ=+0.034, t=+3.44` replay;
   `+0.071, t=+4.31` any-time) 与 vs `revisit` (edge) 显著。但这只说明
   **「用一个提议器在线选任务」优于「预先定死的任务流」**, 而 `random` 与 `random-matched` 同样具备这个优势
   (且 LM4 上还更大)。**这是「调度」的功劳, 不是「价值函数」的功劳。**
5. **方差稳定器说法也被推翻**: 干净配置下 LM4 `value` 的 replay `CV=2.0%`, 并没有小于
   `random`(1.7%) 或 `random-matched`(2.4%); LM5 `value` 反而是全场方差最大的臂 (CV 25.6%,
   而 `random-matched` 5.3%)。旧结论「value 缩方差」只在旧的 `naive` 臂上成立, 不构成普遍结论。

### 与我们的理论预期的关系

`V35.19–V35.22 / LM1` 的结论是「价值函数的价值在**迁移质量与稳定性**, 不在同分布即时收益」。
本轮拆掉了同分布收益这个焦点, 改用边缘设备真正在意的 any-time 与最差遗忘界, 结果是:
**在 lm4/lm5 这两个 benchmark 上, 连这两个新指标也没能证明价值函数的增量。** 目前能站得住的只有
「在线调度 > 固定顺序」, 以及「提议器的存在会大幅缓解 naive 臂的遗忘」(历史结论, 本轮未复测)。

### 局限 (随结论同报)

- **`n=3`, 检验功效极低**。`|t|>=2` 只算边缘证据; 任何一条判定都不该被单独引用。
- **LM4 三个调度臂共用同一个频率 profile** (取自 `bm_value_s42`); `random-matched` 是**分布**对齐,
  不是逐轮轨迹对齐, 严格说只消除了「复习频率分布」这一层混淆。
- **LM5 缺 `random` 臂**, 无法在 lm5 上复现混淆对照。
- **LM5 信号集中在 `wave` + 4 个因果域**, 文本三域贴地, 臂间差异被少数域主导。
- 本轮**未开启学习效率曲线** (`--trace-every`), 达标步数指标缺失 —— 仍是已知执行缺口。
- 上一轮笔记里 `value` 臂的数字 (replay 0.6862 / naive 0.4282) 与本轮 (0.7734) 不可比: 本轮是重构后的
  新 benchmark (`--stream` 任务流 + 18 候选池 + `--norm fixed`), 上一轮是固定域序的旧版。

---

## 5. 复现

```bash
# 服务器
python /tmp/analyze_bm_full.py            # = tests/analyze_bm_full.py
```

```bash
# 补跑 lm4 random-matched (修复后)
for S in 42 1 7; do
  CUDA_VISIBLE_DEVICES=0 /work/liuyuanjie/envs/vllm-cu128/bin/python -u tests/run_lm4_wave.py \
    --backbone mlp --norm fixed --cl-method replay --head linear \
    --epochs-per-domain 1000 --joint-steps 0 --wfr-bands 13 --lshell \
    --pool cat --agg-path --domains 6 --fine-bins 18 --rounds 12 \
    --proposer random-matched --stream fixed --seed $S --device cuda \
    --freq-profile results/freq_profile.json --out results/bm_random-matched_s$S
done
```
