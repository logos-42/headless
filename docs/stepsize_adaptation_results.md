# 步长自适应 (IDBD / Autostep) 在真实持续学习回路的效果

**日期**: 2026-09-14 · **实验**: `tests/rl_driver.sh` + `tests/rl_ext.sh` · **分析**: `tests/analyze_rl.py`

回答的问题 (OaK 第②条): **"每个 learned weight 有自己的 step-size, 由元学习调" 到底带不带来收益?**

---

## 一、设置

- lm4 真实回路: `--backbone mlp --norm fixed --cl-method replay --head linear
  --domains 6 --fine-bins 60 --rounds 12 --stream revisit --proposer-k 3`
- 提议器 (`--proposer rl`): 动作 = 选 3 个物理细区间去训练;
  reward = **Δ(any-time 准确率)**; θ 由 `δ = r − V_θ` 更新
- 步长算法: `--rl-algo {idbd, autostep}`
- 对照: `--proposer value` (手工 V = λ_sim·sim + λ_con·con + λ_cov·cov + λ_fb·fb)

| 臂 | 说明 |
|:--|:--|
| `idbd` | IDBD, μ=0.05, α₀=0.2 (我们之前唯一能找到的活点) |
| `auto` | Autostep, μ=1e-2, α₀=0.1 (**论文推荐默认, 未针对本任务调参**) |
| `auto2` | Autostep, μ=0.05, α₀=0.2 |
| `value` | 手工价值函数 (无步长概念) |

---

## 二、结果 (n=3 seed: 1/7/42)

| arm | n | any_time | std | CV% | replay | **最差遗忘界** | **alpha std / ratio** |
|:--|--:|--:|--:|--:|--:|--:|:--|
| `auto` | 3 | 0.7596 | 0.0274 | 3.6 | 0.7736 | 0.3719 ± 0.0309 | 0.0004 / 1.04 |
| `auto2` | 3 | 0.7575 | 0.0221 | 2.9 | 0.7776 | 0.3368 ± 0.0346 | 0.0017 / 1.11 |
| `idbd` | 3 | 0.7659 | 0.0230 | 3.0 | 0.7633 | **0.2899 ± 0.0179** | **0.2357 / 3.92** |
| `value` | 3 | **0.7702** | 0.0212 | 2.8 | 0.7661 | 0.3808 ± 0.0606 | — |

### 检验 1: 步长是否分化 (alpha_std > 0 且 max/min > 1.5)

```
auto    std=0.0004  ratio=1.04  弱   <-- Autostep 几乎不分化!
auto2   std=0.0017  ratio=1.11  弱
idbd    std=0.2357  ratio=3.92  分化 ✓
value   std=0.0000  ratio=0.00  未分化 (手工公式, 无步长概念)
```

### 检验 2: 两两 Welch t 检验 (any_time)

| 对比 | Δmean | t | p | 判定 |
|:--|--:|--:|--:|:--|
| auto − auto2 | +0.0022 | 0.11 | 0.921 | 不显著 |
| auto − idbd | −0.0062 | −0.30 | 0.779 | 不显著 |
| auto − value | −0.0106 | −0.53 | 0.627 | 不显著 |
| auto2 − idbd | −0.0084 | −0.46 | 0.672 | 不显著 |
| auto2 − value | −0.0127 | −0.72 | 0.512 | 不显著 |
| idbd − value | −0.0043 | −0.24 | 0.822 | 不显著 |

### 检验 3: Autostep 免调参性

```
auto  (论文默认 μ=1e-2, α₀=0.1)  0.7596
auto2 (μ=0.05,  α₀=0.2)          0.7575
差 0.0022  ->  免调参成立 ✓
```

### 检验 4: `idbd` vs `value` 在**最差遗忘界**上 (这一项有信号)

```
idbd  0.2899 ± 0.0179      <- 均值最好, 且 std 最小
value 0.3808 ± 0.0606      <- std 是 idbd 的 3.4 倍

Δ = −0.0909 (−24%)   t = −2.49   p = 0.112   -> 效应量大但 n=3 不显著
```

**正在补 seed** (`tests/rl_ext.sh`: seed 2/3/4/5/6, 只跑 idbd + value 两臂)。

---

## 三、结论 (诚实)

1. **步长分化确实发生了** (IDBD: alpha_std=0.2357, max/min=3.92),但按 any-time 准确率
   **没有带来任何显著收益** (idbd − value: −0.0043, p=0.82)。手工 `value` 反而数值最高。
   → **步长不是我们任务的主瓶颈。** 与 V35.3「学习效率激活 ≠ 组合泛化」一致。

2. **反直觉: 真实回路里 IDBD 分化强 (0.2357), Autostep 几乎不分化 (0.0004)**
   —— 与 weight-flipping 基准 (Autostep 6/8 > IDBD 4/8) **相反**。
   推测: Autostep 的 Modification 2 (`α /= M`, `M = max(Σαx², 1) ≥ 1`) **均匀压缩所有步长**,
   在 fdim=9 的小特征空间里把 max/min 比压平了。

3. **Autostep 的"免调参"确实成立** (两档超参差 0.0022),**但代价是它同时放弃了分化**
   —— 归一化把有用特征和噪声特征一起压平。这是一个真实的权衡,不是免费午餐。

4. **唯一有信号的项是最差遗忘界**: `idbd` 均值最好 (0.2899) **且方差最小** (0.0179, 比
   `value` 小 3.4 倍)。t=−2.49, p=0.112 —— **待 seed 扩展**。
   这个"均值更好 + 方差更小"的签名与 LMT-twister V35.22「价值函数=稳定器(方差缩 3 倍)」一致。

---

## 四、四个 bug (本轮自查)

| bug | 症状 | 严重性 |
|:--|:--|:--|
| `--rl-algo/--rl-mu/--rl-alpha0` **只加签名和 CLI, 调用点未接** | 四臂全报 `algo=autostep`; 不同超参给出**逐位相同**结果 | 🔴 整个对照作废, 已修 (`run_lm4_wave.py` 调用点) |
| JSON **非原子写 + 无 numpy 兜底** | `TypeError: int64 not JSON serializable`; 留下**损坏半截文件** (`bm2_random-matched` 三 seed 全坏) | 🔴 静默产出坏数据, 已修为 `tolist` 兜底 + 临时文件 `os.replace` |
| IDBD 步长更新式误删 `x` | 官方 IDBD Algorithm 4 是 `β += θ·δ·x·h`, 我一度改成 `exp(μδh)` | 🟠 已回滚 |
| `_json_default` 里 `tolist` 排在 `item` 之后 | `np.array([1,2]).item()` 抛 `ValueError` | 🟡 自身测试抓出, 已修 |

**教训**: 「加了 CLI 参数和函数签名」不等于「参数真的生效」—— 必须有断言/回读验证
(实测四个臂输出逐位相同才暴露)。

---

## 五、附: IDBD vs Autostep 在论文原始基准上的对照 (weight-flipping)

20 维输入, 前 15 维目标恒 0、后 5 维每 20 步翻转 ±1。`tests/compare_idbd_autostep.py`

| | IDBD | Autostep |
|:--|:--|:--|
| 分化成功 | 4/8 档 | **6/8 档** |
| **发散 (MSE ~1.6e9)** | **3/8 档** ⚠️ | **0/8 档** ✅ |
| MSE 范围 | 1.4 → 1.6e9 | **2.1 → 4.9 (全稳)** |

与 Degris et al. 2024 / Mahmood et al. 2012 所述一致:
IDBD 好区间窄且紧邻发散点; Autostep「过冲不可能」。

**单位分析 (Autostep 论文 §3) 解释了我们的困境**: IDBD 指数项 `δ·x·h` 量纲是 **y²**,
故最优 μ 量纲是 **1/y²**。我们的 reward = Δ(any-time) ≈ 0.02 → 所需 μ 比 O(1) 目标大约 2500 倍。
这解释了实测: α₀=0.01/0.05 时步长死亡 (`alpha_std≈1e-6`), α₀=0.2 才活。
**与 V35.2/3 的 `β_std=1.6e-05` 是同一现象在两处的独立复现。**
