# LM4 — 持续学习的电磁波模型

> 创建 2026-09-10 · 数据 NASA CDAWeb · 训练机 2×A100 服务器

## 目标

把 headless 的持续学习框架(SSM + OML + Replay)从合成规则世界(lm1)、
文本(lm2)、BPE 跨域(lm3)扩展到**真实空间物理电磁波数据**:

**从电磁波/磁场观测持续学习推断等离子体物理状态,不遗忘先前学到的状态。**

## 数据源

NASA CDAWeb HAPI 2.0 (`https://cdaweb.gsfc.nasa.gov/hapi`),Van Allen Probes (RBSP-A):

| 数据集 | 采样 | 用途 | 时间跨度 |
|---|---|---|---|
| `RBSP-A_DENSITY_EMFISIS-L4` | 6 s | `density` [cm⁻³] → **物理状态标签** | 2012-2019 |
| `RBSP-A_MAGNETOMETER_1SEC-GEI_EMFISIS-L3` | 1 s→6 s | `Magnitude`/`rms`/`delta`/`lambda` → **特征** | 2012-2019 |

本次下载: DENSITY 2015 全年(12 月, 408 万点) + MAG 2015 Q1(3 月, 130 万点),
对齐后 **197,708 个窗口**(窗口 32 步, stride 4)。

## 关键设计决策

### 1. 特征必须排除"解析泄漏"

DENSITY L4 含 `fpe`(等离子体频率)、`fuh`(上杂波频率)、`wpe_over_wce`。
物理上 `fpe = 8980·√ne` [Hz] —— **与目标 density 有解析关系**。
用它们预测密度 = 查公式,不是学习。**已排除**。

保留真正独立的观测量:
- `Magnitude` — 背景磁场强度 [nT]
- `rms` — 磁场波动(RMS)= **波活动指标**
- `lambda` / `delta` — 卫星位置(纬/经度)

变换: `log10(Magnitude)`, `log10(rms)`(跨数量级), 位置原值。

### 2. 物理状态 = 密度分位数域

`log10(density)` 实际跨 **6.3 个数量级**(10⁻³ ~ 10³ cm⁻³, 2015 全年)。
按分位数划分 6 个域(各 ~33,000 样本, 保证可训练且边界由数据决定):

```
D0: log10 < 0.51      D3: 1.67 ~ 2.21
D1: 0.51 ~ 1.09       D4: 2.21 ~ 2.82
D2: 1.09 ~ 1.67       D5: > 2.82
```

### 3. 时间对齐用近邻匹配

DENSITY 与 MAG 采样起始偏移不同 → 精确时间戳匹配会全部失败(曾导致 0 样本)。
修复: numpy `datetime64` 向量化 + `searchsorted` 近邻,容差 3 s。

## 可解性诊断(关键前置)

持续学习实验有意义的前提: **非持续学习(联合训练)能达到明显高于随机的准确率**。

| 任务 | 上限 acc (MLP, 聚合特征) | 随机基线 |
|---|---|---|
| 6 类(密度分位数域) | **0.523** | 0.167 |
| 3 类(低/中/高) | **0.720** | 0.333 |

→ 特征判别力显著 → 任务可解 → 持续学习对比有意义。

**⚠️ 首个 SSM 版本(联合 600 步)只有 0.315** —— 是训练步数不足,
不是任务问题。GPU 版需大幅提高步数(每域 500-1000 步)。

## 实验协议

```
联合训练 (上限) → Naive 顺序 (D0→D5) → Replay 顺序
指标: 最终平均 acc / 遗忘 (历史最佳 - 最终) / 遗忘矩阵
```

模型 `WaveSSM`: `Linear(4→d_model)` → N × `SSM_Layer_V30_3` → `LayerNorm` → `Linear(d_model→6)`,
取末时刻 logits。复用 headless 的 SSM 实现(`hibs_lnn/ssm_v30_3.py`)。

## 文件

| 文件 | 作用 |
|---|---|
| `tests/dl_local.py` | 下载 DENSITY(本机并行, 比服务器快 17×) |
| `tests/dl_mag.py` | 下载 MAG + 提取独立特征 |
| `tests/run_lm4_wave.py` | lm4 实验(数据管线 + 持续学习 + 诊断) |
| `data/wave/*.npz` | 数据(68 MB, gitignore) |
| `results/lm4_wave/` | 结果报告 |

## 坑

1. **服务器直连 CDAWeb 慢 17×**(67 s/天 vs 本机 4 s/天)→ 数据必须本机下载后传输
2. **精确时间戳对齐全失败** → 必须近邻匹配 + 容差
3. **fpe/fuh/wpe_over_wce 是标签泄漏** → 必须排除
4. `timeout` 命令 macOS 不存在 → 用 `gtimeout` 或后台运行
5. HFR_SPECTRA(高频波谱)是**突发模式**,大量时段无数据 → 本次未用
