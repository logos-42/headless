# Paper — Fractal Continual Learning

**Fractal Continual Learning: Recursive Structural Growth, Long-Horizon Adaptation, and the Boundary of Structural Advantage**

中文: 分形持续学习 —— 递归结构生长、长程适应, 以及结构优势的边界

## 内容

一篇方法论论文, 报告分形循环持续学习架构的完整研究:

1. **递归分形循环架构** — 密集根循环 + 稀疏后裔循环(更新频率 /D^depth), 凸组合门控跨层沟通, 相对衰减判据驱动结构生长, 干净-loss 过滤防罚分误触发。
2. **长期持续学习验证** — 非平稳 μ-动力学控制环境(μ₀ 漂移), 固定 vs 分形配对协议, 72,335 轮长跑:
   - 两种结构都持续学习: warm 从 ~130-176 步收敛到 ~9 步(cold 随机 ~32 步, 3.5× 增益)
   - 分形树自动长到 7 节点后稳定
3. **诚实边界(核心)** — 分形树在 小模型 regime 无统计优势:
   - 收敛后配对差异 < 0.3 步, 越界率 5.9% vs 5.8%
   - 3-seed 短跑 / 701 条真实 Forge 数据 5-fold 交叉验证均不显示优势
   - 结论: 递归结构塑性可行且长程稳定, 但性能优势需要更大规模 + 明显收益递减的 regime

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` | 论文正文(英文, article 11pt) |
| `main.bib` | 参考文献(Murray 1926 / Finn 2017 / Li 2017 / Kirkpatrick 2017 / 两个自仓 @misc) |
| `main.pdf` | 编译产物(6 页) |

## 编译

```bash
tectonic main.tex        # 单命令多趟编译(含 bibtex)
```

首次运行会下载 LaTeX 包(需要网络, 给 ≥300s 超时); 之后秒级。

验证: `python3 ../scripts/verify_pdf.py main.pdf`(若脚本在仓库根)。

## 数据与复现(在 HUSHFUSION Forge 仓)

- 长跑守护进程: `scripts/design_daemon_fractal.py` + `scripts/oml_supervisor_fractal.sh`
- 72,335 轮配对趋势: 服务器 `artifacts/oml_daemon_trend_fractal_*.jsonl`(活数据, 未入 git)
- 分析脚本: `scripts/analyze_fractal.py`
- 全部 spike(循环/生长/长期/多seed/干净评估/真实数据): `scripts/spike_fractal_*.py`, `scripts/spike_ag_*.py`
- 计划与完整证据链: `docs/wiki/plan-antigravity-control.md` §2e-2k

## 诚实边界(论文已写明)

- 控制环境是确定性模型, 不是实验装置; λ 是上游第二输入缺口
- 论文的贡献是**边界结果**(递归结构塑性可行 + 优势边界测量), 不是「分形更优」的主张
- 72,335 轮是该协议下的结论; 更大模型/多任务 regime 下的分形优势留待后续