# Paper — Before Fractal Continual Learning

**Before Fractal Continual Learning: An Implementation Audit of Recursive Structural Growth and Long-Horizon Evaluation**

中文: 分形持续学习之前 —— 递归结构生长与长程评估的实现审计

## 内容

一篇**实现审计方法论**论文。原稿声称「分形结构无优势」的边界结果,
三轮对抗审稿(NeurIPS 级 + CNS 级)后发现主实验代码存在致命实现缺陷,
论文改写为审计版:

1. **审计发现(核心)** — 72,335 轮主实验代码(`design_daemon_fractal.py`,
   被审版本 `10ba961^`)里, 分形树的子节点**从不被训练**:
   - children 不在任何优化器的参数表 (opt 只含自身 net+gate)
   - 声称的 D^{-d} 稀疏调度只赋值从未参与调度判断
   - 7 节点是硬编码 safety cap, 不是自主收敛
   - 种子 = time.time(), 不可复现
   - 结论: 长跑测的是「根节点门控未训练随机子网」, 不是分形学习器
2. **仍然成立** — 根节点长程适应真实: warm 130→9 步 (随机基线 ~32 步),
   保持 72,000 轮; 相对衰减生长检测器工作: 6 次生长 1→7 节点 (cap 停)
3. **撤回** — 「分形结构无优势」→「未定论 (undecidable)」:
   该 run 对分形假设的信息量为零
4. **修复** — Forge 仓 `53e4938`: 递归子节点训练 + 稀疏调度生效 +
   warmup 持久优化器 + `--seed` 可复现
5. **决定性重跑(进行中)** — 预注册 `PREREGISTRATION.md` (等价界 ±1 步,
   收敛段 ≥2000 轮, n≥5000 warm 轮, paired bootstrap CI), 服务器
   seed=20261012 配平版, **结果未出, 另行报告**

## 文件

| 文件 | 说明 |
|---|---|
| `main.tex` | 论文正文(英文, article 11pt, 审计版) |
| `main.bib` | 参考文献(Murray 1926 / Finn 2017 / Li 2017 / Kirkpatrick 2017 / 自仓 ×2) |
| `main.pdf` | 编译产物(6 页) |
| `PREREGISTRATION.md` | 决定性重跑预注册(等价界/判定规则/复现信息) |
| `WRITING-STANDARD.md` | 论文写作标准(AI 顶会 + CNS 级, 审稿判据) |

## 编译

```bash
tectonic main.tex        # 单命令多趟编译(含 bibtex)
```

首次运行会下载 LaTeX 包(需要网络, 给 ≥300s 超时); 之后秒级。

## 审稿历程

| 轮次 | 审稿人 | 结论 | 处理 |
|---|---|---|---|
| 第1轮 | NeurIPS 级 + CNS 级 | 退稿(2 致命: 子节点未训练 / 不可复现) | 改写审计版 + 修复代码 |
| 第2轮 | NeurIPS 级 + CNS 级 | 大修(算力不匹配 / 无预注册 / 未钉 SHA) | 配平 53e4938 + 预注册 + 裁决框 |
| 第3轮 | NeurIPS 级 | 小修(README 同步 / 重跑状态声明 / 预注册样本上限) | 本 README 重写 + 正文声明 |

## 数据与复现(在 HUSHFUSION Forge 仓)

- 守护进程: `scripts/design_daemon_fractal.py`(修复版 @ 53e4938)
- 72,335 轮配对趋势(被审计): 服务器 `artifacts/oml_daemon_trend_fractal_*.jsonl`(活数据, 未入 git, 已披露)
- 决定性重跑: 服务器 `artifacts/oml_daemon_fractal_matched_20261012.jsonl`(seed 20261012, 配平版, 进行中)
- 全部 spike: `scripts/spike_fractal_*.py`, `scripts/spike_ag_*.py`
- 证据链: `docs/wiki/plan-antigravity-control.md` §2e-2k

## 诚实边界(论文已写明)

- 控制环境是确定性模型, 不是实验装置; λ 是上游第二输入缺口
- 论文贡献是**审计方法论**(看似干净的对比如何静默地违背假设), 不是「分形更优」或「分形无优势」的主张
- 决定性重跑结果未出; 论文明确区分已有证据 vs 未来证据
