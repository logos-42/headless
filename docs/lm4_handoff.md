# LM4 交接文档 — 两天无人值守运行

> 2026-09-11 起 · 服务器 `ssh -p 6100 root@100.100.30.185` · 代码 `/work/liuyuanjie/headless`

## 一、现在正在跑什么

服务器上 **6 条队列**(全部 `setsid` 脱离 SSH,PPID=1,我退出/你关终端都不影响):

| 队列 | GPU | 内容 | 状态 |
|:--|:--|:--|:--|
| `q1_runs_gpu0/1` | 0/1 | 池化对照(last/mean/cat/cat+agg/±lshell) | 收尾 |
| `q3_runs_gpu0/1` | 0/1 | **持续学习方法扫描**(agg-path × replay{1,2,5,10} × ±lshell) | 进行中 |
| `q4_runs_gpu0/1` | 0/1 | **任务定义变体**(域数 3/8/12、随机域序、±lshell、±agg) | 刚挂上 |

**监控命令**:
```bash
ssh -p 6100 root@100.100.30.185
cd /work/liuyuanjie/headless
tail -f results/queue_q3_runs_gpu0.log      # 队列进度
nvidia-smi                                   # GPU 利用率
ls results/q[134]_*/lm4_wave_results.json    # 完成的结果
```

**聚合结果**:
```bash
/work/liuyuanjie/envs/vllm-cu128/bin/python tests/compare_lm4.py \
  q1_pool_last q1_pool_mean q1_pool_cat q3_agg_rr2 q4_d3 q4_d8 q4_d12
```

## 二、已完成的关键结论

### 1. 特征: WFR 波谱是决定性的(+61% 天花板)

| 特征集 | joint 天花板 | D2 / D3 |
|:--|--:|--:|
| MAG 4 维 | 0.4354 | 0.110 / 0.138(≈随机) |
| **MAG + WFR 30 维** | **0.7007 ± 0.0205**(n=9) | **0.612 / 0.654** |

机制(`docs/lm4_wave_analysis.md` 有完整证据): `|B|` 与波谱**互补** —— `|B|` 管两端
(D0vsD5 AUC 0.9986),波谱管中段(D2vsD3 0.628)。物理上是**高密度区出现 160–500 Hz
波活动鼓包**(D0→D5 涨 1.7 数量级),而 2–30 Hz 是纯仪器底噪。

### 2. 持续学习成立(n=9)

`naive 0.1667(=随机) / replay 0.5852 ± 0.0621 / 保留率 83.6% ± 9.2%`,
硬指标 9/9 全过。**注意这是旧模型(`h[:,-1]` 池化)的结果。**

### 3. ⚠️ 天花板归因: 0.70 不是数据限制,是模型瓶颈

```
非序列聚合 MLP (150 维窗口统计量): h128 0.7859 / h512 0.8045 / h2048 0.8130
旧 WaveSSM (h[:,-1] 池化):                                   0.7213
```

**一个纯 MLP 顶到 0.81。** 但池化修缮**没能**解决问题:

| 池化 | joint |
|:--|--:|
| `last`(旧) | 0.7213 |
| `mean` | 0.7277 |
| `cat` = [last, mean, max] | **0.6681** ← 反而更差 |
| `cat` + L-shell | 0.6830 |

**→ "SSM 学不会池化"这个假设不成立。** 真因待查。`--agg-path`(把统计量直通分类头)
的结果还在跑,那是目前最有希望的方案。

## 三、未解问题(队列跑完后要回答)

1. **`--agg-path` 能否把 SSM 拉回 0.81 附近?** ← 最关键,决定后续用哪个 backbone
2. WFR 的 26 维里有 6 维是死重(AUC≈0.5),剪掉是否掉点?
3. **L-shell 是否冗余?** 预筛判断:它是 `|B|` 的单调函数,唯一可能的贡献是**解耦磁纬**。
   要做**固定 `|B|` 分箱内的配对检验**,而不是看总 acc。
4. 域数(N=3/8/12)和随机域序对保留率的影响 → 任务定义建议
5. 跨年(2016)泛化 —— 数据在 Mac 下载中

## 四、基础设施(已就绪)

- **特征缓存** `data/wave/_cache/`:`150s → 12s`(缓存键含 bands/wfr/lshell)
- `lm4_queue.py`:argv 指纹断点续跑 + 看门狗超时(同一 tag 用不同参数跑过会自动重跑)
- `config 指纹`写入结果 JSON,聚合时据此过滤(防诊断 run 污染统计)
- 可用开关:`--joint-only` `--lshell` `--no-cache` `--pool` `--agg-path` `--shuffle-domains`

## 五、⚠️ 注意事项

1. **Mac 不要休眠** —— 2016 跨年数据还在本机后台下载(`proc_b7496b5a9671`),
   下完需要传到服务器(`data/wave2016/`)。
2. **2015 与 2016 数据必须分目录** —— 特征管线按 `rbsp_a_*.npz` / `mag_*.npz` / `wfr_*.npz`
   通配,放一起会把两年混进同一个实验。
3. **GPU0 有邻居共用**(占约 22GB/40GB,利用率波动)。关键实验已分散到两卡。
4. 队列的 `--max-min` 超时会杀掉超时 run 并继续,`rc=TIMEOUT` 记在队列日志里 ——
   看到 TIMEOUT 说明该 run 需要单独重跑(直接重跑队列即可,断点续跑会跳过已完成的)。
5. 我改了 `--pool` 的**默认值回 `last`**(实测 cat 更差),队列里显式传 `--pool cat` 的不受影响。

## 六、文件索引

| 文件 | 作用 |
|:--|:--|
| `docs/lm4_wave.md` | 主文档: 数据/设计/泄漏修正/A 阶段结果/特征升级 |
| `docs/lm4_wave_analysis.md` | **机制分析**: 为什么 WFR 有效(含全部证据表) |
| `tests/run_lm4_wave.py` | 主实验脚本 |
| `tests/lm4_queue.py` | 队列驱动 |
| `tests/probe_ceiling.py` | 天花板归因探针(便宜, 先跑它) |
| `tests/compare_lm4.py` / `aggregate_lm4.py` | 结果聚合 |
| `tests/dl_{local,mag,wfr,mag_pos}.py` | 四个数据源下载器 |
