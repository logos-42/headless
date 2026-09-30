#!/bin/bash
# oak_lmeff.sh — 学习效率 / 更新效率 的**第一次真实测量**
#
# 规格: docs/learning_efficiency_spec.md
#
# ## 为什么这批
#   leo 把「抗遗忘以**学习效率/更新效率**为一等指标」列为四条主线之一,
#   而该指标**至今一次没测过**(`efficiency`/`curves` 两个字段全空,
#   105 个 result tag 里 `--trace-every` 只开过 1 次)。
#   精度维度已经饱和(修好流后 option 三种配置差 ≤0.8%), 而效率维度是空白。
#
# ## 本批测什么
#   每个 run 内部**同时**跑 `naive` 与 `replay` 两条方法(驱动脚本的既有行为),
#   所以一批就给出对照:
#
#     replay 保留旧域    ⇒ 重遇到同一域时**应该学得更快** ⇒ revisit_gain > 0
#     naive  不保留      ⇒ 每次近似从头学         ⇒ revisit_gain ≈ 0
#
#   ★ 这是"持续学习有没有用"的**正向问法**, 而且它落在**未被否证**的维度上。
#
# ## 变量
#   只开 `--trace-every 10`(学习效率曲线), 其余配置与 L5 **逐字相同**。
#   `--reachable-max` 不传 ⇒ 按域数(6)查表得 0.8065(独立于臂的探针天花板)。
#
# ## 规模
#   3 个新 seed(101/202/303) × 1 run = 3 run。
#   原本每 run ~7 min, 但 trace 每 10 步评一次当前域 ⇒ 会增加评测开销, 需实测校正。
#
# ## 判据(事前写死, 见 spec §4)
#   ① 填充闸门: `n_visits` 全部 ≥1 且 `steps_to_abs` **非全空**,
#      否则判 **"指标未采集成功"**(不是"效率为 0")
#   ② 覆盖率: 报 `_none_ratio`;过高说明阈值与该域可达上限不匹配
#   ③ 复用: `mean(revisit_gain) > 0` 且自助法区间下界 > 0 ⇒ 有复用
#      **不满足就照实写"无复用"** —— 允许否定
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU=${GPU:-1}
LOG=results/oak_lmeff.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== 学习效率首次测量 =========="

# ── 启动前硬闸门 ────────────────────────────────────────────────
# ① 本批的处理变量 = `--trace-every > 0`, 必须证明三处接线在位
if ! grep -q "REACHABLE_MAX_BY_DOMAINS" tests/run_lm4_wave.py; then
  say "✗ 致命: 缺独立可达上限表 —— 绝对达标步数会退回臂自己的 max(自欺), 拒绝启动"
  exit 1
fi
if ! grep -q '"visits": {' tests/run_lm4_wave.py; then
  say "✗ 致命: 逐次访问曲线未落盘 —— 复用指标无法计算, 拒绝启动"
  exit 1
fi
if ! grep -q "revisit_gain" tests/run_lm4_wave.py; then
  say "✗ 致命: 复用增益未实现, 拒绝启动"
  exit 1
fi
# ② 默认必须是关的(旧结果逐位不变)
if ! grep -qE 'add_argument\("--trace-every", type=int, default=0' tests/run_lm4_wave.py; then
  say "✗ 致命: --trace-every 默认值被改成非 0 —— 旧结果不再可复现, 拒绝启动"
  exit 1
fi
say "✓ 闸门: 可达上限表 + 逐次访问落盘 + 复用增益 + trace 默认关 全部在位"

# ③ 单元测试必须过
if ! $PY tests/test_efficiency_metric.py >/tmp/lmeff_unit.log 2>&1; then
  say "✗ 致命: 效率指标单测未过, 拒绝启动"; tail -20 /tmp/lmeff_unit.log >>"$LOG"; exit 1
fi
say "✓ 闸门: tests/test_efficiency_metric.py 全过"

# ④ 显存
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
if [ -z "$FREE" ] || [ "$FREE" -lt 4000 ]; then
  say "✗ 致命: GPU$GPU 空闲 ${FREE:-?}MiB < 4000, 拒绝启动"
  exit 1
fi
say "✓ GPU$GPU 空闲 ${FREE}MiB"

LMEFF="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda --trace-every 10"

SEEDS="101 202 303"
say "起跑 (GPU$GPU): $SEEDS, trace-every=10, stream=revisit"
(
  for S in $SEEDS; do
    say "LMEFF seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $LMEFF \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/lmeff_s$S >>"$LOG" 2>&1
    say "LMEFF seed=$S rc=$?"
  done
  touch results/LMEFF_DONE
  say "学习效率批完成"
) >>"$LOG" 2>&1 &

echo $! > results/lmeff.pid
disown
sleep 8
say "已起跑 pid=$(cat results/lmeff.pid)"
