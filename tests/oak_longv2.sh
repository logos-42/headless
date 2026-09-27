#!/bin/bash
# oak_longv2.sh — **L2 批**: K1v2 配置 + rounds 300
#
# ## 为什么是这批
#
# K1v2(rounds=120, 修好管线, 25 臂)给出了第一次有效测量:
#   所有 option 档显著差于无 option; 闭环 ≈ 开环 (p=0.3287); term_reasons≈expired
#
# 用户假设 H2: "Option 样本量不足, 没学出来" -> 需要 2.5x 轮数才公平。
#
# 本批**只改一个变量**: rounds 120 -> 300。其余配置与 K1v2 逐字相同,
# 因此若结果与 K1v2 同向, H2 即被排除(不是样本量问题)。
#
# 对照的上轮"长测"已作废: 它的 goal 三档 alpha_mean 与 e9 逐位相同(0.8025),
# 是 coverage 未接线的空转症状。本批在**修复后的代码**上重跑。
#
# ## 矩阵 (5 档 x 5 seed = 25 臂)
#   E9 无 option | O1 fixed 开环 | O2 goal 闭环 | O3 +β_o终止 | O4 +不确定性抢占
#
# ## 判定(事前写死)
#   ① 机制闸门: 闭环档必须真的执行(option_steps>0) 且 alpha 不等于 e9
#   ② 主判据:   O2 vs O1 —— 若显著 > 则"执行方式"是症结; 若不显著则同 K1v2
#   ③ H2 判据:  O1-O4 在 rounds=300 下仍显著差于 e9 -> H2 排除
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU=${GPU:-0}
LOG=results/oak_longv2.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== L2 批启动 (K1v2 配置 + rounds 300) =========="

# ── 启动前硬闸门(不通过就拒绝启动) ──────────────────────────────
if ! grep -q "observe_action" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: oak_proposer.py 无 observe_action —— coverage 未接线, 整轮空转, 拒绝启动"
  exit 1
fi
if ! grep -q "start_blocked" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: 缺 start_blocked 计数器, 无法判定静默空转"
  exit 1
fi
say "✓ 前置检查通过: observe_action 已接线 + start_blocked 计数器存在"

# ── 显存门槛(不许和邻居抢满卡) ────────────────────────────────
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
if [ -z "$FREE" ] || [ "$FREE" -lt 8000 ]; then
  say "✗ 致命: GPU$GPU 空闲显存 ${FREE:-?}MiB < 8000MiB, 拒绝启动"
  exit 1
fi
say "✓ GPU$GPU 空闲显存 ${FREE}MiB"

# ── 配置: 与 K1v2 逐字相同, 仅 rounds 120 -> 300 ────────────────
L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda"

say "L2 起跑 (GPU$GPU): E9 + O1-O4 x 5 seed = 25 臂, rounds=300"
(
  for S in 42 1 7 13 100; do
    say "L2 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oakL2_e9_s$S >>"$LOG" 2>&1
    say "L2 E9 seed=$S rc=$?"
  done

  for M in fixed goal goal_term goal_term_override; do
    for S in 42 1 7 13 100; do
      say "L2 $M seed=$S"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oakL2_${M}_s$S >>"$LOG" 2>&1
      say "L2 $M seed=$S rc=$?"
    done
  done
  touch results/OAKL2_DONE
  say "L2 全部 25 臂完成"
) >>"$LOG" 2>&1 &

echo $! > results/oakL2.pid
disown
sleep 8
say "L2 已起跑 pid=$(cat results/oakL2.pid) (25 臂 x ~10min ≈ 4.2h)"
