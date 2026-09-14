#!/bin/bash
# oak10h_ext.sh — 十小时包的**延伸队列** (在上一个包之后接续, 把两个 GPU 都填满)
#
# 配比核算 (实测):
#   lm4 每臂 ~12 min  (120 轮 x 100 步)
#   lm5 每臂 ~4.7 min (120 轮 x 120 步)
#   -> 原包里 GPU0 约 5.4h, GPU1 只有 ~0.8h。必须补 GPU1。
#
# GPU0 追加: 步长 E 表 (用户表里的 E1~E5)
#   E2 IDBD / E3 idbd-raw / E4 autostep / E5 cidbd  (E1 fixed-lr = 无步长自适应基线)
#   ★ 全部跑在 **oak proposer** 上, 这样与 E9/E10 同一 base, 可与其它臂直接对比。
#   ★ alpha0=0.2 是实测唯一"活着"的点 (alpha0<0.2 时步长死亡; 见
#     docs/stepsize_idbd_results.md)
#
# GPU1 追加: lm5 加长配置 x 10 seed
#   text-steps 400 / wave-epochs 400 / causal-n 40000 / rounds 200
#   -> 单臂约 3.3 倍 -> ~15 min/臂, 10 臂 x 2 = 300 min = 5h
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak10h_ext.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"

L5L="--backbone mlp --norm fixed --cl-method replay --head linear \
--text-steps 400 --wave-epochs 400 --causal-n 40000 --d-model 128 \
--n-layers 2 --domains 6 --rounds-per-domain 25 --rounds 200 \
--stream revisit --device cuda"

# ══════════════════════════════════════════════════════════════════════
# GPU0: 步长 E 表 (接在 S3 之后)
# ══════════════════════════════════════════════════════════════════════
(
  say "GPU0-ext: 等待 S3 完成 ..."
  for i in $(seq 1 1200); do
    [ -f results/OAK_S3_DONE ] && break
    sleep 30
  done
  say "GPU0-ext: 开始步长 E 表"

  # E1: 固定学习率基线 (无步长自适应) —— 用 autostep 的 mulr 太大模拟不了,
  # 这里用 value 臂作固定基线不合适; 直接跑 rl-algo=idbd-raw 的对照在 E3。
  # 真正"固定"的等价物是 --rl-mu 0 (步长不更新) -> beta 恒 0 -> alpha 恒 alpha0
  say "E1 fixed-alpha (rl-mu=0, 步长冻结)"
  CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
    --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
    --rl-mu 0.0 --rl-alpha0 0.2 --rl-algo idbd \
    --seed 42 --out results/oak_e1_fixedalpha_s42 >>"$LOG" 2>&1
  RC=$?; say "E1 rc=$RC"

  for S in 42 1 7; do
    for A in idbd idbd-raw autostep cidbd; do
      say "E-algo $A seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --rl-mu 0.05 --rl-alpha0 0.2 --rl-algo $A \
        --seed $S --out results/oak_algo_${A}_s$S >>"$LOG" 2>&1
      RC=$?; say "E-algo $A seed=$S rc=$RC"
    done
  done
  touch results/OAK_EXT0_DONE
  say "GPU0-ext 完成"
) &
disown

# ══════════════════════════════════════════════════════════════════════
# GPU1: lm5 加长配置 (接在 S4 之后)
# ══════════════════════════════════════════════════════════════════════
(
  say "GPU1-ext: 等待 lm5 S4 完成 ..."
  for i in $(seq 1 1200); do
    [ -f results/OAK5_S4_DONE ] && break
    sleep 30
  done
  say "GPU1-ext: 开始 lm5 加长配置"

  for S in 2 3 4 5 6 8 9 11 12 14; do
    say "S5 lm5L E10 seed=$S"
    CUDA_VISIBLE_DEVICES=1 $PY -u tests/run_lm5_mm.py $L5L \
      --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak5L_e10_s$S >>"$LOG" 2>&1
    RC=$?; say "S5 lm5L E10 seed=$S rc=$RC"

    say "S5 lm5L E9 seed=$S"
    CUDA_VISIBLE_DEVICES=1 $PY -u tests/run_lm5_mm.py $L5L \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak5L_e9_s$S >>"$LOG" 2>&1
    RC=$?; say "S5 lm5L E9 seed=$S rc=$RC"
  done
  touch results/OAK5_EXT_DONE
  say "GPU1-ext 完成"
) &
disown

sleep 5
say "延伸队列已启动 (GPU0: 步长 E 表 | GPU1: lm5 加长 x10 seed)"
