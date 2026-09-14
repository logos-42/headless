#!/bin/bash
# oak10h_driver.sh — OaK 十小时双 GPU 实验包
#
# ┌─ GPU0: lm4 (电磁波持续学习) ────────────────────────────────────────┐
# │  Stage 1  E9/E10 补 seed 到 10 个 (可靠性: std < mean/2)            │
# │  Stage 2  E11 单次 regime shift (第 40 轮置换动作->物理区映射)       │
# │  Stage 3  E12 反复 regime shift (每 25 轮漂移一次, 共 4 次)          │
# └─────────────────────────────────────────────────────────────────────┘
# ┌─ GPU1: lm5 (文本+电磁波+因果 多模态) ──────────────────────────────┐
# │  Stage 4  E9/E10 x 5 seed                                          │
# │  Stage 5  E11 regime shift x 2 seed                                │
# └─────────────────────────────────────────────────────────────────────┘
#
# ★ 唯一变量始终是 `--oak-options` (同一 base、同一门控)。
# ★ 为什么要 regime shift: OaK 的核心主张是「内部知识让智能体能适应
#   **持续变化的动力学**」。固定域序下 T(s,a) 不变, 学到的知识不会被
#   挑战 —— 那测不出 OaK 的主张。shift 置换「动作 -> 物理区」的映射,
#   使 **T(s,a) 本身改变**, 这才是对「持续适应」的真检验。
#
# ★ 配置对齐: 总步数 = rounds x (epochs_per_domain * fine_bins / n_dom / k)
#     120 轮 x (100*18/6/3 = 100) = 12000 步
#   与 BM2 的 12 轮 x 1000 步 = 12000 **完全等量**, 没有靠堆步数占便宜。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak10h.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"

# lm5 多模态: 池 = 域 x rounds_per_domain, 总轮数给足 (oak 需要 >=40 条转移)
L5="--backbone mlp --norm fixed --cl-method replay --head linear \
--text-steps 200 --wave-epochs 150 --causal-n 20000 --d-model 128 \
--n-layers 2 --domains 6 --rounds-per-domain 20 --rounds 120 \
--stream revisit --device cuda"

# ══════════════════════════════════════════════════════════════════════
# GPU0: lm4
# ══════════════════════════════════════════════════════════════════════
(
  # 等当前那批 (results/OAK_DONE) 跑完, 避免同卡并发 (纪律: 同卡不许开两条队列)
  say "GPU0: 等待上一批 OAK_DONE ..."
  for i in $(seq 1 200); do
    [ -f results/OAK_DONE ] && break
    sleep 30
  done
  say "GPU0: 上一批完成, 开始十小时包"

  # ── Stage 1: 补 seed 到 10 个 ──
  for S in 2 3 4 5 6; do
    say "S1 E10 options=1 seed=$S"
    CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak_e10_s$S >>"$LOG" 2>&1
    RC=$?; say "S1 E10 seed=$S rc=$RC"

    say "S1 E9  options=0 seed=$S"
    CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak_e9_s$S >>"$LOG" 2>&1
    RC=$?; say "S1 E9  seed=$S rc=$RC"
  done
  touch results/OAK_S1_DONE

  # ── Stage 2: E11 单次 regime shift (第 40 轮) ──
  for S in 1 7 42; do
    for O in 1 0; do
      TAG=$([ "$O" = "1" ] && echo e11opt || echo e11noopt)
      say "S2 $TAG seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options $O --oak-gate 1 --oak-refresh 4 \
        --shift-at 40 --shift-every 0 \
        --seed $S --out results/oak_${TAG}_s$S >>"$LOG" 2>&1
      RC=$?; say "S2 $TAG seed=$S rc=$RC"
    done
  done
  touch results/OAK_S2_DONE

  # ── Stage 3: E12 反复漂移 (每 25 轮一次) ──
  for S in 1 7 42; do
    for O in 1 0; do
      TAG=$([ "$O" = "1" ] && echo e12opt || echo e12noopt)
      say "S3 $TAG seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options $O --oak-gate 1 --oak-refresh 4 \
        --shift-at 30 --shift-every 25 \
        --seed $S --out results/oak_${TAG}_s$S >>"$LOG" 2>&1
      RC=$?; say "S3 $TAG seed=$S rc=$RC"
    done
  done
  touch results/OAK_S3_DONE
  say "GPU0 全部完成"
) &
disown

# ══════════════════════════════════════════════════════════════════════
# GPU1: lm5 多模态
# ══════════════════════════════════════════════════════════════════════
(
  sleep 20
  say "GPU1: lm5 开始"
  for S in 1 7 42 13 100; do
    say "S4 lm5 E10 seed=$S"
    CUDA_VISIBLE_DEVICES=1 $PY -u tests/run_lm5_mm.py $L5 \
      --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak5_e10_s$S >>"$LOG" 2>&1
    RC=$?; say "S4 lm5 E10 seed=$S rc=$RC"

    say "S4 lm5 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=1 $PY -u tests/run_lm5_mm.py $L5 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak5_e9_s$S >>"$LOG" 2>&1
    RC=$?; say "S4 lm5 E9 seed=$S rc=$RC"
  done
  touch results/OAK5_S4_DONE
  say "GPU1 全部完成"
) &
disown

sleep 5
say "十小时包已启动 (GPU0: lm4 S1-S3 | GPU1: lm5 S4)"
