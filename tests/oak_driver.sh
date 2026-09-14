#!/bin/bash
# oak_driver.sh — OaK 结构件在 lm4 主回路上的第一个正式对照
#
# 目的: 检验用户架构主张里最关键的一跳 ——
#         Prediction -> Abstraction -> Option -> Planning -> Control
#       是否比 state -> RL policy -> action 更适合持续变化的磁通动力学。
#
# 设计 (E9 vs E10 是**唯一**变量):
#   E10 = --proposer oak --oak-options 1   (+Options / 时间抽象)
#   E9  = --proposer oak --oak-options 0   (同一 base、同一门控, 只是不用 option)
#
# ★ 为什么 --rounds 120 而不是之前 benchmark 的 12:
#   dynamics model 的特征是 [state(6), onehot(action)(18)] = 24 维。
#   12 条转移对 24 维是**严重欠定**(12x24 的矩阵 rank<=12), 集成分歧
#   与 option 发现都会变成噪声。120 条转移给出 ~5 样本/参数, 才站得住。
#   ★ 同时把 epochs-per-domain 从 300 降到 100, 使
#     `总步数 = 120 轮 x (100*18/6/3=100) 步 = 12000` —— 与 BM2 的
#     12 轮 x 1000 步 = 12000 完全等量。这样既喂饱了动力学模型,
#     又**没有**靠堆步数占便宜。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
G0=0; G1=1
COMMON="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"
LOG=results/oak_driver.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "OaK E9/E10 对照启动 (5 seed x 2 臂, GPU0)"
for S in 1 7 42 13 100; do
  # E10: 有 Options (时间抽象)
  say "E10 options=1 seed=$S"
  CUDA_VISIBLE_DEVICES=$G0 $PY -u tests/run_lm4_wave.py $COMMON \
    --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
    --seed $S --out results/oak_e10_s$S >>"$LOG" 2>&1
  RC=$?; say "E10 seed=$S rc=$RC"

  # E9: 同样体系, 只是关掉选项
  say "E9  options=0 seed=$S"
  CUDA_VISIBLE_DEVICES=$G0 $PY -u tests/run_lm4_wave.py $COMMON \
    --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
    --seed $S --out results/oak_e9_s$S >>"$LOG" 2>&1
  RC=$?; say "E9  seed=$S rc=$RC"
done
touch results/OAK_DONE
say "ALL DONE"
