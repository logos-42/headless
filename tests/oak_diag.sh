#!/bin/bash
# oak_diag.sh — 诊断: **为什么 Options 有害** (E9/E10 主结论: any_time 显著下降)
#
# 主结论 (n=10, 两档配置都成立):
#   lm4  any_time Δ=-0.0119 (-1.6%)  t=-5.02  p<0.0001   E9(无Options) 更好
#   lm5  any_time Δ=-0.0189 (-1.9%)  t=-5.14  p<0.0001   E9 更好
#   机制**确实生效** (option_starts 38~75, option_steps 40~83),
#   所以不能按"机制没触发"来免责 -> 必须查清原因。
#
# 三个假设, 各配一个可判定的实验:
#   H1  option 执行 = 提交一段固定动作序列 -> 减少自适应性
#   H2  option 执行降低了探索多样性 -> 覆盖度更窄
#   H3  option 是从**过期的**转移模型里发现的 -> 结构本身不对
#
# D1 (H1/H2): opt_frac 剂量-反应。只让一部分轮次启用 option。
#     若伤害随 opt_frac **单调放大** -> 指向 H1/H2 (option 执行本身的问题)。
#     若伤害与用量无关 -> 另有来源。
#     ★ opt_frac=0.0 应当**退化为 E9** —— 这是本实验自身的正确性哨兵。
# D2 (H3): 提高 world model 重建频率 (refresh 2/8/16)。
#     若 H3 成立, 更频繁重建应当**减轻**伤害。
# D3: 用**修好的 α 统计**重跑步长 E 表 (早先 alpha_std 取自一个没被更新的
#     死对象, 四个算法报出同一个值, 看着像"算法不生效")。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak_diag.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"
OAK="--proposer oak --oak-gate 1 --oak-refresh 4"

say "诊断队列启动 (GPU0)"
(
  # ── D1: opt_frac 剂量-反应 ──
  for F in 0.0 0.25 0.5 1.0; do
    for S in 42 1 7; do
      say "D1 opt_frac=$F seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 $OAK \
        --oak-options 1 --opt-frac $F \
        --seed $S --out results/oak_frac${F}_s$S >>"$LOG" 2>&1
      RC=$?; say "D1 opt_frac=$F seed=$S rc=$RC"
    done
  done
  touch results/OAK_D1_DONE

  # ── D2: world model 重建频率 ──
  for R in 2 8 16; do
    for S in 42 1 7; do
      say "D2 refresh=$R seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh $R \
        --seed $S --out results/oak_ref${R}_s$S >>"$LOG" 2>&1
      RC=$?; say "D2 refresh=$R seed=$S rc=$RC"
    done
  done
  touch results/OAK_D2_DONE

  # ── D3: 步长 E 表 (修好的 α 统计) ──
  for S in 42 1 7; do
    for A in idbd idbd-raw autostep cidbd; do
      say "D3 algo=$A seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --rl-mu 0.05 --rl-alpha0 0.2 --rl-algo $A \
        --seed $S --out results/oak_astat_${A}_s$S >>"$LOG" 2>&1
      RC=$?; say "D3 algo=$A seed=$S rc=$RC"
    done
  done
  touch results/OAK_D3_DONE
  say "诊断队列完成"
) &
disown
sleep 5
say "诊断队列已启动 (D1 opt_frac x4 | D2 refresh x3 | D3 步长 x4, 共 33 臂)"
