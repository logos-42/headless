#!/bin/bash
# oak_k1.sh — 用户指出的**关键实验** (K1 矩阵): 区分 temporal abstraction 与
#             open-loop option execution
#
# ## 背景 (用户 2026-09-14 指导, 我接受)
#
#   "Option = [a1,a2,...,ak]" 其实**不是一个 adaptive 的 Option**,
#   它像"我发现了一段过去有效的动作脚本, 现在把这段脚本重新播放"。
#   底层 policy 原本每步按当前状态重算 a_t = π(s_t); option 一旦变成固定序列,
#   状态变化就不再触发动作重算 -> 适应性下降。**这正好解释 D1。**
#
#   正确形式:  Option = (I_o, g_o, π_o, β_o)
#              **抽象的是目标, 不是动作。**
#
#   第一版不要给 option 自己的新 policy, 直接用:
#              a_t = π_base(s_t, g_o)      <- 给 base policy 一个高层条件
#
# ## 本实验 (唯一变量 = Option 的执行方式)
#
#   E9                  无 Option                          (基线, 已有 10 seed)
#   O1  --opt-mode fixed              固定动作序列 (open-loop, 即 D1 的病灶)
#   O2  --opt-mode goal               目标条件 + **每步重算动作** (闭环)
#   O3  --opt-mode goal_term          O2 + 自适应终止 β_o(s)
#   O4  --opt-mode goal_term_override O3 + 不确定性抢占 (option 从属于当前证据)
#
# ## 判据 (用户给的)
#   若 O2 > O1  -> 问题不是 temporal abstraction, 而是 **open-loop 执行**
#   若 O3/O4 继续提升 -> option 的价值需要建立在**闭环执行**上
#
# ## 机制生效性必须可见 (否则结论无效)
#   fixed  : replan_steps = 0
#   goal*  : replan_steps ≈ option_steps (每步都在重算)
#   term*  : term_reasons 非空
#   override: override_count > 0
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak_k1.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"

say "K1 矩阵启动 (GPU0): O1-O4 x 5 seed"
(
  for M in fixed goal goal_term goal_term_override; do
    for S in 42 1 7 13 100; do
      say "K1 mode=$M seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oak_k1_${M}_s$S >>"$LOG" 2>&1
      RC=$?; say "K1 mode=$M seed=$S rc=$RC"
    done
  done
  touch results/OAK_K1_DONE
  say "K1 完成"
) &
disown
sleep 5
say "K1 已启动 (20 臂, 预计 ~50 分钟)"
