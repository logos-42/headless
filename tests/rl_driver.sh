#!/bin/bash
# IDBD vs Autostep vs value 的真实回路对照 (lm4)。
# 目的: 步长分化的**免调参性**与**收益**在真任务上是否成立。
set -u
cd /work/liuyuanjie/headless
P=/work/liuyuanjie/envs/vllm-cu128/bin/python
C="--backbone mlp --norm fixed --cl-method replay --head linear --epochs-per-domain 300 --joint-steps 0 --wfr-bands 13 --lshell --pool cat --agg-path --domains 6 --fine-bins 60 --rounds 12 --proposer-k 3 --stream revisit --device cuda"
LOG=results/rl_driver.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for SEED in 1 7 42; do
  # A) IDBD, 之前必须 alpha0=0.2 才活
  say "idbd     seed=$SEED"
  CUDA_VISIBLE_DEVICES=0 $P tests/run_lm4_wave.py $C --proposer rl --rl-algo idbd \
      --rl-mu 0.05 --rl-alpha0 0.2 --seed $SEED --out results/rl_idbd_s$SEED \
      >>"$LOG" 2>&1; say "idbd     seed=$SEED rc=$?"

  # B) Autostep, 论文推荐默认 (mu=1e-2, alpha0=0.1) —— 不针对本任务调参
  say "autostep seed=$SEED (论文默认 mu=1e-2 alpha0=1e-1)"
  CUDA_VISIBLE_DEVICES=0 $P tests/run_lm4_wave.py $C --proposer rl --rl-algo autostep \
      --rl-mu 0.01 --rl-alpha0 0.1 --seed $SEED --out results/rl_auto_s$SEED \
      >>"$LOG" 2>&1; say "autostep seed=$SEED rc=$?"

  # C) Autostep, 我们之前 IDBD 才活的那个点 (看它是否不再敏感)
  say "autostep seed=$SEED (mu=0.05 alpha0=0.2)"
  CUDA_VISIBLE_DEVICES=0 $P tests/run_lm4_wave.py $C --proposer rl --rl-algo autostep \
      --rl-mu 0.05 --rl-alpha0 0.2 --seed $SEED --out results/rl_auto2_s$SEED \
      >>"$LOG" 2>&1; say "autostep2 seed=$SEED rc=$?"

  # D) 对照: 手工价值函数 (固定公式排序器)
  say "value    seed=$SEED"
  CUDA_VISIBLE_DEVICES=0 $P tests/run_lm4_wave.py $C --proposer value \
      --seed $SEED --out results/rl_value_s$SEED >>"$LOG" 2>&1; say "value    seed=$SEED rc=$?"
done
touch results/RL_DONE
say "ALL DONE"
