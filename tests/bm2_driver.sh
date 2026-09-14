#!/bin/bash
# bm2_driver.sh — 修正后的价值函数 benchmark (con 已实现 / sim 自适应 / 池 60)
# 旧结果 (bm_value*, bm_random*, bm_random-matched*) 全部作废, 已删除。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
G0=0; G1=1
COMMON="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 1000 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 60 --rounds 12 --proposer-k 3 --device cuda"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "stage A (GPU0): value x 3 seed  [真实价值函数: sim/con/cov/fb 全活]"
for S in 42 1 7; do
  CUDA_VISIBLE_DEVICES=$G0 $PY -u tests/run_lm4_wave.py $COMMON \
    --proposer value --seed $S --out results/bm2_value_s$S >> results/bm2.log 2>&1
  log "  value seed=$S rc=$?"
done
cp results/bm2_value_s42/lm4_wave_results.json results/freq_profile2.json 2>/dev/null \
  && log "频率 profile2 就绪"

log "stage B (GPU0): value-nofb / random / random-matched x 3 seed"
for S in 42 1 7; do
  for P in value-nofb random random-matched; do
    CUDA_VISIBLE_DEVICES=$G0 $PY -u tests/run_lm4_wave.py $COMMON \
      --proposer $P --seed $S --freq-profile results/freq_profile2.json \
      --out results/bm2_${P}_s$S >> results/bm2.log 2>&1
    log "  $P seed=$S rc=$?"
  done
done

log "stage C (GPU1): lm5 value 系 (con 同样修复后)"
for S in 42 1 7; do
  CUDA_VISIBLE_DEVICES=$G1 $PY -u tests/run_lm5_mm.py \
    --backbone hybrid --norm fixed --cl-method replay --head linear \
    --text-steps 300 --wave-epochs 300 --causal-n 20000 \
    --d-model 192 --n-layers 2 --domains 6 --rounds 8 \
    --proposer value --seed $S --device cuda \
    --out results/bm5b_value_s$S >> results/bm5b.log 2>&1
  log "  lm5 value seed=$S rc=$?"
done
cp results/bm5b_value_s42/lm5_mm_results.json results/freq_profile_lm5b.json 2>/dev/null
for S in 42 1 7; do
  for P in value-nofb random-matched; do
    CUDA_VISIBLE_DEVICES=$G1 $PY -u tests/run_lm5_mm.py \
      --backbone hybrid --norm fixed --cl-method replay --head linear \
      --text-steps 300 --wave-epochs 300 --causal-n 20000 \
      --d-model 192 --n-layers 2 --domains 6 --rounds 8 \
      --proposer $P --seed $S --device cuda \
      --freq-profile results/freq_profile_lm5b.json \
      --out results/bm5b_${P}_s$S >> results/bm5b.log 2>&1
    log "  lm5 $P seed=$S rc=$?"
  done
done
log "全部完成"
touch results/BM2_DONE
