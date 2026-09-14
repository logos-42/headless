#!/bin/bash
# bm_driver.sh — benchmark 全流程驱动 (stage1 收尾 -> 频率 profile -> stage2 -> lm5)
# 在服务器上以 setsid 运行, 不依赖外层会话轮询。
set -u
cd /work/liuyuanjie/headless

PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU0=0
GPU1=1
LOGDIR=results
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ── stage 1 收尾: 等 value x3 与 bins x12 出现 ──
SKIP_STAGE1="${SKIP_STAGE1:-0}"
NV=0; NB=0
log "等待 stage 1 (value x3 / bins x12) ... (SKIP_STAGE1=$SKIP_STAGE1)"
if [ "$SKIP_STAGE1" = "0" ]; then
for i in $(seq 1 240); do
  NV=$(ls -d ${LOGDIR}/bm_value_s*/ 2>/dev/null | wc -l)
  NB=$(ls -d ${LOGDIR}/bm_bins_*_s*/ 2>/dev/null | wc -l)
  if [ "$NV" -ge 3 ] && [ "$NB" -ge 12 ]; then break; fi
  sleep 30
done
fi
log "stage 1 完成: value=$NV bins=$NB"

# ── 提取 value 臂的频率 profile (取 seed 42) ──
PROF=${LOGDIR}/bm_value_s42/lm4_wave_results.json
if [ -f "$PROF" ]; then
  cp "$PROF" ${LOGDIR}/freq_profile.json
  log "频率 profile 就绪: ${LOGDIR}/freq_profile.json"
else
  log "!! 找不到 $PROF — random-matched 将退化为均匀随机 (会在结果里标注)"
fi

# ── stage 2 (GPU0): value-nofb / random / random-matched x 3 seed ──
log "stage 2 启动 (GPU0)"
for S in 42 1 7; do
  for P in value-nofb random random-matched; do
    log "  GPU0 $P seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU0 $PY -u tests/run_lm4_wave.py \
      --backbone mlp --norm fixed --cl-method replay --head linear \
      --epochs-per-domain 1000 --joint-steps 0 --wfr-bands 13 --lshell \
      --pool cat --agg-path --domains 6 --fine-bins 18 --rounds 12 \
      --proposer $P --stream fixed --seed $S --device cuda \
      --freq-profile ${LOGDIR}/freq_profile.json \
      --out ${LOGDIR}/bm_${P}_s${S} >> ${LOGDIR}/bm_stage2.log 2>&1
    log "  GPU0 $P seed=$S rc=$?"
  done
done
log "stage 2 完成"

# ── stage 3 (GPU1): lm5 benchmark, 4 流 x 3 seed ──
log "stage 3 启动 (GPU1): lm5"
for S in 42 1 7; do
  for ST in fixed perm revisit nonstationary; do
    log "  GPU1 lm5 $ST seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU1 $PY -u tests/run_lm5_mm.py \
      --backbone hybrid --norm fixed --cl-method replay --head linear \
      --text-steps 300 --wave-epochs 300 --causal-n 20000 \
      --d-model 192 --n-layers 2 --domains 6 --rounds 8 \
      --proposer fixed --stream $ST --seed $S --device cuda \
      --out ${LOGDIR}/bm5_${ST}_s${S} >> ${LOGDIR}/bm5.log 2>&1
    log "  GPU1 lm5 $ST seed=$S rc=$?"
  done
done

# ── stage 4 (GPU1): lm5 价值函数臂 ──
# 注意顺序: 必须**先**跑完 value 臂, 才有 profile 给 random-matched 用。
log "stage 4a 启动 (GPU1): lm5 value x 3 seed"
for S in 42 1 7; do
  log "  GPU1 lm5 value seed=$S"
  CUDA_VISIBLE_DEVICES=$GPU1 $PY -u tests/run_lm5_mm.py \
    --backbone hybrid --norm fixed --cl-method replay --head linear \
    --text-steps 300 --wave-epochs 300 --causal-n 20000 \
    --d-model 192 --n-layers 2 --domains 6 --rounds 8 \
    --proposer value --stream fixed --seed $S --device cuda \
    --out ${LOGDIR}/bm5_value_s${S} >> ${LOGDIR}/bm5.log 2>&1
  log "  GPU1 lm5 value seed=$S rc=$?"
done

P5=${LOGDIR}/bm5_value_s42/lm5_mm_results.json
if [ -f "$P5" ]; then
  cp "$P5" ${LOGDIR}/freq_profile_lm5.json
  log "lm5 频率 profile 就绪"
else
  log "!! 缺 $P5 — lm5 random-matched 将退化为均匀随机"
fi

log "stage 4b 启动 (GPU1): lm5 value-nofb / random-matched x 3 seed"
for S in 42 1 7; do
  for P in value-nofb random-matched; do
    log "  GPU1 lm5 $P seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU1 $PY -u tests/run_lm5_mm.py \
      --backbone hybrid --norm fixed --cl-method replay --head linear \
      --text-steps 300 --wave-epochs 300 --causal-n 20000 \
      --d-model 192 --n-layers 2 --domains 6 --rounds 8 \
      --proposer $P --stream fixed --seed $S --device cuda \
      --freq-profile ${LOGDIR}/freq_profile_lm5.json \
      --out ${LOGDIR}/bm5_${P}_s${S} >> ${LOGDIR}/bm5.log 2>&1
    log "  GPU1 lm5 $P seed=$S rc=$?"
  done
done

log "全部完成"
touch ${LOGDIR}/BM_DONE
