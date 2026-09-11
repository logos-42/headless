#!/bin/bash
# LM4 A 阶段多 seed 验证: WFR 30 维配置, 每路都跑 joint 天花板
# 用法: bash run_lm4_wfr_seeds.sh <gpu> <seed1> [seed2 ...]
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU="${1:?需要 GPU 编号}"; shift

for s in "$@"; do
  tag="lm4A_wfr_s$s"
  mkdir -p "results/$tag"
  echo "[$(date +%F_%H:%M)] START $tag (GPU$GPU)" >> results/lm4A_seeds.log
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u tests/run_lm4_wave.py \
    --device cuda --wfr-bands 13 --joint-steps 20000 \
    --epochs-per-domain 1000 --replay-ratio 2.0 --seed "$s" \
    --out "results/$tag" > "results/$tag.log" 2>&1
  echo "[$(date +%F_%H:%M)] DONE $tag (rc=$?)" >> results/lm4A_seeds.log
done
echo "[$(date +%F_%H:%M)] GPU$GPU 队列完成" >> results/lm4A_seeds.log
