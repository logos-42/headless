#!/bin/bash
# LM4 队列 2 — 给最优配置 ratio=2.0 补 seed (n=3 → 13)
# 背景: ratio 曲线 0.3/0.5/1.0/2.0 = 0.2501/0.3512/0.3857/0.3955,
#       2.0 最好但只有 3 个 seed, 需要加固统计。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python

for s in 1 2 3 11 13 17 19 23 29 37; do
  tag="lm4c_rr2.0_s$s"
  mkdir -p "results/$tag"
  echo "[$(date +%F_%H:%M)] START $tag" >> results/overnight2.log
  CUDA_VISIBLE_DEVICES=0 "$PY" -u tests/run_lm4_wave.py --device cuda --stride 32 \
    --epochs-per-domain 1000 --joint-steps 0 --replay-ratio 2.0 --seed "$s" \
    --out "results/$tag" > "results/$tag.log" 2>&1
  echo "[$(date +%F_%H:%M)] DONE $tag (rc=$?)" >> results/overnight2.log
done
echo "[$(date +%F_%H:%M)] QUEUE2 全部完成" >> results/overnight2.log
