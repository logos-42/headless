#!/bin/bash
# LM4 夜间批量队列 (不重叠窗口 stride=32, 修正 train/test 泄漏)
# 用法: bash run_lm4_overnight.sh <seeds|ablation>
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
QUEUE="${1:-seeds}"

mkdir -p results
echo "[$(date +%F_%H:%M)] QUEUE=$QUEUE 等待当前任务结束..." >> results/overnight.log

# 等当前 run_lm4_wave 进程结束 (避免抢 GPU; 最长等 40 分钟)
for _ in $(seq 1 40); do
  pgrep -f "run_lm4_wave[.]py" >/dev/null || break
  sleep 60
done

run() {  # $1=gpu  $2=tag  其余=参数
  local gpu="$1" tag="$2"; shift 2
  mkdir -p "results/$tag"
  echo "[$(date +%F_%H:%M)] START $tag (GPU$gpu)" >> results/overnight.log
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tests/run_lm4_wave.py \
    --device cuda --stride 32 --epochs-per-domain 1000 --joint-steps 0 \
    --out "results/$tag" "$@" > "results/$tag.log" 2>&1
  local rc=$?
  echo "[$(date +%F_%H:%M)] DONE $tag (rc=$rc)" >> results/overnight.log
}

if [ "$QUEUE" = "seeds" ]; then
  # GPU0: 多 seed 验证 (主配置 ratio 1.0)。s42/s2026 已在跑, 这里补 14 个。
  for s in 7 123 999 1 2 3 11 13 17 19 23 29 37 41; do
    run 0 "lm4c_s$s" --replay-ratio 1.0 --seed "$s"
  done
elif [ "$QUEUE" = "ablation" ]; then
  # GPU1: 固定 lr=1e-3 下的 ratio 消融, 每个 ratio 3 个 seed
  for r in 0.3 0.5 2.0; do
    for s in 42 7 2026; do
      tag="lm4c_rr${r}_s${s}"
      run 1 "$tag" --replay-ratio "$r" --seed "$s"
    done
  done
fi

echo "[$(date +%F_%H:%M)] QUEUE=$QUEUE 全部完成" >> results/overnight.log
