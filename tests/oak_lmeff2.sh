#!/bin/bash
# oak_lmeff2.sh — 学习效率第一次测量的**扩 seed + 判别性对照**
#
# 承接 docs/lmeff_verdict.md。该报告 §3 明写了一个未决混淆:
#   `replay` 的**总算力更大** ⇒ 可能只是"训练更多/更强正则", 不是抗遗忘。
#
# ## 本批的判别性设计(三臂, 同一个 run 内并行跑, 天然配对)
#   naive        不复习                           (基线)
#   replay       复习**已见过的**旧域(域均衡随机)   (现有机制)
#   replay-self  同样 k 个额外样本、同样 1 次优化步,
#                但样本**全部来自当前域**          (判别性对照)
#
#   判据(事前写死):
#     `replay-self ≈ naive` 且 `replay >> naive`  ⇒ 效果来自**旧域数据本身** ✓
#     `replay-self ≈ replay`                      ⇒ 效果只是**额外算力/正则** ⇒ 结论必须降级
#
#   ★ 另外查清一件事: `replay` 的实现(`run_lm4_wave.py:979-990`)**本来就是域均衡
#     随机复习**(每个见过的域等量、域内随机抽样本), 不是"针对即将遗忘的域"。
#     所以"随机复习 vs 针对性复习"这个对照**已经含在现有臂里**了 ——
#     真正未解决的混淆只有"额外算力"这一个, 本批正对它。
#
# ## 规模
#   10 个新 seed(401…410), 每 run 3 臂。两条队列, seed 不相交。
#   实测单 run ~12 min(2 臂) ⇒ 3 臂约 18 min; 5 run/队列 ⇒ 约 1.5 h。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak_lmeff2.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== 学习效率 扩 seed + 判别性三臂 =========="

# ── 启动前硬闸门 ────────────────────────────────────────────────
# ① 三臂必须在位
if ! grep -q '"replay-self", True, "self"' tests/run_lm4_wave.py; then
  say "✗ 致命: 判别性对照臂 replay-self 不在 runs 里, 拒绝启动"; exit 1
fi
if ! grep -q 'replay_src=_rsrc' tests/run_lm4_wave.py; then
  say "✗ 致命: replay_src 没透传进 run_experiment, 第三个臂会退化, 拒绝启动"; exit 1
fi
if ! grep -q 'replay_src == "self" and _self_buf is not None' tests/run_lm4_wave.py; then
  say "✗ 致命: self 分支不在 replay 块里, 拒绝启动"; exit 1
fi
# ② 表格遍历必须是"本次跑过的臂", 不能硬编码(硬编码会漏掉 replay-self)
if grep -q 'for k in ("naive", "replay"):' tests/run_lm4_wave.py; then
  say "✗ 致命: 报告表格仍硬编码 naive/replay ⇒ replay-self 结果不进报告, 拒绝启动"; exit 1
fi
# ③ 旧口径必须不变
if ! grep -qE 'add_argument\("--trace-every", type=int, default=0' tests/run_lm4_wave.py; then
  say "✗ 致命: --trace-every 默认被改成非 0, 拒绝启动"; exit 1
fi
if ! grep -q "REACHABLE_MAX_BY_DOMAINS" tests/run_lm4_wave.py; then
  say "✗ 致命: 缺独立可达上限表, 拒绝启动"; exit 1
fi
say "✓ 闸门: 三臂 + replay_src 透传 + self 分支 + 报告不硬编码 + trace 默认 0 + 可达上限"

if ! $PY tests/test_efficiency_metric.py >/tmp/lmeff2_unit.log 2>&1; then
  say "✗ 致命: 效率指标单测未过"; tail -20 /tmp/lmeff2_unit.log >>"$LOG"; exit 1
fi
say "✓ 闸门: 单测全过"

GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
SEEDS_A="401 402 403 404 405"
SEEDS_B="406 407 408 409 410"
# ④ 队列间 seed 不相交(硬校验)
_overlap=$(comm -12 <(echo $SEEDS_A | tr ' ' '\n' | sort) <(echo $SEEDS_B | tr ' ' '\n' | sort))
if [ -n "$_overlap" ]; then say "✗ 致命: 两队列 seed 相交: $_overlap"; exit 1; fi
say "✓ 闸门: 两队列 seed 不相交 (A=[$SEEDS_A] B=[$SEEDS_B])"

for G in $GPU_A $GPU_B; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G 2>/dev/null | head -1)
  if [ -z "$F" ] || [ "$F" -lt 3000 ]; then
    say "✗ 致命: GPU$G 空闲 ${F:-?}MiB < 3000, 拒绝启动"; exit 1
  fi
  say "✓ GPU$G 空闲 ${F}MiB"
done

BASE="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda --trace-every 10"
OAK="--proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4"

run_queue(){   # $1=tag $2=gpu $3=seeds
  local TAG=$1 G=$2
  for S in $3; do
    say "[$TAG] seed=$S 起跑 (GPU$G)"
    CUDA_VISIBLE_DEVICES=$G $PY -u tests/run_lm4_wave.py $BASE $OAK \
      --seed $S --out results/lmeff2_${TAG}_s$S >>"$LOG" 2>&1
    say "[$TAG] seed=$S rc=$?"
  done
  touch results/LMEFF2_${TAG}_DONE
  say "[$TAG] 队列完成"
}

# 显存预算: 两队列 ×2 GiB ≤ 空闲 60%(同卡多队列规则, 2026-09-29 修订)
say "起跑: A(GPU$GPU_A)=[$SEEDS_A]  B(GPU$GPU_B)=[$SEEDS_B]  各 3 臂"
( run_queue A $GPU_A "$SEEDS_A" ) &
( run_queue B $GPU_B "$SEEDS_B" ) &
touch results/LMEFF2_STARTED
disown -a
sleep 10
say "两队列已起跑"
