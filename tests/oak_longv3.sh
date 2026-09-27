#!/bin/bash
# oak_longv3.sh — **L3 批**: L2 配置 + β_o 区域判据修复
#
# ## 为什么是这批
#
# L2(rounds=300, 25 臂)把 K1v2 的问题钉死了一半:
#   · 闸门全过; O2(闭环) vs O1(开环) Δ=+0.0090 t=+2.19 p=0.0286 **显著**
#     -> **H2(样本量不足)被证实**: rounds=120 时不显著(p=0.3287),
#        rounds=300 时显著且方向正确。
#   · 但所有 option 档仍显著差于 e9 (Δ=-0.0149…-0.0257, p<=0.0001),
#     且 `goal` 57 次启动里 **56 次 expired、仅 1 次 goal_reached**。
#
# 那个「option 从未到达目标」直接指向 β_o 的终止判据。逐次执行诊断
# (KeyDoor, 2737 次执行)证明判据在离散/高维状态空间里**退化**:
#   `‖s − goal_center‖ < eps` 中 eps 由 `init_radius` 推出, 只接受精确到达,
#   而可达的最小距离远大于它 -> 因「达成」触发 **0 次**。
#   π_o 本身是正常的(把距离推近, 0 次推远)。
#
# ## 本批**只改一个变量**
#   Option 的 I_o 与 β_o 判据: 距离阈值 -> **k-means Voronoi 区域归属**
#   (`in_goal_region` / `region_of`; 无阈值、覆盖全空间、离散空间不退化)
#   其余与 L2 逐字相同。故:
#     若 `goal_reached` 出现且差距收窄 -> 终止判据是症结
#     若 `goal_reached` 出现但差距不减 -> 症结在别处 (目标选取/发现方法)
#
# ## 矩阵 (5 档 x 5 seed = 25 臂)
#   E9 无 option | O1 fixed 开环 | O2 goal 闭环 | O3 +β_o终止 | O4 +不确定性抢占
#
# ## 判定(事前写死)
#   ① 机制闸门(analyze_k1v2.py): 全部臂必须过闸门, 否则对照无效
#   ② ★ 新判据 1: `term_reasons` 里 `goal_reached` 必须显著 > L2 的 ~1 次
#   ③ ★ 新判据 2: option 档 vs e9 的 Δ 若仍显著为负, 则终止判据被排除
#                 (K1v2 已排除"执行方式", L2 已排除"样本量", 本批排除"终止判据")
#   ④ 诚实基线: 若 Δ 仍为负 -> 剩余症结只能来自**目标选取/发现方法**
#                 (即论文说的 reward-respecting subtask, 尚未实现)
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU=${GPU:-0}
LOG=results/oak_longv3.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== L3 批启动 (L2 配置 + β_o 区域判据修复) =========="

# ── 启动前硬闸门(不通过就拒绝启动) ──────────────────────────────
if ! grep -q "observe_action" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: oak_proposer.py 无 observe_action —— coverage 未接线, 整轮空转, 拒绝启动"
  exit 1
fi
if ! grep -q "start_blocked" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: 缺 start_blocked 计数器, 无法判定静默空转"
  exit 1
fi
# ★ 本批的**处理变量**必须在位, 否则跑出来的还是 L2 的重复
if ! grep -q "def in_goal_region" hibs_lnn/option_manager.py; then
  say "✗ 致命: option_manager.py 无 in_goal_region —— 区域判据未部署, 本批等于重跑 L2, 拒绝启动"
  exit 1
fi
if ! grep -q "if self.in_goal_region(s):" hibs_lnn/option_manager.py; then
  say "✗ 致命: terminated() 未接上区域判据 —— 处理变量不在位, 拒绝启动"
  exit 1
fi
say "✓ 前置检查通过: observe_action 接线 + start_blocked 计数器 + **区域判据已在位**"

# ── 显存门槛(不许和邻居抢满卡) ────────────────────────────────
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
if [ -z "$FREE" ] || [ "$FREE" -lt 8000 ]; then
  say "✗ 致命: GPU$GPU 空闲显存 ${FREE:-?}MiB < 8000MiB, 拒绝启动"
  exit 1
fi
say "✓ GPU$GPU 空闲显存 ${FREE}MiB"

# ── 配置: 与 L2 逐字相同 ────────────────────────────────────────
L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda"

say "L3 起跑 (GPU$GPU): E9 + O1-O4 x 5 seed = 25 臂, rounds=300, 区域判据"
(
  for S in 42 1 7 13 100; do
    say "L3 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oakL3_e9_s$S >>"$LOG" 2>&1
    say "L3 E9 seed=$S rc=$?"
  done

  for M in fixed goal goal_term goal_term_override; do
    for S in 42 1 7 13 100; do
      say "L3 $M seed=$S"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oakL3_${M}_s$S >>"$LOG" 2>&1
      say "L3 $M seed=$S rc=$?"
    done
  done
  touch results/OAKL3_DONE
  say "L3 全部 25 臂完成"
) >>"$LOG" 2>&1 &

echo $! > results/oakL3.pid
disown
sleep 8
say "L3 已起跑 pid=$(cat results/oakL3.pid) (25 臂 x ~7min ≈ 3h)"
