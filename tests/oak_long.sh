#!/bin/bash
# oak_long.sh — **长测**: 修好的管线 + 更长的轮数
#
# ## 为什么是长测
#
# 用户的两个未验证假设只能靠长测检验:
#   H2  Option 样本量不足, 没学出来      -> 轮数不足 -> 知识还没成形就结束了
#   H5  Option 只在**非平稳/长期**任务中体现优势
#
# 之前的 K1 是 --rounds 120, 而 option 需要「先积累足够的转移 -> 发现技能 ->
# 再复用技能」这条链条走完才有意义。120 轮里这条链只走了不到一半。
# 本批改 **--rounds 300 (2.5x)**, 并启用**修好的发现管线**:
#   --dyn-model tabular    计数式世界模型 (线性 ridge 表达不了结构化动力学)
#   --use-subgoals 1       子目标驱动发现 (bottleneck 作为 g_o)
#   --n-regions-opt 24     表格模型下按真实状态建图
#
# ## 矩阵 (五档执行方式 x 5 seed, 全部用修好的管线)
#   E9 无 option  |  O1 fixed(开环)  |  O2 goal(闭环)  |  O3 +终止  |  O4 +抢占
#
# 判据 (用户给的): 若 O2 > O1 -> 问题是 open-loop 执行, 不是 temporal abstraction。
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak_long.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ★ 长配置: rounds 300 (2.5x), 采样池同前保证"每轮数据量"可比
L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda"

FIXED="--dyn-model tabular --use-subgoals 1 --n-regions-opt 24 --oak-gate 1 --oak-refresh 4"

say "长测启动 (GPU0): E9 + O1-O4 x 5 seed, rounds=300, 修复管线"
(
  # 先等 K1 跑完 (同卡不许两条队列)
  for i in $(seq 1 400); do
    [ -f results/OAK_K1_DONE ] && break
    sleep 30
  done
  say "K1 完成, 长测开始"

  # E9 (无 option) —— 用同一长配置与修复管线, 保证可比
  for S in 42 1 7 13 100; do
    say "LONG E9 seed=$S"
    CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 $FIXED \
      --seed $S --out results/oakL_e9_s$S >>"$LOG" 2>&1
    RC=$?; say "LONG E9 seed=$S rc=$RC"
  done
  touch results/OAKL_E9_DONE

  # O1-O4
  for M in fixed goal goal_term goal_term_override; do
    for S in 42 1 7 13 100; do
      say "LONG $M seed=$S"
      CUDA_VISIBLE_DEVICES=0 $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 $FIXED --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oakL_${M}_s$S >>"$LOG" 2>&1
      RC=$?; say "LONG $M seed=$S rc=$RC"
    done
  done
  touch results/OAKL_DONE
  say "长测完成"
) &
disown
sleep 5
say "长测已排队 (等待 K1 完成; 25 臂 x ~10 min ≈ 4.2h)"
