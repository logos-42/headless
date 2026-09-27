#!/bin/bash
# oak_k1v2.sh — K1 矩阵 **重跑版**(修复 coverage 写入 bug 之后)
#
# ## 为什么要重跑
#
# 第一版 K1 / 长测有两处执行病理, 使 K1 的 O2/O3 档**不是**对闭环 option 的有效检验:
#
#   档位               starts  steps   replan  诊断
#   fixed                31-46  34-67      0    ✓ 队列回放, 正常
#   goal                  1     94-105   94-105 ✗ **1 次启动跑满 87% 全程 = 锁死**
#   goal_term             1     94-105   94-105 ✗ 且与 goal **逐位相同** -> β_o 没生效
#   goal_term_override   2-23   72-104   72-104 ✓ 唯一行为健康
#
# 长测 (rounds=300) 更糟: goal/goal_term/override 三档的 `alpha_mean` 与 e9 **完全相同**,
# 且 `option_starts > 0` 但 `option_steps == 0` -> 闭环档**一步都没执行**。
#
# ## 根因(已在 hibs_lnn/oak_proposer.py 修掉)
#
#   `InternalKnowledge.observe_action()` 是 coverage 的**唯一写入点**,
#   但主回路从来没调过它 -> `coverage.n` 永远全零
#   -> `UncertaintyGate.allow()` 对所有动作返回 False (因为 `0 <= tau_C=1.0`)
#   -> `_goal_action()` 的每个候选都被 `continue` 掉
#   -> `best_a = None` -> 回落到 base policy -> 轨迹与 E9 逐位相同。
#
# 修: (1) `update()` 里补 `observe_action`
#     (2) 起 option 失败时**不记为"启动"**(新增 `start_blocked` 计数器),
#         否则机制计数会撒谎(旧版 `option_starts>0` 但 `option_steps==0`)。
#
# ## 本次实验设计(比第一版更严)
#
#   E9   无 option                     (基线, 哨兵: 必须 ≥ 各 O 档)
#   O1   --opt-mode fixed              固定动作序列 (open-loop)
#   O2   --opt-mode goal               目标条件 + 每步重算 (闭环)
#   O3   --opt-mode goal_term          O2 + 自适应终止 β_o(s)
#   O4   --opt-mode goal_term_override O3 + 不确定性抢占
#   5 seed x 5 档 = 25 臂
#
# ## 事前写死的判据
#
#   ① 机制生效(否则整轮作废):
#        fixed : replan_steps == 0 且 option_steps > 0
#        goal* : replan_steps ≈ option_steps > 0
#        term* : term_reasons 非空
#        override: override_count > 0
#        且 **start_blocked 必须为 0**(修复后才可能)
#   ② O2 > O1  -> 问题是 open-loop 执行, 不是 temporal abstraction
#   ③ O3/O4 ≥ O2 -> 闭环之上还需要自适应终止/抢占
#
# ## 注意: 本驱动不写任何"结论", 只落盘数字
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
LOG=results/oak_k1v2.log
GPU=${GPU:-1}                     # ★ GPU0 被邻居占满(40GB/96%), 用 GPU1(~18GB 空闲)
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ── 前置检查: 修复必须真的在盘上(否则又是静默空转的一轮) ──────────────
if ! grep -q "observe_action" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: oak_proposer.py 里没有 observe_action —— coverage 写入未接, 整轮会空转, 拒绝启动"
  exit 1
fi
if ! grep -q "start_blocked" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: 缺少 start_blocked 计数器, 无法判定静默空转"
  exit 1
fi
say "前置检查通过: observe_action 已接线 + start_blocked 计数器存在"
say "GPU=$GPU"

L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 120 \
--proposer-k 3 --stream revisit --device cuda"

say "K1v2 启动 (GPU$GPU): E9 + O1-O4 x 5 seed = 25 臂"
(
  # ── E9 基线(无 option) ────────────────────────────────────────────
  for S in 42 1 7 13 100; do
    say "K1v2 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oak_k1v2_e9_s$S >>"$LOG" 2>&1
    say "K1v2 E9 seed=$S rc=$?"
  done
  # ── O1-O4 ─────────────────────────────────────────────────────────
  for M in fixed goal goal_term goal_term_override; do
    for S in 42 1 7 13 100; do
      say "K1v2 mode=$M seed=$S"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oak_k1v2_${M}_s$S >>"$LOG" 2>&1
      say "K1v2 mode=$M seed=$S rc=$?"
    done
  done
  touch results/OAK_K1V2_DONE
  say "K1v2 完成"
) &
disown
sleep 5
say "K1v2 已启动 (25 臂, 预计 ~100 分钟)"
