#!/bin/bash
# oak_longv5.sh — **L5 批**: 方差效应的独立复现(事前注册)
#
# 规格: docs/l5_variance_replication_spec.md(**在任何数据产生之前写下**)
#
# ## 待复现的发现
#   L4(n=29)发现三个闭环 option 档的跨 seed 方差约为基线 `e9` 的 1/3~1/4
#   (sd 0.0258 → 0.0124/0.0135/0.0150), F p ≤ 0.005, 自助法方差比 95% CI 全部排除 1。
#   **那是事后发现的(本来在找均值差异)⇒ 属假说生成, 不是确证。必须独立复现。**
#
# ## ★ 处理变量 = **代码修复, 不是 flag**
#   首版设计把独立设置选成 `--stream nonstationary`, 启动后按"处理变量必须回读确认"
#   去日志里找 `[stream]` 行 —— **它不存在**。核实发现:
#   `run_lm4_wave.py:666` 的 `if proposer == "bins":` 里装着整段流调度, 而
#   实际域由主循环 `dd = domains[step] if step<len(domains) else domains[-1]`
#   决定 ⇒ **oak 路径下 `--stream` 是死参数**, 实际流 = "前 6 轮各域一次,
#   之后 294 轮全在最后一个域"(实测域5 占 295/300)。
#   见 `docs/lm4_stream_bug.md`。
#
#   修复后, 主循环按 `dom_seq[step]` 取域 ⇒ 同一个 `--stream revisit`
#   **才第一次真正产生"每轮随机抽域"的持续学习流**。
#   所以本批的配置串与 L4 **逐字相同**(flag 一个不变), 处理变量是**代码修复**;
#   闸门必须同时证明"配置相同"与"修复在位"两件事。
#
# ## 规模
#   4 臂(e9 / goal / goal_term / goal_term_override) × 18 个新 seed(28…41, 43…46)
#   = 72 run。按 L4 实测 ≈ 10.6 min/run ⇒ 约 **12.7h**。
#   (注: 流变成真正的持续学习后, 每轮的数据加载/训练量可能变化, 耗时需实测校正)
#
#   n=18 的功效: 若真实方差比=4, `z = log(4)/sqrt(4/17) = 2.86 > 2.80` ⇒ 功效≈0.81。
#   若真实方差比只有 2, 功效仅≈0.37 —— **事前承认**, 届时交付**方差比的置信区间**。
#
# ## 判定(事前写在 spec, 由 analyze_l4_extension.py 实现)
#   H2(主要):  `goal_term` 与 `goal_term_override` **两个都** F p<0.05 且
#               自助法 CI 下界 > 1 ⇒ 复现; 只有一个 ⇒ 部分复现; 都不满足 ⇒ 未复现
#   H2b(机制): `goal`(无 β_o)是否也降方差 ⇒ 分辨"option 层本身" vs "β_o 是必要条件"
#   H3(否决):  天花板检查(作用域 = e9 + H2 的两臂)不过 ⇒ H2 判无效
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU=${GPU:-0}
LOG=results/oak_longv5.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== L5 批启动 (方差复现; 处理变量 = 流修复, flag 与 L4 相同) =========="

# ── 启动前硬闸门 ────────────────────────────────────────────────
# ★ 本批的处理变量 = **代码修复, 不是 flag**。
#   修 `run_lm4_wave.py` 之前, `--stream` 对 `--proposer oak` 是死参数,
#   实际流 = "前 n_domains 轮各域一次 + 其余全在最后一个域"(实测 295/300)。
#   修复后同一个 `--stream revisit` 才真正产生"每轮随机抽域"的流。
#   所以闸门必须同时证明**两件事**:
#     ① 配置串与 L4 **逐字相同**(flag 一个没变)
#     ② 修复在位(主循环按 `dom_seq[step]` 取域)
EXPECT=$(grep -A3 '^L4="' tests/oak_longv4.sh | head -4 | sed 's/^L4=/CFG=/')
GOT=$(grep -A3 '^L5="' tests/oak_longv5.sh | head -4 | sed 's/^L5=/CFG=/')
if [ "$EXPECT" != "$GOT" ]; then
  say "✗ 致命: L5 配置串与 L4 不一致 —— 处理变量应当是**代码修复**, flag 必须一个不变"
  echo "--- L4 ---" >>"$LOG"; echo "$EXPECT" >>"$LOG"
  echo "--- L5 ---" >>"$LOG"; echo "$GOT" >>"$LOG"
  exit 1
fi
say "✓ 闸门①: L5 配置串与 L4 逐字相同(处理变量是代码修复, 不是 flag)"

if ! grep -q 'dd = dom_seq\[step\]' tests/run_lm4_wave.py; then
  say "✗ 致命: 流修复不在位(主循环未按 dom_seq[step] 取域) —— 本批会退化成 L4 的重复, 拒绝启动"
  exit 1
fi
if ! grep -q 'STREAM_HIST' tests/run_lm4_wave.py; then
  say "✗ 致命: 缺 stream 直方图回读闸门 —— 无法证明流生效, 拒绝启动"
  exit 1
fi
say "✓ 闸门②: 流修复在位 + 直方图回读闸门在位"

# L4 的处理变量(β_o 区域判据)必须在位, 否则本批不是 L4 的复现
if ! grep -q "def in_goal_region" hibs_lnn/option_manager.py \
   || ! grep -q "if self.in_goal_region(s):" hibs_lnn/option_manager.py; then
  say "✗ 致命: β_o 区域判据不在位 —— 与 L4 不可比, 拒绝启动"
  exit 1
fi
if ! grep -q "observe_action" hibs_lnn/oak_proposer.py \
   || ! grep -q "start_blocked" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: oak_proposer 接线缺失(coverage / start_blocked)"
  exit 1
fi
say "✓ 闸门: β_o 区域判据在位 + observe_action 接线"

# seed 必须与 L3/L4 完全不相交
for S in 42 1 7 13 100; do
  ls -d results/oakL3_*_s$S >/dev/null 2>&1 && say "  (L3 已用 seed $S)"
done
say "✓ L3 已用 seed: 42 1 7 13 100 | L4 已用: 2…27 | 本批(28…45) 应无交集"

N_EXIST=$(ls -d results/oakL4_* 2>/dev/null | wc -l)
say "✓ L4 已有臂数 = $N_EXIST (期望 96)"

# ── 显存门槛(启动时刻) ────────────────────────────────────────
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
if [ -z "$FREE" ] || [ "$FREE" -lt 8000 ]; then
  say "✗ 致命: GPU$GPU 空闲显存 ${FREE:-?}MiB < 8000MiB, 拒绝启动"
  exit 1
fi
say "✓ GPU$GPU 空闲显存 ${FREE}MiB"

# ── 配置: L4 **逐字相同**(flag 一个不变) ────────────────────────
L5="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda"

# 新 seed: 28…41 + 43…46 = 18 个。
# **刻意跳过 42**(L3 用过), 也与 L4 的 2…27 无交集。
SEEDS="28 29 30 31 32 33 34 35 36 37 38 39 40 41 43 44 45 46"

say "L5 起跑 (GPU$GPU): 18 个新 seed × 4 臂 = 72 run, rounds=300, stream=nonstationary"
say "新 seed: $SEEDS"
(
  for S in $SEEDS; do
    # ★ 运行期巡检: 启动时的显存闸门管不住全程(L4 实测: 开跑 21GB → 结束 1.4GB)
    RUNTIME_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    if [ -n "$RUNTIME_FREE" ] && [ "$RUNTIME_FREE" -lt 3000 ]; then
      say "⚠⚠ 运行期告警: GPU$GPU 空闲显存仅 ${RUNTIME_FREE}MiB (<3000) —— 可能被邻居挤占, 继续但记录"
    fi

    say "L5 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L5 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oakL5_e9_s$S >>"$LOG" 2>&1
    say "L5 E9 seed=$S rc=$?"

    for M in goal goal_term goal_term_override; do
      say "L5 $M seed=$S"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L5 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oakL5_${M}_s$S >>"$LOG" 2>&1
      say "L5 $M seed=$S rc=$?"
    done
  done
  touch results/OAKL5_DONE
  say "L5 全部 72 run 完成"
) >>"$LOG" 2>&1 &

echo $! > results/oakL5.pid
disown
sleep 8
say "L5 已起跑 pid=$(cat results/oakL5.pid) (72 run x ~10.6min ≈ 12.7h)"
