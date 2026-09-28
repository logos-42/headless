#!/bin/bash
# oak_longv4.sh — **L4 批**: L3 的 seed 扩展(只加 seed, 配置逐字不变)
#
# ## 为什么是这批
#
# L3(25 臂 × 5 seed)得到: `goal_reached` 1 -> 95(机制闸门过), 但主指标
# `replay acc` 上三个闭环 option 档与 `e9` 的差 **全部小于本批分辨极限**:
#
#   档                    |Δ|      本批 MDE(n=5)   要 p<0.05 需每组 n
#   goal_term            0.0060      0.0317              72
#   goal_term_override   0.0156      0.0345              10
#   goal                 0.0267      0.0287               4
#   fixed                0.0525      0.0453               3   (已判定: 显著更差)
#
# ⇒ L3 对 O2/O3/O4 是「没有能力回答」, 不是「回答是没有差异」。
#   本批的唯一目的是把这个分辨力补上。
#
# ## 本批**只改一个变量**
#   seed 数 5 -> 29。`L4` 配置串与 L3 **逐字相同**; 臂集去掉 `fixed`(已判定)
#   与 `naive`(跨 seed 极差 0.515, 无锚定价值), 保留 `e9` 以便配对。
#
# ## 规模(按实测耗时: e9 5.9 min / option 臂 7.5 min)
#   24 个新 seed × (e9 + goal + goal_term + goal_term_override)
#   = 24 × (5.9 + 3×7.5) = 24 × 28.4 = 682 min ≈ 11h22m
#
# ## 判定(事前写死, 三值: 显著更差 / 不显著 / **inconclusive**)
#   ① 机制闸门(analyze_k1v2.py): 全部臂必须过, 否则对照无效
#   ② ★ 对每个 option 档 vs `e9` 报 |Δ| / MDE / 所需 n, 并**显式输出 inconclusive**
#      当 |Δ| < MDE 时。**不许把"样本量不足"写成"无差异", 也不许写成"未排除"**
#   ③ ★ 主口径 = **配对**(同 seed, 先验默认); 非配对作为敏感性一并报。
#      两口径不一致时必须明写不一致, 不许挑有利的那个
#   ④ 若 O3/O4 的 |Δ| 上界仍 < 0.01 ⇒ 结论是「闭环 option 对最终 acc 影响 < 1%」,
#      这是**有信息量的结论**, 不是失败
#
# ## 本批明确**不做**
#   · 不改 `fixed`/`naive`  · 不改配置  · 不碰 LM4 主链
#   · 不填 `efficiency` 字段(那是 runner 的改动, 属另一变量)
#   · 效率指标从 `any_time_curve` / `acc_matrix` **事后**算(新旧臂同一份代码)
set -u
cd /work/liuyuanjie/headless
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
GPU=${GPU:-0}
LOG=results/oak_longv4.log
: > "$LOG"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "========== L4 批启动 (L3 的 seed 扩展, 配置逐字不变) =========="

# ── 启动前硬闸门(不通过就拒绝启动) ──────────────────────────────
# ① 本批的处理变量 = 无。所以先证明**上一批的处理变量仍在位** ——
#    否则本批跑出来不是 L3 的扩展, 而是另一种配置, 与 L3 不可比。
if ! grep -q "def in_goal_region" hibs_lnn/option_manager.py; then
  say "✗ 致命: option_manager.py 无 in_goal_region —— L3 的处理变量丢失, 与 L3 不可比, 拒绝启动"
  exit 1
fi
if ! grep -q "if self.in_goal_region(s):" hibs_lnn/option_manager.py; then
  say "✗ 致命: terminated() 未接上区域判据 —— 与 L3 不可比, 拒绝启动"
  exit 1
fi
if ! grep -q "observe_action" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: oak_proposer.py 无 observe_action —— coverage 未接线, 整轮空转, 拒绝启动"
  exit 1
fi
if ! grep -q "start_blocked" hibs_lnn/oak_proposer.py; then
  say "✗ 致命: 缺 start_blocked 计数器, 无法判定静默空转"
  exit 1
fi
# ② 配置串必须与 L3 **逐字相同** —— 从 L3 脚本里读出来比对, 不凭记忆
if ! diff <(grep -A3 '^L4="' tests/oak_longv3.sh | head -4) \
          <(grep -A3 '^L4="' tests/oak_longv4.sh  | head -4) >/dev/null 2>&1; then
  say "✗ 致命: 本脚本的 L4 配置串与 oak_longv3.sh 不一致 —— 只改一个变量的纪律被破坏, 拒绝启动"
  exit 1
fi
say "✓ 前置检查通过: 区域判据在位 + observe_action 接线 + 配置串与 L3 逐字一致"

# ③ 已有臂数(证明这是扩展, 不是重跑)
N_EXIST=$(ls -d results/oakL3_* 2>/dev/null | wc -l)
say "✓ L3 已有臂数 = $N_EXIST (期望 25)"

# ── 显存门槛(不许和邻居抢满卡) ────────────────────────────────
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
if [ -z "$FREE" ] || [ "$FREE" -lt 8000 ]; then
  say "✗ 致命: GPU$GPU 空闲显存 ${FREE:-?}MiB < 8000MiB, 拒绝启动"
  exit 1
fi
say "✓ GPU$GPU 空闲显存 ${FREE}MiB"

# ── 配置: 与 L3 逐字相同 ────────────────────────────────────────
L4="--backbone mlp --norm fixed --cl-method replay --head linear \
--epochs-per-domain 100 --joint-steps 0 --wfr-bands 13 --lshell \
--pool cat --agg-path --domains 6 --fine-bins 18 --rounds 300 \
--proposer-k 3 --stream revisit --device cuda"

# L3 已用 seed: 42 1 7 13 100 —— 新 seed 必须完全不相交
SEEDS="2 3 4 5 6 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27"

say "L4 起跑 (GPU$GPU): 24 个新 seed × 4 臂 = 96 run, rounds=300"
say "新 seed: $SEEDS"
(
  for S in $SEEDS; do
    say "L4 E9 seed=$S"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
      --proposer oak --oak-options 0 --oak-gate 1 --oak-refresh 4 \
      --seed $S --out results/oakL4_e9_s$S >>"$LOG" 2>&1
    say "L4 E9 seed=$S rc=$?"

    for M in goal goal_term goal_term_override; do
      say "L4 $M seed=$S"
      CUDA_VISIBLE_DEVICES=$GPU $PY -u tests/run_lm4_wave.py $L4 \
        --proposer oak --oak-options 1 --oak-gate 1 --oak-refresh 4 \
        --opt-mode $M --opt-frac 1.0 \
        --seed $S --out results/oakL4_${M}_s$S >>"$LOG" 2>&1
      say "L4 $M seed=$S rc=$?"
    done
  done
  touch results/OAKL4_DONE
  say "L4 全部 96 run 完成"
) >>"$LOG" 2>&1 &

echo $! > results/oakL4.pid
disown
sleep 8
say "L4 已起跑 pid=$(cat results/oakL4.pid) (96 run x ~7min ≈ 11h22m)"
