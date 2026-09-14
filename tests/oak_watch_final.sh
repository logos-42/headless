#!/bin/bash
# 最终完成触发: 等所有队列 (S3 / S4 / ext0 / ext5) 全部 done, 跑**最终完整**分析。
# 只读式等待, 不轮询打扰用户; 结果落盘。
cd /work/liuyuanjie/headless
for i in $(seq 1 2400); do          # 最多等 20 小时
  if [ -f results/OAK_S3_DONE ] && [ -f results/OAK5_S4_DONE ] \
     && [ -f results/OAK_EXT0_DONE ] && [ -f results/OAK5_EXT_DONE ]; then
    break
  fi
  sleep 30
done
PY=/work/liuyuanjie/envs/vllm-cu128/bin/python
$PY -u tests/analyze_oak.py results > results/oak_analysis_final.txt 2>&1
echo "=== FINAL_ANALYSIS_DONE $(date -u +%H:%M:%S) ===" >> results/oak_analysis_final.txt
echo "=== 各队列 DONE 标记 ===" >> results/oak_analysis_final.txt
ls results/OAK*DONE results/OAK5*DONE 2>/dev/null >> results/oak_analysis_final.txt
