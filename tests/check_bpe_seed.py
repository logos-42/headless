#!/usr/bin/env python3
"""查: 各 seed 的 BPE 分词器是否相同 (若不同, 跨 seed 方差就不是"随机性")。"""
import glob
import hashlib
import json
import os

os.chdir("/work/liuyuanjie/headless/data/text")
fs = sorted(glob.glob("bpe_lm5_*.json"))
print("=== 各 seed 的 BPE 缓存 ===")
sigs = {}
for f in fs:
    d = json.load(open(f))
    sig = hashlib.md5(json.dumps(d["merges"], sort_keys=True).encode()).hexdigest()
    sigs[f] = sig
    print("  %-20s vocab=%-6d merges=%-5d md5=%s" %
          (f, d["vocab"], len(d["merges"]), sig[:16]))
uniq = set(sigs.values())
print()
print("  不同分词器个数:", len(uniq))
print("  ->", "各 seed 用的是**同一个**分词器 (seed 只影响模型随机性)" if len(uniq) == 1
      else "**各 seed 用了不同分词器 —— 跨 seed 方差被污染**")

# 编码是否相同
efs = sorted(glob.glob("enc_lm5_*.json"))
print()
print("=== 各 seed 的编码缓存 ===")
for f in efs:
    h = hashlib.md5(open(f, "rb").read()).hexdigest()[:16]
    print("  %-20s %s" % (f, h))
