---
title: github-and-raw-strategy
source_hash: abcdef1234567890
source: headless
created: 2026-09-06
last_confirmed: 2026-09-06
audience: public
stage: draft
---

# github-and-raw-strategy

## raw 文件存储策略
- 原始数据存入 `headless_raw/` 目录
- 通过 `intake_filter.py` 进行隐私脱敏
- 通过 `ingest_raw.py` 注册到 manifest

## Git LFS 建议
- model checkpoint: *.pt
- large CSV: data/*.csv
