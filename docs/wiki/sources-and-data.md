---
title: 资料与数据
source: session
created: 2026-09-06
last_confirmed: 2026-09-06
audience: reader
stage: draft
tags: [data, raw]
status: current
---

原始资料默认放在本地 raw 根目录，不直接进 Git。

raw 根目录建议：

```text
../sovereign_ai_raw/
```

GitHub 里只保留 manifest 和编译结果。

少量 raw 可以手工登记；新文件一多，直接跑：

```bash
python3 scripts/ingest_raw.py
python3 scripts/stale_report.py
python3 scripts/delta_compile.py --write-drafts
```