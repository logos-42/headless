---
title: GitHub 与 Raw 仓分工策略
source: session
created: 2026-09-06
last_confirmed: 2026-09-06
audience: reader
stage: draft
tags: [strategy, git]
status: current
---

## 结论

- GitHub private repo 放：`code + wiki + manifests + verified_cases`
- 本地 raw 仓放：`pdf/xlsx/xls/rar/图片/客户原件`
- memory repo 放：编译后的长期记忆，不放 raw 本体

## 为什么

把全量 raw 塞进 Git，只会让仓库越来越肥，diff 也基本没用。

真正该版本化的是：

- 结论
- 规则
- 答案表
- 清晰的索引

不是一堆二进制原件。