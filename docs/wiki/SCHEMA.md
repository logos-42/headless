# Wiki Schema v2 (LLM Wiki v2.0)

v2 schema 是**双轨**的：一个页面可以用 `schema_version: 1`（v1 兼容）或 `schema_version: 2`（v2 严格）声明遵循哪种 schema。`wiki_lint.py --strict=v1` / `--strict=v2` 选择对应的校验器。这意味着已有的 v1.3.0 项目可以继续工作，而新的 v2 页面会获得更强的校验。

## Schema version selector

```yaml
---
schema_version: 2   # 1 (v1 兼容) | 2 (v2 严格). Default: 1
---
```

如果缺省，页面按 v1 处理。这是唯一的 meta 字段，下面的都是按版本分类的。

---

## v1 兼容模式 (schema_version: 1)

继承自 v1.3.0，零迁移成本，新项目仍可用。

### 必填字段

- `title` — 页面标题
- `source` — 编译来源（raw 路径 / URL / "session"）
- `created` — 创建日期 (YYYY-MM-DD)

### 可选字段

- `updated` — 最后更新日期 (YYYY-MM-DD)
- `tags` — 分类标签
- `status` — `current` (默认) / `draft` / `stale`
- `source_hash` — 16 位 SHA-256 前缀
- `compiled_at` — ISO-8601 UTC
- `compiled_from` — list\[source_id\]

---

## v2 严格模式 (schema_version: 2)

继承 v1 全部必填 + 新增 v2 字段。所有 v2 字段都可选填，但 **运行 wiki_lint --strict=v2 时**部分字段会成为强制项。

### v2 必填字段（继承 v1 + 增 3 字段）

- `title` / `source` / `created` / `last_confirmed` / `audience` / `stage`

### v2 可选字段

- v1 全部可选字段
- `confidence` — `high` / `medium` / `low` / `unverified` (默认 `unverified`)
- `entity_type` — `concept` / `person` / `protocol` / `chapter` / `claim` / `meta` (默认 `meta`)
- `supersedes` — list\[page_path\]，旧页面被取代的链
- `contradicts` — list\[page_path\]，与本页面观点冲突的页
- `owner` — string (默认 `me`)
- `crystallized_claims` — list\[claim_id\]
- `source_hash` — 16 位 hex
- `compiled_at` — ISO-8601
- `compiled_from` — list\[source_id\]

### v2 enum 完整约定

| 字段 | 合法值 |
|---|---|
| `stage` | `draft` / `current` / `stale` / `archived` / `crystallized` |
| `audience` | `self` / `internal` / `reader` / `public` |
| `status` | `draft` / `current` / `stale` |
| `confidence` | `high` / `medium` / `low` / `unverified` |
| `entity_type` | `concept` / `person` / `protocol` / `chapter` / `claim` / `meta` |
| `owner` | 自由字符串 (默认 `me`) |

**stage 详细语义**：
- `draft` — 草稿，wiki_lint 会提醒
- `current` — 当前共识，可被引用
- `stale` — 已过期，需重编译或被 supersede
- `archived` — 已被新版本替代，不应被引用
- `crystallized` — 由 `crystallize.py` 生成的稳定断言，可对外引用

---

## 关系与版本链

### supersedes 链

```yaml
# 在 v2 风格指南里写
---
supersedes: [./v1-style-guide.md]
---

# 在 v1 风格指南里写
---
stage: archived
status: stale
---
```

`supersede_check.py` 会验证：
1. 目标文件存在
2. 目标文件 `stage == archived`
3. 没有环 (A supersedes B, B supersedes A → 报错)
4. 报告"被替代但未归档"的孤儿

### contradicts 对

```yaml
# 页面 A
contradicts: [./chapter-1.md]

# 页面 B (在 B 自己的 frontmatter 里也写)
contradicts: [./chapter-2.md]
```

`wiki_lint.py` 严格模式会要求成对声明。

---

## 双 schema 切换示例

```yaml
# 旧 v1 页面保持不变
---
title: 旧风格指南
source: session
created: 2026-05-01
status: current
tags: [style]
---

# 同一文件升 v2
---
title: 风格指南
source: session
created: 2026-05-01
last_confirmed: 2026-06-08
schema_version: 2
audience: self
stage: current
status: current
confidence: high
tags: [style, v2]
---
```

---

## 页面 (15 个标准页)

### 必出页（v1 沿用）
- `index.md` — 免 frontmatter，纯索引
- `log.md` — 免 frontmatter，日志（兼容表格或 `## [date]` 头）
- `project-overview.md` — 项目总体说明
- `current-status.md` — 写作/项目状态
- `sources-and-data.md` — 原始资料清单
- `style-guide.md` — 写作风格规约
- `github-and-raw-strategy.md` — GitHub / raw 仓分工

### v2 新增概念页
- `frontmatter-schema.md` — schema 自描述 (指向 SCHEMA.md)
- `memory-lifecycle.md` — 4 tier 记忆模型
- `knowledge-graph.md` — 节点与类型化关系
- `retrieval-playbook.md` — 混合检索用法
- `hooks-and-automation.md` — 钩子触发器
- `quality-and-self-heal.md` — 质量自愈
- `collaboration-protocol.md` — 主编权 / 合并
- `privacy-and-redaction.md` — 摄入过滤 / 隐私
- `crystallized-claims.md` — 晶化条目

---

## 规则

1. **新 raw 进来，先登记 manifest** (`ingest_raw.py`)
2. **新结论，必须回写 wiki**（带 frontmatter）
3. **新规则，必须同时补测试或脚本**
4. **没证据，不要写成结论**
5. **memory repo 只放编译结果，不放 raw 本体**
6. **supersede 旧页面时**，旧页面 `stage: archived` + `status: stale` + 在新页面 `supersedes: [...]` 写入
7. **矛盾页面成对声明** `contradicts`
8. **`last_confirmed` 任何改动后必须更新**

---

## 为什么这样设计

- **AI 读一个页面就知道信息从哪来**：source / compiled_from / source_hash
- **机器可以校验 schema 完整性**：v2 严格模式让规范有强制力
- **版本链可追溯**：supersedes 链让"哪个页面被什么替代"一目了然
- **矛盾自裁决**：contradicts 让多页面观点冲突有显式记录
- **零额外 token 开销**：frontmatter 是页面自身的一部分
- **支持渐进升级**：v1 项目 → 加 `schema_version: 2` 即可启用严格检查
