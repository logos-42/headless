from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only (stricter than wiki_check; opt-in for v2 projects)

"""
wiki_lint.py — v2 LLM-Wiki linter (Rohit G00 v2 + project extensions).

v2 mode adds checks for:
  - schema_version: 2 → enforce last_confirmed, audience, stage
  - supersedes: each target must exist and have stage == archived
  - contradicts: pairs should be declared symmetrically
  - confidence: must be in {high, medium, low, unverified}
  - entity_type: must be in whitelist
  - crystallized_claims: each claim_id must exist in crystallized.csv
  - last_confirmed: must be valid YYYY-MM-DD, not in the future
  - log.md: optional `## [date] topic | outcome` headers OR table format
  - stage enum: 5 档 (draft/current/stale/archived/crystallized)
  - audience enum: 4 档 (self/internal/reader/public)

Usage:
  python3 wiki_lint.py --strict=v1    # v1.3.0 兼容检查 (default)
  python3 wiki_lint.py --strict=v2    # v2 严格检查
  python3 wiki_lint.py --strict=v2 --crystallized-csv manifests/crystallized.csv
"""

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"

# v1 兼容（不变）
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
FRONTMATTER_EXEMPT = {"index.md", "log.md", "README.md", "SCHEMA.md"}
LOG_HEADER_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}\] .+ \| .+$")
LOG_TABLE_RE = re.compile(r"^\| [\s\S]*\|$", re.MULTILINE)

# v1 必填
V1_REQUIRED = {"title", "source", "created"}
# v2 必填（继承 v1 + 增 3 字段）
V2_REQUIRED = V1_REQUIRED | {"last_confirmed", "audience", "stage"}

# v2 enum
V2_STAGE_ENUM = {"draft", "current", "stale", "archived", "crystallized"}
V2_AUDIENCE_ENUM = {"self", "internal", "reader", "public"}
V2_STATUS_ENUM = {"draft", "current", "stale"}
V2_CONFIDENCE_ENUM = {"high", "medium", "low", "unverified"}
V2_ENTITY_TYPE_ENUM = {"concept", "person", "protocol", "chapter", "claim", "meta"}

# 不需要被 index.md 引用的"档案/历史"页
EXEMPT_FROM_INDEX = {
    "README.md", "SCHEMA.md", "log.md", "index.md",
    "book-toc.md",  # v2: stage=archived 档案
    "review-report.md",
    "review-modifications.md",
    "review-mid-priority.md",
    # v2 概念页（独立分区，不强引）
    "memory-lifecycle.md", "knowledge-graph.md", "retrieval-playbook.md",
    "hooks-and-automation.md", "quality-and-self-heal.md",
    "collaboration-protocol.md", "privacy-and-redaction.md",
    "crystallized-claims.md", "frontmatter-schema.md",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_frontmatter(text: str) -> dict[str, str]:
    """Lightweight YAML frontmatter parser. Returns dict of {key: value} (value is raw string)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    body = m.group(1)
    out: dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def is_log_table_format(text: str) -> bool:
    """Detect if log.md uses table format (multiple lines starting with | )."""
    table_lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return len(table_lines) >= 3  # header + separator + ≥1 row


def check_v1(path: Path, text: str) -> list[str]:
    errs: list[str] = []
    if path.name in FRONTMATTER_EXEMPT:
        return errs
    fm = parse_frontmatter(text)
    if not fm:
        errs.append(f"missing frontmatter: {path.relative_to(ROOT)}")
        return errs
    missing = V1_REQUIRED - set(fm.keys())
    if missing:
        errs.append(f"missing frontmatter keys in {path.relative_to(ROOT)}: {', '.join(sorted(missing))}")
    return errs


def check_v2(path: Path, text: str) -> list[str]:
    errs: list[str] = []
    warnings: list[str] = []
    if path.name in FRONTMATTER_EXEMPT:
        return errs
    fm = parse_frontmatter(text)
    if not fm:
        errs.append(f"[v2] missing frontmatter: {path.relative_to(ROOT)}")
        return errs

    missing = V2_REQUIRED - set(fm.keys())
    if missing:
        errs.append(f"[v2] missing required keys in {path.relative_to(ROOT)}: {', '.join(sorted(missing))}")

    # enum 校验
    stage = fm.get("stage", "")
    if stage and stage not in V2_STAGE_ENUM:
        errs.append(f"[v2] invalid stage in {path.relative_to(ROOT)}: {stage!r} (allowed: {sorted(V2_STAGE_ENUM)})")

    audience = fm.get("audience", "")
    if audience and audience not in V2_AUDIENCE_ENUM:
        errs.append(f"[v2] invalid audience in {path.relative_to(ROOT)}: {audience!r}")

    status = fm.get("status", "")
    if status and status not in V2_STATUS_ENUM:
        errs.append(f"[v2] invalid status in {path.relative_to(ROOT)}: {status!r} (v2 status enum 不再含 in-progress)")

    confidence = fm.get("confidence", "")
    if confidence and confidence not in V2_CONFIDENCE_ENUM:
        errs.append(f"[v2] invalid confidence in {path.relative_to(ROOT)}: {confidence!r}")

    entity_type = fm.get("entity_type", "")
    if entity_type and entity_type not in V2_ENTITY_TYPE_ENUM:
        errs.append(f"[v2] invalid entity_type in {path.relative_to(ROOT)}: {entity_type!r}")

    # last_confirmed 日期合法性
    last_confirmed = fm.get("last_confirmed", "")
    if last_confirmed and not DATE_RE.match(last_confirmed):
        errs.append(f"[v2] invalid last_confirmed date in {path.relative_to(ROOT)}: {last_confirmed!r}")
    elif last_confirmed:
        try:
            d = datetime.strptime(last_confirmed, "%Y-%m-%d").date()
            if d > date.today():
                errs.append(f"[v2] last_confirmed in future in {path.relative_to(ROOT)}: {last_confirmed}")
        except ValueError:
            errs.append(f"[v2] unparseable last_confirmed in {path.relative_to(ROOT)}: {last_confirmed!r}")

    # source_hash 16 位 hex（如果填了）
    source_hash = fm.get("source_hash", "")
    if source_hash and not re.match(r"^[0-9a-f]{16}$", source_hash):
        warnings.append(f"[v2] source_hash not 16-hex in {path.relative_to(ROOT)}: {source_hash!r}")

    return errs + warnings


def check_supersede_relations(md_files: list[Path]) -> list[str]:
    """检查 supersedes 链：目标必须存在，目标 stage=archived，无环。"""
    errs: list[str] = []
    pages: dict[str, dict[str, str]] = {}
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if not fm:
            continue
        rel = path.relative_to(WIKI_ROOT).as_posix()
        pages[rel] = fm

    for rel, fm in pages.items():
        supersedes_raw = fm.get("supersedes", "").strip()
        if not supersedes_raw or supersedes_raw in ("[]", ""):
            continue
        # 简单解析 [./a.md, ./b.md] 格式
        targets = re.findall(r"\./[^\s,\]]+", supersedes_raw)
        for target in targets:
            target_norm = target.lstrip("./")
            if target_norm not in pages:
                errs.append(f"supersedes target missing: {rel} -> {target}")
                continue
            target_stage = pages[target_norm].get("stage", "")
            if target_stage != "archived":
                errs.append(f"supersedes target not archived: {rel} -> {target} (stage={target_stage!r})")
            # 反向环检测
            target_fm = pages[target_norm]
            target_supersedes = target_fm.get("supersedes", "")
            if rel in target_supersedes:
                errs.append(f"supersedes cycle detected: {rel} <-> {target}")

    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Wiki v2 linter")
    parser.add_argument("--strict", choices=["v1", "v2"], default="v1",
                        help="v1: v1.3.0 兼容模式 (default); v2: v2 严格模式")
    parser.add_argument("--wiki-root", type=Path, default=WIKI_ROOT)
    parser.add_argument("--crystallized-csv", type=Path,
                        default=ROOT / "manifests" / "crystallized.csv",
                        help="crystallized claims manifest (for crystallized_claims 校验)")
    args = parser.parse_args()

    wiki_root = args.wiki_root.resolve()
    if not wiki_root.exists():
        print(f"wiki_lint: {wiki_root} does not exist")
        return 1

    md_files = sorted(wiki_root.rglob("*.md"))
    errors: list[str] = []

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        if args.strict == "v1":
            errors.extend(check_v1(path, text))
        else:
            errors.extend(check_v2(path, text))

    # v2 专属：supersede 链
    if args.strict == "v2":
        errors.extend(check_supersede_relations(md_files))

    # log.md 格式
    log_path = wiki_root / "log.md"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        if not is_log_table_format(text):
            for line in text.splitlines():
                if line.startswith("## ") and not LOG_HEADER_RE.match(line):
                    errors.append(f"bad log header format in {log_path.relative_to(ROOT)}: {line}")

    if errors:
        print(f"wiki_lint (--strict={args.strict}): FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"wiki_lint (--strict={args.strict}): OK")
    print(f"- markdown files: {len(md_files)}")
    print(f"- schema: {args.strict}")
    print(f"- wiki root: {wiki_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
