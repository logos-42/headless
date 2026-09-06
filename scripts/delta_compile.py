from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only (needs PROJECT_RAW_ROOT)

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"
MANIFEST = ROOT / "manifests" / "raw_sources.csv"
LOCK_FILE = ROOT / "manifests" / "raw_index.json"
REPORT_FILE = ROOT / "manifests" / "delta_compile_report.md"
DRAFT_DIR = WIKI_ROOT / "drafts"
DEFAULT_RAW_ROOT = (ROOT.parent / "__RAW_ROOT_NAME__").resolve()
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
SKIP_FILES = {"index.md", "log.md", "README.md", "SCHEMA.md"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "draft"


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def parse_list_field(value: str) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [item.strip().strip("'").strip('"') for item in raw.split(",") if item.strip()]


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def load_lock() -> dict[str, dict]:
    if not LOCK_FILE.exists():
        return {}
    try:
        data = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        files = data.get("files", {})
        return files if isinstance(files, dict) else {}
    except Exception:
        return {}


def resolve_row(source_value: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if source_value == row.get("source_id") or source_value == row.get("raw_rel_path"):
            return row
    for row in rows:
        if row.get("source_id") and row["source_id"] in source_value:
            return row
        if row.get("raw_rel_path") and source_value.endswith(row["raw_rel_path"]):
            return row
    return None


def choose_target_page(row: dict[str, str]) -> str:
    compiled_into = (row.get("compiled_into") or "").strip()
    if compiled_into:
        first = compiled_into.split(",")[0].strip()
        if first:
            return first
    stem = slugify(Path(row.get("filename") or row.get("raw_rel_path") or row.get("source_id") or "source").stem)
    return f"docs/wiki/{stem}.md"


def draft_path(target_page: str, source_id: str) -> Path:
    stem = slugify(Path(target_page).stem)
    return DRAFT_DIR / f"{stem}--{source_id}.md"


def unique_items(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def render_draft(
    *,
    title: str,
    source_id: str,
    source_hash: str,
    target_page: str,
    raw_rel_path: str,
    source_summary: str,
    change_summary: list[str],
    compiled_from: list[str],
    reason: str,
) -> str:
    compiled_from_values = unique_items(compiled_from or [source_id])
    compiled_from_line = f"[{', '.join(compiled_from_values)}]"
    lines = [
        "---",
        f"title: {title}",
        f"source: {source_id}",
        f"source_hash: {source_hash}",
        f"compiled_at: {utc_now()}",
        f"compiled_from: {compiled_from_line}",
        f"created: {today()}",
        "tags: [draft, delta-compile]",
        "status: draft",
        "---",
        "",
        f"# {title}",
        "",
        "## Why this draft exists",
        "",
        f"- reason: {reason}",
        f"- suggested target page: `{target_page}`",
        f"- raw source: `{raw_rel_path}`",
        "",
        "## Source Summary",
        "",
        f"- {source_summary or 'no summary available'}",
        "",
        "## Structured Change Summary",
        "",
    ]
    if change_summary:
        lines.extend(f"- {item}" for item in change_summary)
    else:
        lines.append("- no prior diff summary available")
    lines.extend([
        "",
        "## Draft Notes",
        "",
        "- Pull confirmed facts from the raw source into the target page.",
        "- Update the target page frontmatter with the current `source_hash` and `compiled_at`.",
        "- Keep this draft until the recompile is merged, then delete it.",
        "",
    ])
    return "\n".join(lines)


def build_report(stale_items: list[dict[str, object]], new_items: list[dict[str, object]], written_drafts: list[str]) -> str:
    lines = [
        "# Delta Compile Report",
        "",
        f"- generated_at: `{utc_now()}`",
        f"- stale_pages: `{len(stale_items)}`",
        f"- new_raw_sources: `{len(new_items)}`",
        f"- drafts_written: `{len(written_drafts)}`",
        "",
        "> This report suggests recompilation work. It does not auto-overwrite wiki content.",
        "",
        "## Stale Pages",
        "",
    ]
    if stale_items:
        for item in stale_items:
            lines.append(
                f"- `{item['page_rel']}` <- `{item['raw_rel_path']}` "
                f"(target: `{item['target_page']}`)"
            )
            for change in item.get("change_summary", [])[:3]:
                lines.append(f"  - {change}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## New Raw Sources",
        "",
    ])
    if new_items:
        for item in new_items:
            lines.append(
                f"- `{item['raw_rel_path']}` -> suggested page `{item['target_page']}`"
            )
            lines.append(f"  - {item['source_summary']}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Draft Files",
        "",
    ])
    if written_drafts:
        lines.extend(f"- `{path}`" for path in written_drafts)
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manual delta-compile suggestions and optional wiki draft stubs.")
    parser.add_argument("--raw-root", default=os.environ.get("PROJECT_RAW_ROOT", ""), help="Local raw root path")
    parser.add_argument("--report-file", default=str(REPORT_FILE), help="Markdown report output path")
    parser.add_argument("--write-drafts", action="store_true", help="Write draft markdown stubs into docs/wiki/drafts/")
    parser.add_argument("--dry-run", action="store_true", help="Print the report without writing files")
    args = parser.parse_args()

    raw_root = None
    if args.raw_root:
        candidate = Path(args.raw_root).expanduser().resolve()
        if candidate.exists():
            raw_root = candidate
    elif DEFAULT_RAW_ROOT.exists():
        raw_root = DEFAULT_RAW_ROOT

    rows = load_manifest()
    lock = load_lock()
    referenced_ids: set[str] = set()
    referenced_paths: set[str] = set()
    stale_items: list[dict[str, object]] = []
    new_items: list[dict[str, object]] = []
    written_drafts: list[str] = []

    for path in sorted(WIKI_ROOT.rglob("*.md")):
        if path.name in SKIP_FILES or "drafts" in path.parts:
            continue
        fm = parse_frontmatter(path)
        if not fm:
            continue
        source = fm.get("source", "")
        if source == "session":
            continue
        row = resolve_row(source, rows)
        if not row:
            continue
        referenced_ids.add(row.get("source_id", ""))
        referenced_paths.add(row.get("raw_rel_path", ""))
        for extra in parse_list_field(fm.get("compiled_from", "")):
            extra_row = resolve_row(extra, rows)
            if extra_row:
                referenced_ids.add(extra_row.get("source_id", ""))
                referenced_paths.add(extra_row.get("raw_rel_path", ""))
        source_hash = fm.get("source_hash", "")
        current_hash = ""
        if raw_root:
            source_path = raw_root / row.get("raw_rel_path", "")
            if source_path.exists():
                current_hash = sha256_prefix(source_path)
        if not current_hash and row.get("raw_rel_path") in lock:
            current_hash = lock[row["raw_rel_path"]].get("content_hash", "")
        if not source_hash or not current_hash or source_hash == current_hash:
            continue
        lock_entry = lock.get(row.get("raw_rel_path", ""), {})
        target_page = path.relative_to(ROOT).as_posix()
        compiled_from = unique_items([row["source_id"]] + parse_list_field(fm.get("compiled_from", "")))
        stale_items.append({
            "page_rel": target_page,
            "target_page": target_page,
            "raw_rel_path": row.get("raw_rel_path", ""),
            "source_id": row.get("source_id", ""),
            "source_hash": current_hash,
            "source_summary": lock_entry.get("summary", ""),
            "change_summary": list(lock_entry.get("change_summary", [])),
            "compiled_from": compiled_from,
            "reason": f"source hash changed ({source_hash} -> {current_hash})",
        })

    for row in rows:
        raw_rel_path = row.get("raw_rel_path", "")
        source_id = row.get("source_id", "")
        if row.get("status") != "new" or not raw_rel_path:
            continue
        if source_id in referenced_ids or raw_rel_path in referenced_paths:
            continue
        lock_entry = lock.get(raw_rel_path, {})
        new_items.append({
            "raw_rel_path": raw_rel_path,
            "source_id": source_id,
            "source_hash": lock_entry.get("content_hash", ""),
            "source_summary": lock_entry.get("summary", ""),
            "change_summary": list(lock_entry.get("change_summary", [])),
            "compiled_from": [source_id],
            "target_page": choose_target_page(row),
            "reason": "new raw source has not been compiled into wiki yet",
        })

    if args.write_drafts and not args.dry_run:
        DRAFT_DIR.mkdir(parents=True, exist_ok=True)
        for item in stale_items + new_items:
            title = f"Draft - {Path(str(item['target_page'])).stem.replace('-', ' ').title()}"
            draft = draft_path(str(item["target_page"]), str(item["source_id"]))
            draft.write_text(
                render_draft(
                    title=title,
                    source_id=str(item["source_id"]),
                    source_hash=str(item["source_hash"]),
                    target_page=str(item["target_page"]),
                    raw_rel_path=str(item["raw_rel_path"]),
                    source_summary=str(item["source_summary"]),
                    change_summary=[str(entry) for entry in item.get("change_summary", [])],
                    compiled_from=[str(entry) for entry in item.get("compiled_from", []) if str(entry)],
                    reason=str(item["reason"]),
                ),
                encoding="utf-8",
            )
            written_drafts.append(draft.relative_to(ROOT).as_posix())

    report_text = build_report(stale_items, new_items, written_drafts)
    if args.dry_run:
        print(report_text)
    else:
        report_path = Path(args.report_file).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print(f"delta_compile: wrote {report_path}")
    print(
        f"delta_compile: OK ({len(stale_items)} stale page(s), "
        f"{len(new_items)} new raw source(s), {len(written_drafts)} draft(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
