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
REPORT_FILE = ROOT / "manifests" / "stale_report.md"
DEFAULT_RAW_ROOT = (ROOT.parent / "__RAW_ROOT_NAME__").resolve()
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
SKIP_FILES = {"index.md", "log.md", "README.md", "SCHEMA.md"}


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


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
    items = [item.strip().strip("'").strip('"') for item in raw.split(",")]
    return [item for item in items if item]


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


def build_report(
    raw_root: Path | None,
    session_exempt: int,
    fresh: list[str],
    stale: list[str],
    missing_hash: list[str],
    unresolved: list[str],
    archived_refs: list[str],
    manifest_new: list[str],
) -> str:
    def section(title: str, items: list[str]) -> list[str]:
        lines = ["", f"## {title}", ""]
        if not items:
            lines.append("- none")
        else:
            lines.extend(f"- `{item}`" for item in items[:40])
        return lines

    lines = [
        "# Stale Report",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}`",
        f"- raw_root: `{raw_root}`" if raw_root else "- raw_root: `not configured`",
        f"- fresh_pages: `{len(fresh)}`",
        f"- stale_pages: `{len(stale)}`",
        f"- missing_hash: `{len(missing_hash)}`",
        f"- unresolved_sources: `{len(unresolved)}`",
        f"- archived_refs: `{len(archived_refs)}`",
        f"- manifest_new: `{len(manifest_new)}`",
        f"- session_exempt: `{session_exempt}`",
    ]
    lines += section("Fresh Pages", fresh)
    lines += section("Pages Needing Recompile (stale)", stale)
    lines += section("Pages Missing source_hash", missing_hash)
    lines += section("Pages With Unresolved source", unresolved)
    lines += section("Pages Pointing At Archived Sources", archived_refs)
    lines += section("Raw Files Still Waiting For First Compile", manifest_new)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report stale wiki pages and raw files that still need compilation.")
    parser.add_argument("--raw-root", default=os.environ.get("PROJECT_RAW_ROOT", ""), help="Local raw root path. If empty, falls back to the default sibling raw root when present.")
    parser.add_argument("--report-file", default=str(REPORT_FILE), help="Markdown report output path")
    parser.add_argument("--dry-run", action="store_true", help="Print the report without writing it")
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
    fresh: list[str] = []
    stale: list[str] = []
    missing_hash: list[str] = []
    unresolved: list[str] = []
    archived_refs: list[str] = []
    referenced_source_ids: set[str] = set()
    referenced_paths: set[str] = set()
    session_exempt = 0

    for path in sorted(WIKI_ROOT.rglob("*.md")):
        if path.name in SKIP_FILES:
            continue
        fm = parse_frontmatter(path)
        if not fm:
            continue
        rel = path.relative_to(ROOT).as_posix()
        source = fm.get("source", "")
        compiled_from = parse_list_field(fm.get("compiled_from", ""))
        source_hash = fm.get("source_hash", "")
        if source == "session":
            session_exempt += 1
            continue
        if not source_hash:
            missing_hash.append(rel)
            continue

        row = resolve_row(source, rows)
        if not row:
            unresolved.append(rel)
            continue

        referenced_rows = [row]
        for extra in compiled_from:
            extra_row = resolve_row(extra, rows)
            if not extra_row:
                unresolved.append(f"{rel} <- {extra}")
                continue
            referenced_rows.append(extra_row)

        for referenced_row in referenced_rows:
            if referenced_row.get("source_id"):
                referenced_source_ids.add(referenced_row["source_id"])
            if referenced_row.get("raw_rel_path"):
                referenced_paths.add(referenced_row["raw_rel_path"])

        if row.get("status") == "archived":
            archived_refs.append(rel)
            continue

        for extra_row in referenced_rows[1:]:
            if extra_row.get("status") == "archived" and extra_row.get("raw_rel_path"):
                archived_refs.append(f"{rel} <- {extra_row['raw_rel_path']}")

        current_hash = ""
        if raw_root:
            source_path = raw_root / row["raw_rel_path"]
            if source_path.exists():
                current_hash = sha256_prefix(source_path)
        if not current_hash and row["raw_rel_path"] in lock:
            current_hash = lock[row["raw_rel_path"]].get("content_hash", "")
        if not current_hash:
            unresolved.append(rel)
            continue
        if current_hash != source_hash:
            stale.append(f"{rel} <- {row['raw_rel_path']} ({source_hash} -> {current_hash})")
        else:
            fresh.append(rel)

    manifest_new = [
        row["raw_rel_path"]
        for row in rows
        if row.get("status") == "new"
        and row.get("raw_rel_path")
        and row.get("source_id") not in referenced_source_ids
        and row.get("raw_rel_path") not in referenced_paths
    ]

    report_text = build_report(raw_root, session_exempt, fresh, stale, missing_hash, unresolved, archived_refs, manifest_new)

    if args.dry_run:
        print(report_text)
    else:
        report_path = Path(args.report_file).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print(f"stale_report: wrote {report_path}")

    if stale or missing_hash or unresolved or archived_refs:
        print(
            f"stale_report: ATTENTION ({len(stale)} stale, {len(missing_hash)} missing_hash, "
            f"{len(unresolved)} unresolved, {len(archived_refs)} archived_refs)"
        )
        return 1

    print(
        f"stale_report: OK ({len(fresh)} fresh, {len(manifest_new)} manifest-new, "
        f"{session_exempt} session-exempt)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
