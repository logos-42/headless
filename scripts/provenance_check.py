from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only (needs PROJECT_RAW_ROOT pointing at real raw files)
#   Use --ci to skip raw-file resolution and only verify that source_hash is
#   present in every non-session page. That sub-check is safe to run in CI.

import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"
MANIFEST = ROOT / "manifests" / "raw_sources.csv"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
SOURCE_HASH_RE = re.compile(r"source_hash:\s*([a-f0-9]{12,})")

SKIP_FILES = {"index.md", "log.md", "README.md", "SCHEMA.md"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_manifest_paths() -> dict[str, Path]:
    if not MANIFEST.exists():
        return {}
    raw_root_env = os.environ.get("PROJECT_RAW_ROOT")
    raw_root = Path(raw_root_env).expanduser().resolve() if raw_root_env else None
    result: dict[str, Path] = {}
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("source_id") or "").strip()
            rel = (row.get("raw_rel_path") or "").strip()
            if sid and rel and raw_root:
                result[sid] = (raw_root / rel).resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify wiki pages reflect current raw sources.")
    parser.add_argument("--ci", action="store_true",
                        help="Skip raw-file resolution; only verify that every non-session "
                             "page has a source_hash field. Safe to run without raw files.")
    args = parser.parse_args()
    # Explicit opt-in only. Don't auto-detect generic CI=true — users may
    # mount raw files in CI and want the full check.
    ci_mode = args.ci or os.environ.get("LLM_WIKI_CI") == "1"

    if not WIKI_ROOT.exists():
        print("provenance_check: docs/wiki does not exist")
        return 1

    manifest_paths = {} if ci_mode else load_manifest_paths()
    checked = 0
    fresh = 0
    stale: list[tuple[str, str, str]] = []
    no_hash: list[str] = []
    unresolved: list[str] = []
    session_exempt = 0

    for path in sorted(WIKI_ROOT.rglob("*.md")):
        if path.name in SKIP_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        m = FRONTMATTER_RE.match(text)
        if not m:
            continue

        fm = m.group(1)
        source_line = ""
        for line in fm.splitlines():
            if line.startswith("source:"):
                source_line = line.split(":", 1)[1].strip()
                break

        if source_line == "session":
            session_exempt += 1
            continue

        hash_match = SOURCE_HASH_RE.search(fm)
        if not hash_match:
            no_hash.append(path.relative_to(ROOT).as_posix())
            continue

        stored_hash = hash_match.group(1)
        checked += 1

        if ci_mode:
            # Structural check only: source_hash exists, that's all CI can verify.
            fresh += 1
            continue

        # Find the source file
        if not source_line:
            fresh += 1
            continue

        # Try to resolve source path
        source_path = None
        if source_line.startswith("raw/"):
            candidate = ROOT / source_line
            if candidate.exists():
                source_path = candidate
        for sid, spath in manifest_paths.items():
            if sid in source_line or source_line in str(spath):
                source_path = spath
                break

        if source_path and source_path.exists():
            current = file_hash(source_path)
            if current == stored_hash:
                fresh += 1
            else:
                stale.append((
                    path.relative_to(ROOT).as_posix(),
                    stored_hash,
                    current,
                ))
        else:
            unresolved.append(path.relative_to(ROOT).as_posix())

    if stale:
        print(f"provenance_check: {len(stale)} STALE page(s) detected")
        for page, old, new in stale:
            print(f"  {page}: hash was {old}, source now {new}")
        print()
        print("These wiki pages were compiled from source files that have since changed.")
        print("Recompile them to update the wiki with current information.")

    if no_hash:
        print(f"provenance_check: {len(no_hash)} page(s) without source_hash (required for non-session sources)")
        for page in no_hash:
            print(f"  {page}")

    if unresolved:
        print(f"provenance_check: {len(unresolved)} page(s) with unresolved source")
        for page in unresolved:
            print(f"  {page}")

    if not stale and not no_hash and not unresolved:
        suffix = " [ci-mode: structural check only]" if ci_mode else ""
        print(
            f"provenance_check: OK ({checked} checked, {fresh} fresh, "
            f"{session_exempt} session-exempt){suffix}"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
