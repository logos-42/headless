from __future__ import annotations
# runtime: dev-only (writes outside repo)

import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COPY_FILES = [
    "AGENTS.md",
    "scripts/wiki_check.py",
    "scripts/raw_manifest_check.py",
    "scripts/init_raw_root.py",
]

COPY_DIRS = [
    ("docs/wiki", "docs/wiki"),
    ("manifests", "manifests"),
]

OPTIONAL_COPY_DIRS = [
    ("verified_cases", "verified_cases"),
    ("backend/verified_cases", "verified_cases"),
]


def wipe_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_dir(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the repo memory surface into a dedicated GitHub/wiki repo checkout.")
    parser.add_argument("target_dir", help="Target memory repo checkout directory")
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    managed_paths = [
        target / "AGENTS.md",
        target / "README.md",
        target / "docs",
        target / "manifests",
        target / "verified_cases",
        target / "scripts",
    ]
    for path in managed_paths:
        wipe_path(path)

    exported_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = f"""# Memory Repo

This repo is the exported long-term memory surface for a project.

## What belongs here

- `docs/wiki/`
- `manifests/`
- `verified_cases/`
- repo-level operating rules such as `AGENTS.md`

## What does not belong here

- raw PDFs / XLSX / XLS / ZIP / RAR files
- main application runtime code
- build artifacts

## Export time

`{exported_at}`

## Source repo

`{ROOT}`

## Refresh

Run from the source repo:

```bash
python3 scripts/export_memory_repo.py /path/to/memory-repo-checkout
```
"""
    (target / "README.md").write_text(readme, encoding="utf-8")

    for rel in COPY_FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, target / rel)

    for src_rel, dest_rel in COPY_DIRS:
        src = ROOT / src_rel
        if src.exists():
            copy_dir(src, target / dest_rel)

    for src_rel, dest_rel in OPTIONAL_COPY_DIRS:
        src = ROOT / src_rel
        if src.exists():
            copy_dir(src, target / dest_rel)
            break

    print(f"Exported memory repo snapshot to: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
