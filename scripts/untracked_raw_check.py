from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: ci-safe (scans only what exists; missing PROJECT_RAW_ROOT is fine)

import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "raw_sources.csv"

# File extensions that should be tracked in the manifest
RAW_EXTENSIONS = {
    ".pdf", ".xlsx", ".xls", ".xlsm", ".csv", ".tsv",
    ".doc", ".docx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp3", ".mp4", ".wav", ".mov",
}

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".obsidian", ".next", "dist", "build",
    "manifests",  # manifest CSVs are the index, not raw data
}


def load_manifest_filenames() -> set[str]:
    if not MANIFEST.exists():
        return set()
    names: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if fn:
                names.add(fn.lower())
    return names


def main() -> int:
    known = load_manifest_filenames()
    scan_roots = [ROOT]

    raw_root_env = os.environ.get("PROJECT_RAW_ROOT")
    if raw_root_env:
        raw_root = Path(raw_root_env).expanduser().resolve()
        if raw_root.exists():
            scan_roots.append(raw_root)

    untracked: list[tuple[str, Path]] = []

    for scan_root in scan_roots:
        for dirpath, dirnames, filenames in os.walk(scan_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext in RAW_EXTENSIONS and fn.lower() not in known:
                    full = Path(dirpath) / fn
                    rel = full.relative_to(scan_root) if full.is_relative_to(scan_root) else full
                    untracked.append((fn, rel))

    if untracked:
        print(f"untracked_raw_check: FOUND {len(untracked)} untracked raw file(s)")
        print("These files exist in the project but are NOT registered in manifests/raw_sources.csv:")
        print()
        for fn, rel in sorted(untracked, key=lambda x: str(x[1])):
            print(f"  {rel}")
        print()
        print("To fix: add each file to manifests/raw_sources.csv with status 'new',")
        print("then compile its key information into the relevant wiki page.")
        return 1

    print("untracked_raw_check: OK (no untracked raw files found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
