from __future__ import annotations
# runtime: dev-only (one-shot setup)

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = (ROOT.parent / "__RAW_ROOT_NAME__").resolve()

DIRS = [
    "inbox",
    "internal_sources",
    "external_sources",
    "customer_answers",
    "screenshots",
    "extracted_text",
    "archive",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize the local raw root directory.")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.environ.get("PROJECT_RAW_ROOT", str(DEFAULT_RAW_ROOT)),
        help="Absolute or relative path for the local raw root",
    )
    args = parser.parse_args()

    raw_root = Path(args.path).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    for rel in DIRS:
        (raw_root / rel).mkdir(parents=True, exist_ok=True)

    print(f"Initialized raw root: {raw_root}")
    for rel in DIRS:
        print(f"- {raw_root / rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
