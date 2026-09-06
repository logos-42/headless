from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: ci-safe (skips raw existence check when PROJECT_RAW_ROOT unset)

import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "raw_sources.csv"
META = ROOT / "manifests" / "raw_sources.meta.json"

# Schema versions this script knows how to validate.
KNOWN_SCHEMA_VERSIONS = {1, 2}
LATEST_SCHEMA_VERSION = 2

# Default schema for projects bootstrapped before meta.json existed.
LEGACY_DEFAULT = {
    "schema_version": 1,
    "columns": [
        "source_id",
        "company",
        "vendor",
        "kind",
        "filename",
        "raw_rel_path",
        "status",
        "compiled_into",
        "notes",
    ],
    "allowed_status": ["new", "compiled", "archived"],
}


def load_schema() -> tuple[dict, str]:
    """Return (schema_dict, source) where source is 'meta' or 'legacy'."""
    if META.exists():
        try:
            data = json.loads(META.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"raw_manifest_check: malformed {META.name}: {exc}")
        return data, "meta"
    return LEGACY_DEFAULT, "legacy"


def main() -> int:
    errors: list[str] = []
    if not MANIFEST.exists():
        print(f"raw_manifest_check: missing {MANIFEST}")
        return 1

    schema, schema_source = load_schema()
    schema_version = schema.get("schema_version", 1)

    if schema_version not in KNOWN_SCHEMA_VERSIONS:
        if schema_version > LATEST_SCHEMA_VERSION:
            print(f"raw_manifest_check: SKIPPED")
            print(f"- manifest schema_version is {schema_version}, this script knows only up to {LATEST_SCHEMA_VERSION}")
            print(f"- run: python3 scripts/version_check.py  (or upgrade LLM-wiki to a newer release)")
            return 0  # Forward-compat: don't fail the user's CI on a newer manifest.
        errors.append(f"unknown schema_version {schema_version} (this script supports {sorted(KNOWN_SCHEMA_VERSIONS)})")

    expected_columns = schema["columns"]
    allowed_status = set(schema["allowed_status"])

    raw_root_env = os.environ.get("PROJECT_RAW_ROOT")
    raw_root = Path(raw_root_env).expanduser().resolve() if raw_root_env else None

    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            errors.append(f"manifest columns mismatch: expected {expected_columns}, got {reader.fieldnames}")
        seen_ids: set[str] = set()
        for idx, row in enumerate(reader, start=2):
            source_id = (row.get("source_id") or "").strip()
            filename = (row.get("filename") or "").strip()
            raw_rel_path = (row.get("raw_rel_path") or "").strip()
            status = (row.get("status") or "").strip()
            compiled_into = (row.get("compiled_into") or "").strip()

            if not source_id:
                errors.append(f"row {idx}: empty source_id")
            elif source_id in seen_ids:
                errors.append(f"row {idx}: duplicate source_id {source_id}")
            else:
                seen_ids.add(source_id)

            if not filename:
                errors.append(f"row {idx}: empty filename")
            if not raw_rel_path:
                errors.append(f"row {idx}: empty raw_rel_path")
            if status not in allowed_status:
                errors.append(f"row {idx}: bad status {status}")

            if raw_root and raw_rel_path:
                candidate = (raw_root / raw_rel_path).resolve()
                if not candidate.exists():
                    errors.append(f"row {idx}: missing local raw file {candidate}")

            if status == "compiled" and not compiled_into:
                errors.append(f"row {idx}: status {status} requires compiled_into")

    if errors:
        print("raw_manifest_check: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print("raw_manifest_check: OK")
    print(f"- manifest: {MANIFEST}")
    print(f"- schema: v{schema_version} (loaded from {schema_source})")
    if raw_root:
        print(f"- PROJECT_RAW_ROOT: {raw_root}")
    else:
        print("- PROJECT_RAW_ROOT: not set, existence checks skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
