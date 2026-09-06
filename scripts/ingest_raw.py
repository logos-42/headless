from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only (needs PROJECT_RAW_ROOT)

import argparse
import csv
import hashlib
import json
import os
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "raw_sources.csv"
LOCK_FILE = ROOT / "manifests" / "raw_index.json"
REPORT_FILE = ROOT / "manifests" / "intake_report.md"
DEFAULT_RAW_ROOT = (ROOT.parent / "__RAW_ROOT_NAME__").resolve()
EXPECTED_COLUMNS = [
    "source_id",
    "company",
    "vendor",
    "kind",
    "filename",
    "raw_rel_path",
    "status",
    "compiled_into",
    "notes",
]
TRACKED_EXTENSIONS = {
    ".pdf", ".md", ".txt",
    ".xlsx", ".xls", ".xlsm", ".csv", ".tsv",
    ".doc", ".docx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg",
    ".zip", ".rar", ".7z", ".tar", ".gz",
}
SKIP_DIRS = {
    ".git", ".svn", "__pycache__", ".DS_Store",
    "node_modules", ".venv", "venv",
}
KEY_COLUMN_HINTS = ("sku", "part", "item", "model", "spec", "code", "pn", "material", "id", "series")
VALUE_COLUMN_HINTS = ("price", "cost", "amount", "qty", "quantity", "lead", "currency", "discount", "rate")
MAX_HEADERS = 8
MAX_ROW_WIDTH = 12
MAX_TRACKED_ROWS = 2000
MAX_SHEETS = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def safe_read_text(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" #	-")
        if stripped:
            return stripped[:120]
    return ""


def detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls", ".xlsm", ".csv", ".tsv"}:
        return "spreadsheet"
    if ext in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".md", ".txt"}:
        return "document"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}:
        return "image"
    if ext in {".zip", ".rar", ".7z", ".tar", ".gz"}:
        return "archive"
    return "raw"


def clean_cells(values: list[str], *, limit: int = MAX_ROW_WIDTH) -> list[str]:
    cleaned = [
        str(value or "").replace("\n", " ").replace("\r", " ").strip()[:80]
        for value in values[:limit]
    ]
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return cleaned


def guess_key_column(headers: list[str]) -> str:
    lowered = [(header, header.lower()) for header in headers if header]
    for hint in KEY_COLUMN_HINTS:
        for original, value in lowered:
            if hint in value:
                return original
    return headers[0] if headers else ""


def suspicious_columns(headers: list[str]) -> list[str]:
    flagged: list[str] = []
    for header in headers:
        lowered = header.lower()
        if any(hint in lowered for hint in KEY_COLUMN_HINTS + VALUE_COLUMN_HINTS):
            flagged.append(header)
    return flagged[:6]


def row_signature(cells: list[str]) -> str:
    payload = "|".join(clean_cells(cells, limit=MAX_ROW_WIDTH))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compare_named_lists(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    old = set(before)
    new = set(after)
    return sorted(new - old), sorted(old - new)


def compare_row_signatures(before: dict[str, str], after: dict[str, str]) -> str:
    if not before or not after:
        return ""
    old_keys = set(before)
    new_keys = set(after)
    added = len(new_keys - old_keys)
    removed = len(old_keys - new_keys)
    changed = len([key for key in old_keys & new_keys if before[key] != after[key]])
    parts: list[str] = []
    if added:
        parts.append(f"{added} added")
    if removed:
        parts.append(f"{removed} removed")
    if changed:
        parts.append(f"{changed} changed")
    return ", ".join(parts)


def summarize_delimited(path: Path, delimiter: str, parser: str) -> dict[str, object]:
    headers: list[str] = []
    key_column = ""
    key_index = -1
    row_count = 0
    column_count = 0
    tracked_rows: dict[str, str] = {}
    sample_rows: list[str] = []
    truncated = False
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for raw_row in reader:
            cleaned = clean_cells(raw_row)
            if not cleaned:
                continue
            column_count = max(column_count, len(cleaned))
            if not headers:
                headers = [cell[:60] for cell in cleaned[:MAX_HEADERS]]
                key_column = guess_key_column(headers)
                key_index = headers.index(key_column) if key_column in headers else -1
                continue
            row_count += 1
            if len(sample_rows) < 3:
                sample_rows.append(" | ".join(cleaned[:MAX_HEADERS]))
            if row_count > MAX_TRACKED_ROWS:
                truncated = True
                continue
            if key_index >= 0 and key_index < len(cleaned):
                key = cleaned[key_index]
                if key:
                    tracked_rows[key] = row_signature(cleaned)
    return {
        "parser": parser,
        "summary": (
            f"{row_count} data row(s); {column_count} column(s); "
            f"headers: {', '.join(headers) if headers else 'none'}"
        ),
        "metadata": {
            "row_count": row_count,
            "column_count": column_count,
            "headers": headers,
            "delimiter": delimiter,
            "key_column": key_column,
            "suspicious_columns": suspicious_columns(headers),
            "sample_rows": sample_rows,
            "tracked_rows": tracked_rows,
            "truncated": truncated,
        },
    }


def summarize_csv(path: Path) -> dict[str, object]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    return summarize_delimited(path, delimiter, "csv-local")


def normalize_zip_path(base: str, target: str) -> str:
    parts = [part for part in (Path(base).parent / target).as_posix().split("/") if part not in {"", "."}]
    normalized: list[str] = []
    for part in parts:
        if part == "..":
            if normalized:
                normalized.pop()
            continue
        normalized.append(part)
    return "/".join(normalized)


def spreadsheet_column_index(label: str) -> int:
    total = 0
    for char in label.upper():
        if not char.isalpha():
            break
        total = total * 26 + (ord(char) - 64)
    return total


def parse_sheet_dimension(ref: str) -> tuple[int, int]:
    if not ref:
        return 0, 0
    tail = ref.split(":")[-1]
    letters = "".join(char for char in tail if char.isalpha())
    digits = "".join(char for char in tail if char.isdigit())
    return int(digits or 0), spreadsheet_column_index(letters)


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        with zf.open("xl/sharedStrings.xml") as handle:
            root = ET.fromstring(handle.read())
        return ["".join(node.itertext()).strip() for node in root.findall(".//{*}si")]
    except Exception:
        return []


def read_workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    with zf.open("xl/workbook.xml") as handle:
        workbook_root = ET.fromstring(handle.read())
    rel_map: dict[str, str] = {}
    try:
        with zf.open("xl/_rels/workbook.xml.rels") as handle:
            rel_root = ET.fromstring(handle.read())
        for node in rel_root.findall(".//{*}Relationship"):
            rel_id = node.attrib.get("Id", "")
            target = node.attrib.get("Target", "")
            if rel_id and target:
                rel_map[rel_id] = normalize_zip_path("xl/workbook.xml", target)
    except Exception:
        rel_map = {}
    sheets: list[tuple[str, str]] = []
    for node in workbook_root.findall(".//{*}sheet"):
        name = node.attrib.get("name", "")
        rel_id = ""
        for key, value in node.attrib.items():
            if key.endswith("}id") or key == "id":
                rel_id = value
                break
        target = rel_map.get(rel_id, "")
        if name and target:
            sheets.append((name, target))
    return sheets


def resolve_xlsx_cell(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(cell.itertext()).strip()
    value = cell.findtext("{*}v", default="").strip()
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except Exception:
            return value
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    if value:
        return value
    return "".join(cell.itertext()).strip()


def summarize_xlsx_sheet(zf: zipfile.ZipFile, sheet_name: str, sheet_path: str, shared_strings: list[str]) -> dict[str, object]:
    with zf.open(sheet_path) as handle:
        root = ET.fromstring(handle.read())
    dimension_node = root.find("{*}dimension")
    dim_rows, dim_cols = parse_sheet_dimension(dimension_node.attrib.get("ref", "") if dimension_node is not None else "")
    headers: list[str] = []
    key_column = ""
    key_index = -1
    row_count = 0
    column_count = 0
    tracked_rows: dict[str, str] = {}
    sample_rows: list[str] = []
    truncated = False
    for row_node in root.findall(".//{*}sheetData/{*}row"):
        cells_by_index: dict[int, str] = {}
        for cell in row_node.findall("{*}c"):
            ref = cell.attrib.get("r", "")
            letters = "".join(char for char in ref if char.isalpha())
            index = spreadsheet_column_index(letters) or (len(cells_by_index) + 1)
            cells_by_index[index] = resolve_xlsx_cell(cell, shared_strings)
        if not cells_by_index:
            continue
        max_index = min(max(cells_by_index), MAX_ROW_WIDTH)
        cleaned = clean_cells([cells_by_index.get(idx, "") for idx in range(1, max_index + 1)])
        if not cleaned:
            continue
        column_count = max(column_count, len(cleaned))
        if not headers:
            headers = [cell[:60] for cell in cleaned[:MAX_HEADERS]]
            key_column = guess_key_column(headers)
            key_index = headers.index(key_column) if key_column in headers else -1
            continue
        row_count += 1
        if len(sample_rows) < 3:
            sample_rows.append(" | ".join(cleaned[:MAX_HEADERS]))
        if row_count > MAX_TRACKED_ROWS:
            truncated = True
            continue
        if key_index >= 0 and key_index < len(cleaned):
            key = cleaned[key_index]
            if key:
                tracked_rows[key] = row_signature(cleaned)
    return {
        "name": sheet_name,
        "row_count": row_count or dim_rows,
        "column_count": column_count or dim_cols,
        "headers": headers,
        "key_column": key_column,
        "suspicious_columns": suspicious_columns(headers),
        "sample_rows": sample_rows,
        "tracked_rows": tracked_rows,
        "truncated": truncated,
    }


def summarize_xlsx(path: Path) -> dict[str, object]:
    sheets: list[dict[str, object]] = []
    try:
        with zipfile.ZipFile(path) as zf:
            shared_strings = read_shared_strings(zf)
            for sheet_name, sheet_path in read_workbook_sheets(zf)[:MAX_SHEETS]:
                try:
                    sheets.append(summarize_xlsx_sheet(zf, sheet_name, sheet_path, shared_strings))
                except Exception:
                    sheets.append({
                        "name": sheet_name,
                        "row_count": 0,
                        "column_count": 0,
                        "headers": [],
                        "key_column": "",
                        "suspicious_columns": [],
                        "sample_rows": [],
                        "tracked_rows": {},
                        "truncated": False,
                    })
    except Exception:
        sheets = []
    sheet_names = [sheet["name"] for sheet in sheets]
    summary_bits = [
        f"{sheet['name']}[{sheet.get('row_count', 0)}x{sheet.get('column_count', 0)}]"
        for sheet in sheets[:4]
    ]
    return {
        "parser": "xlsx-local",
        "summary": f"{len(sheet_names)} sheet(s): {', '.join(summary_bits) if summary_bits else 'unknown'}",
        "metadata": {"sheet_names": sheet_names, "sheets": sheets},
    }


def summarize_docx(path: Path) -> dict[str, object]:
    paragraph_count = 0
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("word/document.xml") as handle:
                root = ET.fromstring(handle.read())
            paragraph_count = len(root.findall(".//{*}p"))
    except Exception:
        pass
    return {
        "parser": "docx-local",
        "summary": f"{paragraph_count} paragraph block(s)",
        "metadata": {"paragraph_blocks": paragraph_count},
    }


def summarize_pptx(path: Path) -> dict[str, object]:
    slide_count = 0
    try:
        with zipfile.ZipFile(path) as zf:
            slide_count = len([name for name in zf.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")])
    except Exception:
        pass
    return {
        "parser": "pptx-local",
        "summary": f"{slide_count} slide(s)",
        "metadata": {"slide_count": slide_count},
    }


def summarize_pdf(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    page_count = raw.count(b"/Type /Page")
    return {
        "parser": "pdf-local",
        "summary": f"{page_count or 'unknown'} page(s)",
        "metadata": {"page_count": int(page_count)},
    }


def image_size(path: Path) -> tuple[int | None, int | None]:
    raw = path.read_bytes()[:64]
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw[:6] in {b"GIF87a", b"GIF89a"} and len(raw) >= 10:
        return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
    if raw.startswith(b"BM") and len(raw) >= 26:
        return int.from_bytes(raw[18:22], "little"), int.from_bytes(raw[22:26], "little")
    if raw.startswith(b"\xff\xd8"):
        with path.open("rb") as handle:
            handle.read(2)
            while True:
                marker_prefix = handle.read(1)
                if marker_prefix != b"\xff":
                    break
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    _length = int.from_bytes(handle.read(2), "big")
                    handle.read(1)
                    height = int.from_bytes(handle.read(2), "big")
                    width = int.from_bytes(handle.read(2), "big")
                    return width, height
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                seg_length = int.from_bytes(handle.read(2), "big")
                handle.seek(seg_length - 2, 1)
    return None, None


def summarize_image(path: Path) -> dict[str, object]:
    width, height = image_size(path)
    dims = f"{width}x{height}" if width and height else "unknown"
    return {
        "parser": "image-local",
        "summary": f"dimensions: {dims}",
        "metadata": {"width": width, "height": height},
    }


def summarize_archive(path: Path) -> dict[str, object]:
    ext = path.suffix.lower()
    entry_count = None
    parser = "archive-local"
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path) as zf:
                entry_count = len(zf.namelist())
        elif ext in {".tar", ".gz"} or path.name.endswith(".tar.gz"):
            with tarfile.open(path) as tf:
                entry_count = len(tf.getmembers())
    except Exception:
        entry_count = None
    return {
        "parser": parser,
        "summary": f"archive entries: {entry_count if entry_count is not None else 'unknown'}",
        "metadata": {"entry_count": entry_count, "format": ext.lstrip('.')},
    }


def summarize_plaintext(path: Path) -> dict[str, object]:
    text = safe_read_text(path)
    summary = first_nonempty_line(text) or "no preview"
    return {
        "parser": "text-local",
        "summary": summary,
        "metadata": {"preview": summary},
    }


def summarize_xls_legacy(path: Path) -> dict[str, object]:
    return {
        "parser": "xls-legacy",
        "summary": "legacy .xls workbook (sheet metadata unavailable without extra parser)",
        "metadata": {"format": "xls", "size_bytes": path.stat().st_size},
    }


def summarize_csv_change(previous: dict[str, object], current: dict[str, object]) -> list[str]:
    notes: list[str] = []
    prev_headers = [str(item) for item in previous.get("headers", [])]
    curr_headers = [str(item) for item in current.get("headers", [])]
    added_headers, removed_headers = compare_named_lists(prev_headers, curr_headers)
    if added_headers:
        notes.append(f"headers added: {', '.join(added_headers[:4])}")
    if removed_headers:
        notes.append(f"headers removed: {', '.join(removed_headers[:4])}")
    prev_rows = int(previous.get("row_count", 0) or 0)
    curr_rows = int(current.get("row_count", 0) or 0)
    if prev_rows != curr_rows:
        notes.append(f"rows {prev_rows} -> {curr_rows}")
    prev_key = str(previous.get("key_column", "") or "")
    curr_key = str(current.get("key_column", "") or "")
    if prev_key != curr_key and curr_key:
        notes.append(f"key column {prev_key or 'none'} -> {curr_key}")
    row_changes = compare_row_signatures(
        {str(k): str(v) for k, v in dict(previous.get("tracked_rows", {})).items()},
        {str(k): str(v) for k, v in dict(current.get("tracked_rows", {})).items()},
    )
    if row_changes:
        notes.append(f"tracked rows: {row_changes}")
    if not notes:
        notes.append("content changed; structural summary unchanged")
    return notes


def summarize_xlsx_change(previous: dict[str, object], current: dict[str, object]) -> list[str]:
    notes: list[str] = []
    prev_sheets = {str(sheet.get("name", "")): sheet for sheet in previous.get("sheets", []) if sheet.get("name")}
    curr_sheets = {str(sheet.get("name", "")): sheet for sheet in current.get("sheets", []) if sheet.get("name")}
    added_sheets, removed_sheets = compare_named_lists(list(prev_sheets), list(curr_sheets))
    if added_sheets:
        notes.append(f"sheets added: {', '.join(added_sheets[:4])}")
    if removed_sheets:
        notes.append(f"sheets removed: {', '.join(removed_sheets[:4])}")
    for sheet_name in sorted(prev_sheets.keys() & curr_sheets.keys())[:4]:
        before = prev_sheets[sheet_name]
        after = curr_sheets[sheet_name]
        sheet_notes: list[str] = []
        if int(before.get("row_count", 0) or 0) != int(after.get("row_count", 0) or 0):
            sheet_notes.append(f"rows {before.get('row_count', 0)} -> {after.get('row_count', 0)}")
        if int(before.get("column_count", 0) or 0) != int(after.get("column_count", 0) or 0):
            sheet_notes.append(f"cols {before.get('column_count', 0)} -> {after.get('column_count', 0)}")
        added_headers, removed_headers = compare_named_lists(
            [str(item) for item in before.get("headers", [])],
            [str(item) for item in after.get("headers", [])],
        )
        if added_headers:
            sheet_notes.append(f"headers +{', '.join(added_headers[:3])}")
        if removed_headers:
            sheet_notes.append(f"headers -{', '.join(removed_headers[:3])}")
        row_changes = compare_row_signatures(
            {str(k): str(v) for k, v in dict(before.get("tracked_rows", {})).items()},
            {str(k): str(v) for k, v in dict(after.get("tracked_rows", {})).items()},
        )
        if row_changes:
            sheet_notes.append(row_changes)
        if sheet_notes:
            notes.append(f"sheet {sheet_name}: {'; '.join(sheet_notes)}")
    if not notes:
        notes.append("content changed; workbook structure summary unchanged")
    return notes


def summarize_change(previous_entry: dict[str, object] | None, current_payload: dict[str, object]) -> list[str]:
    if not previous_entry:
        return []
    previous_parser = str(previous_entry.get("parser", ""))
    current_parser = str(current_payload.get("parser", ""))
    previous_meta = dict(previous_entry.get("metadata", {}))
    current_meta = dict(current_payload.get("metadata", {}))
    if previous_parser == current_parser == "csv-local":
        return summarize_csv_change(previous_meta, current_meta)
    if previous_parser == current_parser == "xlsx-local":
        return summarize_xlsx_change(previous_meta, current_meta)
    return [f"summary: {previous_entry.get('summary', 'unknown')} -> {current_payload.get('summary', 'unknown')}"]


def summarize_file(path: Path) -> dict[str, object]:
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        return summarize_csv(path)
    if ext in {".xlsx", ".xlsm"}:
        return summarize_xlsx(path)
    if ext == ".xls":
        return summarize_xls_legacy(path)
    if ext == ".docx":
        return summarize_docx(path)
    if ext == ".pptx":
        return summarize_pptx(path)
    if ext == ".pdf":
        return summarize_pdf(path)
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}:
        return summarize_image(path)
    if ext in {".zip", ".rar", ".7z", ".tar", ".gz"} or path.name.endswith(".tar.gz"):
        return summarize_archive(path)
    return summarize_plaintext(path)


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"manifest missing: {MANIFEST}")
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"manifest columns mismatch: expected {EXPECTED_COLUMNS}, got {reader.fieldnames}")
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EXPECTED_COLUMNS})


def load_lock() -> dict[str, object]:
    if not LOCK_FILE.exists():
        return {"files": {}}
    try:
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}


def write_lock(payload: dict[str, object]) -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def next_source_id(existing_ids: set[str], content_hash: str) -> str:
    base = f"src_{content_hash[:10]}"
    candidate = base
    index = 2
    while candidate in existing_ids:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def build_report(
    raw_root: Path,
    rows: list[dict[str, str]],
    kinds: Counter[str],
    new_paths: list[str],
    changed_paths: list[str],
    archived_paths: list[str],
    duplicate_paths: list[str],
    change_summaries: dict[str, list[str]],
) -> str:
    def bullets(items: list[str], *, details: dict[str, list[str]] | None = None) -> str:
        if not items:
            return "- none\n"
        lines: list[str] = []
        for item in items[:20]:
            detail_lines = (details or {}).get(item, [])
            if detail_lines:
                lines.append(f"- `{item}` — {detail_lines[0]}\n")
                for extra in detail_lines[1:3]:
                    lines.append(f"  - {extra}\n")
            else:
                lines.append(f"- `{item}`\n")
        return "".join(lines)

    lines = [
        "# Raw Intake Report",
        "",
        f"- generated_at: `{utc_now()}`",
        f"- raw_root: `{raw_root}`",
        f"- manifest_rows: `{len(rows)}`",
        f"- new: `{len(new_paths)}`",
        f"- changed: `{len(changed_paths)}`",
        f"- archived: `{len(archived_paths)}`",
        f"- duplicates: `{len(duplicate_paths)}`",
        "",
        "## Kind Summary",
        "",
    ]
    if kinds:
        for kind, count in sorted(kinds.items()):
            lines.append(f"- `{kind}`: {count}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## New Files",
        "",
        bullets(sorted(new_paths)),
        "",
        "## Changed Files",
        "",
        bullets(sorted(changed_paths), details=change_summaries),
        "",
        "## Archived Files",
        "",
        bullets(sorted(archived_paths)),
        "",
        "## Duplicate Files",
        "",
        bullets(sorted(duplicate_paths)),
    ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a local raw root, update the manifest, and record low-cost structural metadata.")
    parser.add_argument("--raw-root", default=os.environ.get("PROJECT_RAW_ROOT", str(DEFAULT_RAW_ROOT)), help="Local raw root path")
    parser.add_argument("--report-file", default=str(REPORT_FILE), help="Markdown report output path")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing manifest or lock files")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"ingest_raw: raw root does not exist: {raw_root}")
        return 1

    rows = load_manifest()
    previous_lock = load_lock().get("files", {})
    rows_by_path = {row["raw_rel_path"]: row for row in rows if row.get("raw_rel_path")}
    existing_ids = {row["source_id"] for row in rows if row.get("source_id")}
    seen_paths: set[str] = set()
    kinds: Counter[str] = Counter()
    new_paths: list[str] = []
    changed_paths: list[str] = []
    archived_paths: list[str] = []
    lock_entries: dict[str, object] = {}
    hash_to_primary: dict[str, str] = {}
    duplicate_paths: list[str] = []
    change_summaries: dict[str, list[str]] = {}

    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(raw_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            path = Path(dirpath) / name
            ext = path.suffix.lower()
            if ext in TRACKED_EXTENSIONS or path.name.endswith(".tar.gz"):
                candidates.append(path)

    for path in sorted(candidates):
        rel = path.relative_to(raw_root).as_posix()
        seen_paths.add(rel)
        content_hash = sha256_prefix(path)
        kind = detect_kind(path)
        kinds[kind] += 1
        summary_payload = summarize_file(path)
        existing = rows_by_path.get(rel)
        old_hash = None
        previous_entry = None
        if isinstance(previous_lock, dict) and rel in previous_lock:
            previous_entry = previous_lock[rel]
            old_hash = previous_entry.get("content_hash")

        if existing is None:
            source_id = next_source_id(existing_ids, content_hash)
            existing_ids.add(source_id)
            row = {
                "source_id": source_id,
                "company": "",
                "vendor": "",
                "kind": kind,
                "filename": path.name,
                "raw_rel_path": rel,
                "status": "new",
                "compiled_into": "",
                "notes": "",
            }
            rows.append(row)
            rows_by_path[rel] = row
            new_paths.append(rel)
            existing = row
        else:
            existing["kind"] = kind
            existing["filename"] = path.name
            if old_hash and old_hash != content_hash and existing.get("status") != "archived":
                existing["status"] = "new"
                changed_paths.append(rel)
                change_summaries[rel] = summarize_change(previous_entry, summary_payload)

        primary = hash_to_primary.setdefault(content_hash, rel)
        duplicate_of = None if primary == rel else rows_by_path[primary]["source_id"]
        if duplicate_of:
            duplicate_paths.append(rel)

        lock_entries[rel] = {
            "source_id": existing["source_id"],
            "filename": path.name,
            "kind": kind,
            "content_hash": content_hash,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
            "parser": summary_payload["parser"],
            "summary": summary_payload["summary"],
            "metadata": summary_payload["metadata"],
            "duplicate_of": duplicate_of,
            "previous_content_hash": old_hash or "",
            "change_summary": change_summaries.get(rel, []),
        }

    for row in rows:
        rel = row.get("raw_rel_path", "")
        if rel and rel not in seen_paths and row.get("status") != "archived":
            row["status"] = "archived"
            archived_paths.append(rel)

    rows.sort(key=lambda row: (row.get("status", ""), row.get("raw_rel_path", "")))
    report_text = build_report(raw_root, rows, kinds, new_paths, changed_paths, archived_paths, duplicate_paths, change_summaries)

    if args.dry_run:
        print("ingest_raw: DRY RUN")
        print(report_text)
        return 0

    write_manifest(rows)
    write_lock({
        "llm_wiki_version": "1.2.2",
        "generated_at": utc_now(),
        "raw_root": str(raw_root),
        "summary": {
            "tracked_files": len(candidates),
            "manifest_rows": len(rows),
            "new": len(new_paths),
            "changed": len(changed_paths),
            "archived": len(archived_paths),
            "duplicates": len(duplicate_paths),
            "kinds": dict(sorted(kinds.items())),
        },
        "files": lock_entries,
    })
    report_path = Path(args.report_file).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(f"ingest_raw: OK ({len(candidates)} tracked file(s))")
    print(f"- manifest: {MANIFEST}")
    print(f"- lock: {LOCK_FILE}")
    print(f"- report: {report_path}")
    print(f"- new: {len(new_paths)} | changed: {len(changed_paths)} | archived: {len(archived_paths)} | duplicates: {len(duplicate_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
