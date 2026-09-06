from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: ci-safe (structural only)

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"

REQUIRED_FILES = [
    "README.md",
    "SCHEMA.md",
    "index.md",
    "log.md",
    "project-overview.md",
    "current-status.md",
    "sources-and-data.md",
    "github-and-raw-strategy.md",
]

LOG_HEADER_RE = re.compile(r"^## \[\d{4}-\d{2}-\d{2}\] .+ \| .+$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
FRONTMATTER_EXEMPT = {"index.md", "log.md", "README.md", "SCHEMA.md"}
REQUIRED_FRONTMATTER_KEYS = {"title", "source", "created"}

# v2 升级：log.md 支持表格格式（"table"）或 H2 日期头（"header"），默认 "table"
# 表格格式 = 第一行以 "| " 开头（markdown 表格）
LOG_FORMAT = "table"  # "table" | "header"


def is_log_table_format(text: str) -> bool:
    """log.md 是否用 markdown 表格（≥3 行以 | 开头）。"""
    table_lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return len(table_lines) >= 3

# Don't validate links inside fenced code (```), indented code, or inline code.
FENCE_RE = re.compile(r"^(\s*)(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
LINK_OPEN_RE = re.compile(r"!?\[(?:[^\[\]\\]|\\.)*\]\(")


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub("", line)


def iter_link_targets(text: str) -> list[str]:
    """Return every (link target) found outside fenced/inline code.

    Properly handles balanced parens inside the URL portion (rare but legal).
    """
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for raw_line in text.splitlines():
        m = FENCE_RE.match(raw_line)
        if m:
            marker = m.group(2)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue
        if in_fence:
            continue
        line = strip_inline_code(raw_line)
        pos = 0
        while True:
            m = LINK_OPEN_RE.search(line, pos)
            if not m:
                break
            start = m.end()
            depth = 1
            i = start
            while i < len(line) and depth > 0:
                ch = line[i]
                if ch == "\\" and i + 1 < len(line):
                    i += 2
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                target = line[start:i].strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1].strip()
                title_match = re.search(r'\s+["\'(].*["\')]$', target)
                if title_match:
                    target = target[: title_match.start()].strip()
                out.append(target)
                pos = i + 1
            else:
                pos = len(line)
    return out


def check_frontmatter(path: Path, text: str) -> list[str]:
    errs: list[str] = []
    if path.name in FRONTMATTER_EXEMPT:
        return errs
    m = FRONTMATTER_RE.match(text)
    if not m:
        errs.append(f"missing frontmatter: {path.relative_to(ROOT)}")
        return errs
    body = m.group(1)
    found_keys: set[str] = set()
    for line in body.splitlines():
        if ":" in line:
            key = line.split(":", 1)[0].strip()
            found_keys.add(key)
    missing = REQUIRED_FRONTMATTER_KEYS - found_keys
    if missing:
        errs.append(f"missing frontmatter keys in {path.relative_to(ROOT)}: {', '.join(sorted(missing))}")
    return errs


def resolve_link(source_file: Path, target: str) -> Path | None:
    if target.startswith(("http://", "https://", "mailto:", "tel:", "ftp://")):
        return None
    if target.startswith("#"):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return None
    return (source_file.parent / target).resolve()


def index_referenced_pages(index_path: Path) -> set[str]:
    """Set of wiki-relative posix paths that index.md actually links to.

    Uses real link parsing instead of substring matching, so a page named
    `policy.md` doesn't accidentally pass because index.md mentions
    `pricing-policy.md`.
    """
    if not index_path.exists():
        return set()
    text = index_path.read_text(encoding="utf-8")
    refs: set[str] = set()
    for raw_target in iter_link_targets(text):
        resolved = resolve_link(index_path, raw_target)
        if resolved is None:
            continue
        try:
            rel = resolved.relative_to(WIKI_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        refs.add(rel)
    return refs


def main() -> int:
    errors: list[str] = []
    if not WIKI_ROOT.exists():
        print("wiki_check: docs/wiki does not exist")
        return 1

    for rel in REQUIRED_FILES:
        if not (WIKI_ROOT / rel).exists():
            errors.append(f"missing required file: docs/wiki/{rel}")

    md_files = sorted(WIKI_ROOT.rglob("*.md"))
    index_refs = index_referenced_pages(WIKI_ROOT / "index.md")
    EXEMPT_FROM_INDEX = {"README.md", "SCHEMA.md", "log.md", "index.md"}

    fm_ok = 0
    for path in md_files:
        text = path.read_text(encoding="utf-8")

        fm_errors = check_frontmatter(path, text)
        errors.extend(fm_errors)
        if not fm_errors and path.name not in FRONTMATTER_EXEMPT:
            fm_ok += 1

        if path.name == "log.md":
            # v2 升级：表格格式下跳过 H2 日期头校验
            if LOG_FORMAT == "header":
                for line in text.splitlines():
                    if line.startswith("## ") and not LOG_HEADER_RE.match(line):
                        errors.append(f"bad log header format in {path.relative_to(ROOT)}: {line}")
            elif LOG_FORMAT == "table":
                if not is_log_table_format(text):
                    errors.append(f"log.md not in table format (LOG_FORMAT=table) in {path.relative_to(ROOT)}")

        rel = path.relative_to(WIKI_ROOT).as_posix()
        if rel not in EXEMPT_FROM_INDEX and rel not in index_refs:
            errors.append(f"index.md does not reference docs/wiki/{rel}")

        for raw_target in iter_link_targets(text):
            resolved = resolve_link(path, raw_target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(f"broken link in {path.relative_to(ROOT)} -> {raw_target}")

    if errors:
        print("wiki_check: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print("wiki_check: OK")
    print(f"- markdown files: {len(md_files)}")
    print(f"- required files: {len(REQUIRED_FILES)}")
    print(f"- frontmatter valid: {fm_ok}")
    print(f"- index.md links: {len(index_refs)}")
    print(f"- wiki root: {WIKI_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
