from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only

"""
crystallize.py — v2 晶化 (crystallization) tool.

Rohit G00 v2 范式：从散落各页的"重复断言"中蒸馏出稳定的晶化条目。

工作流程：
  1. 扫描 docs/wiki/ 下所有页面，提取 `<!-- claim: <text> | sources: src_xxx, src_yyy -->` 注释
  2. 按 (normalized text) 聚合
  3. 出现 ≥min_occurrences 次的 claim 视为"晶化候选"
  4. 写入 manifests/crystallized.csv
  5. 在 docs/wiki/crystallized-claims.md 渲染晶化表

Usage:
  python3 scripts/crystallize.py --min-occurrences 2
  python3 scripts/crystallize.py --min-occurrences 1 --dry-run
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"
MANIFEST_DIR = ROOT / "manifests"
CRYSTALLIZED_CSV = MANIFEST_DIR / "crystallized.csv"
CRYSTALLIZED_MD = WIKI_ROOT / "crystallized-claims.md"

CLAIM_RE = re.compile(r"<!--\s*claim:\s*(.+?)\s*\|\s*sources:\s*(.+?)\s*-->", re.IGNORECASE)


def normalize_claim(text: str) -> str:
    """Normalize claim text for grouping: lowercase, strip whitespace, collapse spaces."""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_claims_from_page(path: Path) -> list[tuple[str, list[str]]]:
    """Extract (claim_text, [source_ids]) tuples from a page's HTML comments."""
    text = path.read_text(encoding="utf-8")
    out: list[tuple[str, list[str]]] = []
    for match in CLAIM_RE.finditer(text):
        claim_text = match.group(1).strip()
        sources_raw = match.group(2).strip()
        sources = [s.strip() for s in re.split(r"[,;]\s*", sources_raw) if s.strip()]
        out.append((claim_text, sources))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 crystallization tool")
    parser.add_argument("--wiki-root", type=Path, default=WIKI_ROOT)
    parser.add_argument("--min-occurrences", type=int, default=2,
                        help="minimum times a claim must appear to be crystallized (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="don't write files, just report")
    parser.add_argument("--output-csv", type=Path, default=CRYSTALLIZED_CSV)
    parser.add_argument("--output-md", type=Path, default=CRYSTALLIZED_MD)
    args = parser.parse_args()

    if not args.wiki_root.exists():
        print(f"crystallize: {args.wiki_root} does not exist")
        return 1

    md_files = sorted(args.wiki_root.rglob("*.md"))
    # claim_text_normalized -> {claim_text_canonical, sources_set, pages}
    claim_index: dict[str, dict] = defaultdict(lambda: {
        "text": "",
        "sources": set(),
        "pages": set(),
    })

    for path in md_files:
        rel = path.relative_to(args.wiki_root).as_posix()
        for text, sources in extract_claims_from_page(path):
            key = normalize_claim(text)
            claim_index[key]["text"] = text  # use first occurrence as canonical
            claim_index[key]["sources"].update(sources)
            claim_index[key]["pages"].add(rel)

    # Filter by min_occurrences
    candidates = {
        k: v for k, v in claim_index.items() if len(v["pages"]) >= args.min_occurrences
    }

    # Stable sort: by occurrence count desc, then by text
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda kv: (-len(kv[1]["pages"]), kv[1]["text"]),
    )

    if not sorted_candidates:
        print("crystallize: no claims found with --min-occurrences={}".format(args.min_occurrences))
        print("- hint: add <!-- claim: <text> | sources: src_xxx --> comments to wiki pages")
        return 0

    today = date.today().isoformat()

    if not args.dry_run:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["claim_id", "text", "sources", "first_seen", "last_validated", "occurrences", "pages"])
            for i, (key, info) in enumerate(sorted_candidates, start=1):
                claim_id = f"clm_{i:03d}"
                writer.writerow([
                    claim_id,
                    info["text"],
                    ";".join(sorted(info["sources"])),
                    today,
                    today,
                    len(info["pages"]),
                    ";".join(sorted(info["pages"])),
                ])

        # Render MD
        md_lines: list[str] = ["---", "title: 晶化断言", "source: crystallize.py",
                               f"created: {today}", f"updated: {today}", f"last_confirmed: {today}",
                               "audience: self", "stage: crystallized", "status: current",
                               "tags: [crystallized, claims, v2]", "---", "",
                               "# 晶化断言 (Crystallized Claims)", "",
                               f"生成时间: {today} | 最小出现次数: {args.min_occurrences} | 总数: {len(sorted_candidates)}",
                               "",
                               "| ID | 断言 | 出现次数 | 来源 | 出现页 |",
                               "|----|------|----------|------|--------|"]
        for i, (key, info) in enumerate(sorted_candidates, start=1):
            claim_id = f"clm_{i:03d}"
            md_lines.append(
                f"| {claim_id} | {info['text']} | {len(info['pages'])} | "
                f"{', '.join(sorted(info['sources']))} | {', '.join(sorted(info['pages']))} |"
            )
        args.output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Report
    print("crystallize:")
    print(f"- scanned pages: {len(md_files)}")
    print(f"- unique claims: {len(claim_index)}")
    print(f"- candidates (occurrences >= {args.min_occurrences}): {len(sorted_candidates)}")
    if not args.dry_run:
        print(f"- wrote: {args.output_csv}")
        print(f"- wrote: {args.output_md}")
    else:
        print("- dry-run: no files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
