from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only

"""
supersede_check.py — v2 supersede/contradicts chain validator.

Rohit G00 v2 范式：每个 wiki 页可以 supersede 旧页 / contradicts 兄弟页。
这个脚本验证：
  1. supersedes 目标存在
  2. 目标 stage == archived
  3. 没有环
  4. 没有"被 supersede 但未归档"的孤儿
  5. contradicts 对称性（A contradicts B 要求 B 也填 contradicts A）

Usage:
  python3 scripts/supersede_check.py
  python3 scripts/supersede_check.py --json
"""

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def parse_list_field(value: str) -> list[str]:
    """Parse a YAML-ish list: [./a.md, ./b.md] -> [./a.md, ./b.md]"""
    if not value or value in ("[]", ""):
        return []
    return re.findall(r"\./[^\s,\]]+", value)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 supersede / contradicts chain validator")
    parser.add_argument("--wiki-root", type=Path, default=WIKI_ROOT)
    parser.add_argument("--json", action="store_true", help="emit JSON report")
    args = parser.parse_args()

    if not args.wiki_root.exists():
        print(f"supersede_check: {args.wiki_root} does not exist")
        return 1

    md_files = sorted(args.wiki_root.rglob("*.md"))
    pages: dict[str, dict[str, str]] = {}
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if not fm:
            continue
        rel = path.relative_to(args.wiki_root).as_posix()
        pages[rel] = fm

    errors: list[str] = []
    orphans: list[str] = []  # 被 supersede 但未 archived

    # Pass 1: validate each page's supersedes
    for rel, fm in pages.items():
        supersedes = parse_list_field(fm.get("supersedes", ""))
        for target in supersedes:
            target_norm = target.lstrip("./")
            if target_norm not in pages:
                errors.append(f"supersedes target missing: {rel} -> {target}")
                continue
            target_stage = pages[target_norm].get("stage", "")
            if target_stage != "archived":
                errors.append(
                    f"supersedes target not archived: {rel} -> {target} "
                    f"(stage={target_stage!r}, expected 'archived')"
                )

    # Pass 2: detect cycles
    # Build adjacency: page -> set of pages it supersedes
    adj: dict[str, set[str]] = {}
    for rel, fm in pages.items():
        supersedes = parse_list_field(fm.get("supersedes", ""))
        adj[rel] = {t.lstrip("./") for t in supersedes if t.lstrip("./") in pages}

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in adj}
    cycles: list[list[str]] = []

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for nxt in adj.get(node, set()):
            if color[nxt] == GRAY:
                # Found cycle
                if nxt in path:
                    cycle = path[path.index(nxt):] + [nxt]
                    cycles.append(cycle)
            elif color[nxt] == WHITE:
                dfs(nxt, path)
        path.pop()
        color[node] = BLACK

    for node in list(adj.keys()):
        if color[node] == WHITE:
            dfs(node, [])

    for cycle in cycles:
        errors.append(f"supersedes cycle detected: {' -> '.join(cycle)}")

    # Pass 3: detect orphans (被 supersede 但 stage != archived)
    all_superseded_targets: set[str] = set()
    for fm in pages.values():
        supersedes = parse_list_field(fm.get("supersedes", ""))
        for t in supersedes:
            all_superseded_targets.add(t.lstrip("./"))

    for rel in all_superseded_targets:
        if rel in pages and pages[rel].get("stage", "") != "archived":
            orphans.append(f"{rel} (stage={pages[rel].get('stage', '')!r})")

    # Pass 4: contradicts symmetry
    for rel, fm in pages.items():
        contradicts = parse_list_field(fm.get("contradicts", ""))
        for target in contradicts:
            target_norm = target.lstrip("./")
            if target_norm not in pages:
                continue
            target_contradicts = parse_list_field(pages[target_norm].get("contradicts", ""))
            if rel not in [t.lstrip("./") for t in target_contradicts]:
                errors.append(
                    f"contradicts asymmetric: {rel} -> {target} "
                    f"(target {target_norm} does not contradict back)"
                )

    # Report
    report = {
        "wiki_root": str(args.wiki_root),
        "page_count": len(pages),
        "supersedes_total": sum(len(parse_list_field(fm.get("supersedes", ""))) for fm in pages.values()),
        "contradicts_total": sum(len(parse_list_field(fm.get("contradicts", ""))) for fm in pages.values()),
        "orphans": orphans,
        "errors": errors,
        "cycles": cycles,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("supersede_check:")
        print(f"- pages: {report['page_count']}")
        print(f"- supersedes total: {report['supersedes_total']}")
        print(f"- contradicts total: {report['contradicts_total']}")
        if orphans:
            print(f"- orphans (superseded but not archived): {len(orphans)}")
            for o in orphans:
                print(f"  - {o}")
        if errors:
            print(f"- errors: {len(errors)}")
            for e in errors:
                print(f"  - {e}")
            return 1
        if cycles:
            print(f"- cycles: {len(cycles)}")
            return 1
        print("- OK")

    return 0 if not errors and not cycles else 1


if __name__ == "__main__":
    sys.exit(main())
