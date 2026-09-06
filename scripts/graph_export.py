from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only

"""
graph_export.py — v2 知识图谱导出.

Rohit G00 v2 范式：把 wiki 页面 / 概念 / 章节 / 人物 抽取为节点，关系作为类型化边。
v2.0 实现: 节点 = wiki 页面 (含 entity_type) + 概念 (来自 [[双链]] / frontmatter.entity_type)
         边 = supersedes / contradicts / references (来自 frontmatter + 文本 [[链接]])

输出 manifests/knowledge_graph.json:
  {
    "nodes": [{"id": "...", "type": "...", "label": "...", "sources": [...]}],
    "edges": [{"from": "...", "to": "...", "type": "supersedes|contradicts|references"}],
    "stats": {...}
  }

Usage:
  python3 scripts/graph_export.py
  python3 scripts/graph_export.py --output manifests/knowledge_graph.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"
OUTPUT = ROOT / "manifests" / "knowledge_graph.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MDLINK_RE = re.compile(r"\[([^\]]+)\]\(\.\/([^)]+)\)")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 knowledge graph export")
    parser.add_argument("--wiki-root", type=Path, default=WIKI_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not args.wiki_root.exists():
        print(f"graph_export: {args.wiki_root} does not exist")
        return 1

    md_files = sorted(args.wiki_root.rglob("*.md"))
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_set: set[tuple[str, str, str]] = set()  # dedupe (from, to, type)

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        rel = path.relative_to(args.wiki_root).as_posix()

        # Node: this page
        nodes[rel] = {
            "id": rel,
            "type": fm.get("entity_type", "meta"),
            "label": fm.get("title", rel),
            "tags": fm.get("tags", ""),
            "stage": fm.get("stage", ""),
            "audience": fm.get("audience", ""),
            "source_file": str(path.relative_to(ROOT)),
        }

        # Edges from frontmatter
        for edge_type in ("supersedes", "contradicts"):
            raw = fm.get(edge_type, "")
            if not raw:
                continue
            for t in re.findall(r"\./[^\s,\]]+", raw):
                target = t.lstrip("./")
                key = (rel, target, edge_type)
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({"from": rel, "to": target, "type": edge_type})

        # Edges from inline wikilinks [[...]]
        for m in WIKILINK_RE.finditer(text):
            target_label = m.group(1).strip()
            # Try to resolve to a page
            target = target_label if target_label.endswith(".md") else f"{target_label}.md"
            key = (rel, target, "references")
            if key not in edge_set:
                edge_set.add(key)
                edges.append({"from": rel, "to": target, "type": "references"})

        # Edges from markdown links [text](./page.md)
        for m in MDLINK_RE.finditer(text):
            target = m.group(2)
            key = (rel, target, "references")
            if key not in edge_set:
                edge_set.add(key)
                edges.append({"from": rel, "to": target, "type": "references"})

    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": dict(Counter(n["type"] for n in nodes.values())),
        "edge_types": dict(Counter(e["type"] for e in edges)),
    }

    graph = {
        "schema_version": 2,
        "wiki_root": str(args.wiki_root),
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": stats,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"graph_export:")
        print(f"- nodes: {stats['node_count']}")
        print(f"- edges: {stats['edge_count']}")
        print(f"- node_types: {stats['node_types']}")
        print(f"- edge_types: {stats['edge_types']}")
        print(f"- wrote: {args.output}")
    else:
        print(json.dumps(graph, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
