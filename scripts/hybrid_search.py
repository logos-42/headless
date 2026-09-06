from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only

"""
hybrid_search.py — v2 混合检索 (BM25 + 图遍历).

Rohit G00 v2 范式：BM25 (keyword) + 向量 (semantic) + 图遍历 (structural).
v2.0 实现: BM25 over wiki 页面 + 图遍历 (follows supersedes/contradicts links).
v2.1+ 可加 embedding-based vector search.

Usage:
  python3 scripts/hybrid_search.py "货币主权"
  python3 scripts/hybrid_search.py "DIAP" --depth 2
  python3 scripts/hybrid_search.py "价值函数" --json --top 5
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)  # 英文/数字/汉字

# BM25 parameters
K1 = 1.5
B = 0.75


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


def tokenize(text: str) -> list[str]:
    """CJK + ASCII tokenization. Each CJK char is its own token, ASCII words grouped."""
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(text):
        s = m.group(0)
        if re.search(r"[一-鿿]", s):
            tokens.extend(s)  # CJK: one char per token
        else:
            tokens.append(s.lower())
    return tokens


def bm25_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float, df: dict[str, int], n_docs: int) -> float:
    """Compute BM25 score for a single document."""
    score = 0.0
    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)
    for q in query_tokens:
        if q not in tf:
            continue
        idf = math.log((n_docs - df[q] + 0.5) / (df[q] + 0.5) + 1)
        denom = tf[q] + K1 * (1 - B + B * doc_len / avg_dl)
        score += idf * (tf[q] * (K1 + 1)) / denom
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 hybrid search (BM25 + graph)")
    parser.add_argument("query", help="search query")
    parser.add_argument("--wiki-root", type=Path, default=WIKI_ROOT)
    parser.add_argument("--depth", type=int, default=1, help="graph traversal depth (0=no graph)")
    parser.add_argument("--top", type=int, default=5, help="top N results")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    args = parser.parse_args()

    if not args.wiki_root.exists():
        print(f"hybrid_search: {args.wiki_root} does not exist")
        return 1

    md_files = sorted(args.wiki_root.rglob("*.md"))
    docs: list[dict] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        # Strip frontmatter for tokenization
        body = FRONTMATTER_RE.sub("", text, count=1)
        tokens = tokenize(body)
        fm = parse_frontmatter(text)
        rel = path.relative_to(args.wiki_root).as_posix()
        docs.append({"rel": rel, "tokens": tokens, "fm": fm, "path": path})

    if not docs:
        print("hybrid_search: no wiki pages found")
        return 1

    # Build DF
    df: dict[str, int] = defaultdict(int)
    for d in docs:
        for term in set(d["tokens"]):
            df[term] += 1

    avg_dl = sum(len(d["tokens"]) for d in docs) / len(docs)
    query_tokens = tokenize(args.query)

    # BM25 scores
    bm25_hits: list[tuple[float, dict]] = []
    for d in docs:
        score = bm25_score(query_tokens, d["tokens"], avg_dl, df, len(docs))
        if score > 0:
            bm25_hits.append((score, d))
    bm25_hits.sort(key=lambda x: -x[0])

    # Build graph from supersedes + contradicts
    graph: dict[str, set[str]] = defaultdict(set)
    for d in docs:
        rel = d["rel"]
        for edge_field in ("supersedes", "contradicts"):
            raw = d["fm"].get(edge_field, "")
            if not raw:
                continue
            for t in re.findall(r"\./[^\s,\]]+", raw):
                graph[rel].add(t.lstrip("./"))

    # Graph traversal: for each top BM25 hit, expand neighbors up to depth
    bm25_paths = {d["rel"]: score for score, d in bm25_hits[:args.top]}
    expanded_paths: dict[str, list[str]] = {}

    if args.depth > 0:
        # BFS from top hits
        visited: set[str] = set()
        queue: list[tuple[str, int, list[str]]] = []
        for rel in bm25_paths:
            queue.append((rel, 0, [rel]))
            visited.add(rel)
        while queue:
            node, d, path = queue.pop(0)
            if d >= args.depth:
                continue
            for nxt in graph.get(node, set()):
                if nxt not in visited and nxt in {d2["rel"] for d2 in docs}:
                    visited.add(nxt)
                    new_path = path + [nxt]
                    queue.append((nxt, d + 1, new_path))
                    if d + 1 == args.depth:
                        # Add as expansion
                        expanded_paths.setdefault(nxt, []).extend(new_path)
        # Flatten
        graph_hits: list[tuple[str, int, list[str]]] = []
        for nxt, paths in expanded_paths.items():
            for p in paths:
                graph_hits.append((0.5 / (len(p)), d, p))
        graph_hits.sort(key=lambda x: -x[0])
    else:
        graph_hits = []

    # Reciprocal Rank Fusion (RRF)
    rrf_scores: dict[str, float] = defaultdict(float)
    for rank, (score, d) in enumerate(bm25_hits, start=1):
        rrf_scores[d["rel"]] += 1.0 / (60 + rank)
    for rank, (score, _, path) in enumerate(graph_hits, start=1):
        rrf_scores[path[-1]] += 0.5 / (60 + rank)

    # Final ranking
    final_ranked = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:args.top]

    # Build snippets
    results: list[dict] = []
    for rel, rrf in final_ranked:
        doc = next(d for d in docs if d["rel"] == rel)
        body = FRONTMATTER_RE.sub("", doc["path"].read_text(encoding="utf-8"), count=1)
        # Find first matching line
        snippet = ""
        for line in body.splitlines():
            if any(q.lower() in line.lower() for q in [args.query]):
                snippet = line.strip()[:200]
                break
        if not snippet:
            snippet = body.splitlines()[0][:200] if body.splitlines() else ""
        results.append({
            "rel": rel,
            "rrf_score": round(rrf, 5),
            "title": doc["fm"].get("title", ""),
            "stage": doc["fm"].get("stage", ""),
            "snippet": snippet,
        })

    report = {
        "query": args.query,
        "query_tokens": query_tokens,
        "wiki_pages_scanned": len(docs),
        "bm25_hits": len(bm25_hits),
        "graph_depth": args.depth,
        "results": results,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"hybrid_search: query={args.query!r} (tokens: {query_tokens[:10]}{'...' if len(query_tokens) > 10 else ''})")
        print(f"- scanned: {len(docs)} pages")
        print(f"- BM25 hits: {len(bm25_hits)}")
        print(f"- graph depth: {args.depth}")
        print(f"- top {args.top} results (RRF):")
        for r in results:
            print(f"  {r['rrf_score']:.4f}  {r['rel']}  [{r['stage']}]  {r['title']}")
            if r['snippet']:
                print(f"           {r['snippet'][:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
