"""Estimate the wiki's token footprint and tell the user when to consider RAG.

The "wiki before RAG" rule is the project's core claim. This script makes
the threshold quantitative: it counts pages, characters, and a rough token
estimate (chars/4 — conservative for English+CJK mix), then prints buckets.

Buckets (single-session full-read budget):
  GREEN  < 30k tokens   — read everything every session, no problem
  YELLOW 30k–80k        — start being selective; read index + status + log
                          and only pull other pages as needed
  RED    80k–200k       — wiki-first still works but design for partial reads;
                          consider per-domain sub-indices
  PURPLE > 200k         — direct LLM reading hits diminishing returns; build
                          a vector index over docs/wiki/ as a SECONDARY layer
                          (the wiki stays canonical)

Numbers are deliberately conservative — Claude/GPT-4 class models can fit
much more, but session-start full-reads waste cache and money long before
the hard context limit.
"""
from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: ci-safe (reads wiki only)

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "docs" / "wiki"


GREEN = 30_000
YELLOW = 80_000
RED = 200_000


def estimate_tokens(text: str) -> int:
    """Conservative chars/4. Tokens for mixed CN/EN markdown average closer
    to chars/3 — chars/4 underestimates slightly, which is the safe direction
    for "you have headroom" claims."""
    return max(1, len(text) // 4)


def bucket(total_tokens: int) -> tuple[str, str]:
    if total_tokens < GREEN:
        return "GREEN", "Read the whole wiki every session — no concern."
    if total_tokens < YELLOW:
        return "YELLOW", "Start being selective: index + status + log on session start, pull other pages on demand."
    if total_tokens < RED:
        return "RED", "Stay wiki-first but plan for partial reads. Consider per-domain sub-indices and tighter page scoping."
    return "PURPLE", "Direct LLM reading is hitting diminishing returns. Build a vector index over docs/wiki/ as a secondary lookup layer; the wiki stays canonical."


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate wiki size and recommend a reading strategy.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output")
    parser.add_argument("--top", type=int, default=10, help="Show the N largest pages")
    args = parser.parse_args()

    if not WIKI_ROOT.exists():
        print("wiki_size_report: docs/wiki does not exist", file=sys.stderr)
        return 1

    pages: list[tuple[Path, int, int]] = []
    total_chars = 0
    total_tokens = 0
    for path in sorted(WIKI_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        chars = len(text)
        tokens = estimate_tokens(text)
        pages.append((path, chars, tokens))
        total_chars += chars
        total_tokens += tokens

    label, advice = bucket(total_tokens)

    if args.json:
        import json
        out = {
            "page_count": len(pages),
            "total_chars": total_chars,
            "estimated_tokens": total_tokens,
            "bucket": label,
            "advice": advice,
            "thresholds": {"green": GREEN, "yellow": YELLOW, "red": RED},
            "largest": [
                {"path": str(p.relative_to(ROOT)), "chars": c, "tokens": t}
                for p, c, t in sorted(pages, key=lambda x: -x[2])[: args.top]
            ],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"wiki_size_report: {label}")
    print(f"- pages: {len(pages)}")
    print(f"- total chars: {total_chars:,}")
    print(f"- estimated tokens: ~{total_tokens:,}  (chars/4)")
    print(f"- thresholds: green<{GREEN:,}, yellow<{YELLOW:,}, red<{RED:,}")
    print(f"- advice: {advice}")
    if pages:
        print(f"\nLargest pages (top {min(args.top, len(pages))}):")
        for p, c, t in sorted(pages, key=lambda x: -x[2])[: args.top]:
            print(f"  {t:>7,} tok  {c:>8,} ch  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
