from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only

"""
intake_filter.py — v2 intake privacy filter.

Rohit G00 v2 范式：raw 摄入前脱敏，wiki 写入时按 audience 分级。
v2.0 实现：
  1. 读 manifests/redact_rules.yaml（每条 = name + pattern + replacement）
  2. 扫 raw 根目录（默认 ../<project>_raw/，可 --raw-root 覆盖）
  3. 对每个 .md/.txt/.csv/.json 文件应用正则替换
  4. 在 manifests/redaction_log.md 写日志（每文件命中几条）
  5. 退出口为 0（仅警告），CI 默认不阻塞

Usage:
  python3 scripts/intake_filter.py
  python3 scripts/intake_filter.py --dry-run
  python3 scripts/intake_filter.py --raw-root ../my_raw
  python3 scripts/intake_filter.py --rules manifests/redact_rules.yaml
"""

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "manifests" / "redact_rules.yaml"
DEFAULT_LOG = ROOT / "manifests" / "redaction_log.md"

# 默认 deny extensions / max size (写在脚本里，不进 rules.yaml 避免 YAML 解析复杂)
DEFAULT_DENY_EXT = {".zip", ".rar", ".7z", ".tar", ".gz"}
DEFAULT_MAX_SIZE_MB = 200

# 10 行内建 PII 正则（无外部依赖）
BUILTIN_PATTERNS = [
    ("phone_cn", re.compile(r"1[3-9]\d{9}"), "[PHONE]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    ("id_card_cn", re.compile(r"\d{17}[\dXx]"), "[ID]"),
    ("credit_card", re.compile(r"\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}"), "[CC]"),
]


def parse_yaml_rules(path: Path) -> list[tuple[str, re.Pattern, str]]:
    """Minimal YAML parser: each rule is a block of 3 lines (name/pattern/replacement)."""
    if not path.exists():
        return []
    rules: list[tuple[str, re.Pattern, str]] = []
    block: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue
        if line_stripped.startswith("- name:"):
            if block:
                _flush(block, rules)
            block = {"name": line_stripped.split(":", 1)[1].strip()}
        elif line_stripped.startswith("pattern:") and block:
            block["pattern"] = line_stripped.split(":", 1)[1].strip().strip('"').strip("'")
        elif line_stripped.startswith("replacement:") and block:
            block["replacement"] = line_stripped.split(":", 1)[1].strip().strip('"').strip("'")
    if block:
        _flush(block, rules)
    return rules


def _flush(block: dict[str, str], rules: list) -> None:
    name = block.get("name", "unnamed")
    pat = block.get("pattern", "")
    repl = block.get("replacement", "[REDACTED]")
    if not pat:
        return
    try:
        rules.append((name, re.compile(pat), repl))
    except re.error:
        pass  # 跳过非法正则


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 intake privacy filter")
    parser.add_argument("--raw-root", type=Path,
                        default=lambda: Path(os.environ.get("PROJECT_RAW_ROOT", ROOT.parent / "_raw")),
                        help="raw root directory to scan (env PROJECT_RAW_ROOT or --raw-root override; default: ../_raw)")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES,
                        help="redact_rules.yaml path (optional, falls back to builtins)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    if not args.raw_root.exists():
        print(f"intake_filter: raw_root {args.raw_root} does not exist, skipping")
        return 0

    # Load rules: yaml first, then builtins
    rules = parse_yaml_rules(args.rules) or BUILTIN_PATTERNS
    print(f"intake_filter: {len(rules)} patterns loaded ({'yaml' if args.rules.exists() else 'builtin'})")

    # Scan
    hits: list[dict] = []
    scanned = 0
    skipped = 0
    for path in args.raw_root.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix.lower() in DEFAULT_DENY_EXT:
            skipped += 1
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > DEFAULT_MAX_SIZE_MB:
            skipped += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        file_hits: dict[str, int] = {}
        new_text = text
        for name, pat, repl in rules:
            count = len(pat.findall(new_text))
            if count > 0:
                file_hits[name] = count
                new_text = pat.sub(repl, new_text)
        if file_hits:
            if not args.dry_run:
                # 仅当命中数 > 0 时回写
                path.write_text(new_text, encoding="utf-8")
            hits.append({"file": str(path.relative_to(args.raw_root)), "hits": file_hits})

    # Write log
    today = date.today().isoformat()
    log_lines = [f"# Intake Redaction Log", "",
                 f"- date: {today}",
                 f"- raw_root: {args.raw_root}",
                 f"- rules: {args.rules} ({'yaml' if args.rules.exists() else 'builtin'})",
                 f"- dry_run: {args.dry_run}",
                 f"- scanned: {scanned}, skipped: {skipped}, modified: {len(hits)}",
                 ""]
    if hits:
        log_lines.append("## Files modified")
        log_lines.append("")
        for h in hits:
            log_lines.append(f"- `{h['file']}`: " + ", ".join(f"{k}={v}" for k, v in h["hits"].items()))
    else:
        log_lines.append("## No PII / redact patterns matched.")

    if not args.dry_run:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # Report
    print(f"- scanned: {scanned}")
    print(f"- skipped: {skipped} (deny ext or >{DEFAULT_MAX_SIZE_MB}MB)")
    print(f"- modified: {len(hits)}")
    if hits:
        for h in hits[:10]:
            print(f"  - {h['file']}: " + ", ".join(f"{k}={v}" for k, v in h["hits"].items()))
        if len(hits) > 10:
            print(f"  ... and {len(hits) - 10} more")
    if not args.dry_run:
        print(f"- log: {args.log}")
    else:
        print("- dry-run: no files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
