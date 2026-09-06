"""Bootstrap a compile-first knowledge system into a target repo.

The actual file contents live as real files under
`skills/knowledge-system-bootstrap/templates/`. This script copies them
into the target with sentinel-string substitution. That keeps every
generated script lintable, testable, and reviewable on its own.

Substitution sentinels (plain string.replace, no Template machinery so
templates can contain literal `$`):
  __PROJECT_NAME__    — human-readable project name passed on the CLI
  __RAW_ROOT_NAME__   — folder name for the local raw root
  __TODAY__           — today's date in YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import date, datetime
from pathlib import Path

__version__ = "2.0.0"  # bumped for v2 scripts (wiki_lint, supersede_check, crystallize, hybrid_search, graph_export, intake_filter) merged under SKILL_ROOT/scripts/

SKILL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = SKILL_ROOT / "templates"


# (template-relative path, target-relative path)
TEMPLATE_TO_TARGET: list[tuple[str, str]] = [
    ("configs/AGENTS.md", "AGENTS.md"),
    ("configs/CLAUDE.md", "CLAUDE.md"),
    ("configs/.cursorrules", ".cursorrules"),
    ("configs/.windsurfrules", ".windsurfrules"),
    ("configs/.gitignore", ".gitignore"),
    ("wiki/README.md", "docs/wiki/README.md"),
    ("wiki/SCHEMA.md", "docs/wiki/SCHEMA.md"),
    ("wiki/index.md", "docs/wiki/index.md"),
    ("wiki/log.md", "docs/wiki/log.md"),
    ("wiki/project-overview.md", "docs/wiki/project-overview.md"),
    ("wiki/current-status.md", "docs/wiki/current-status.md"),
    ("wiki/sources-and-data.md", "docs/wiki/sources-and-data.md"),
    ("wiki/github-and-raw-strategy.md", "docs/wiki/github-and-raw-strategy.md"),
    ("wiki/runtime-profile.md", "docs/wiki/runtime-profile.md"),
    ("manifests/README.md", "manifests/README.md"),
    ("manifests/raw_sources.csv", "manifests/raw_sources.csv"),
    ("manifests/raw_sources.meta.json", "manifests/raw_sources.meta.json"),
    ("scripts/wiki_check.py", "scripts/wiki_check.py"),
    ("scripts/raw_manifest_check.py", "scripts/raw_manifest_check.py"),
    ("scripts/ingest_raw.py", "scripts/ingest_raw.py"),
    ("scripts/untracked_raw_check.py", "scripts/untracked_raw_check.py"),
    ("scripts/provenance_check.py", "scripts/provenance_check.py"),
    ("scripts/stale_report.py", "scripts/stale_report.py"),
    ("scripts/delta_compile.py", "scripts/delta_compile.py"),
    ("scripts/version_check.py", "scripts/version_check.py"),
    ("scripts/wiki_size_report.py", "scripts/wiki_size_report.py"),
    ("scripts/init_raw_root.py", "scripts/init_raw_root.py"),
    ("scripts/export_memory_repo.py", "scripts/export_memory_repo.py"),
    ("scripts/upgrade.sh", "scripts/upgrade.sh"),
    # v2.0.0 — new scripts merged under scripts/, no longer in templates/scripts/
    ("scripts/wiki_lint.py", "scripts/wiki_lint.py"),
    ("scripts/supersede_check.py", "scripts/supersede_check.py"),
    ("scripts/crystallize.py", "scripts/crystallize.py"),
    ("scripts/hybrid_search.py", "scripts/hybrid_search.py"),
    ("scripts/graph_export.py", "scripts/graph_export.py"),
    ("scripts/intake_filter.py", "scripts/intake_filter.py"),
    ("claude-commands/wiki-check.md", ".claude/commands/wiki-check.md"),
    ("claude-commands/wiki-upgrade.md", ".claude/commands/wiki-upgrade.md"),
    ("claude-commands/wiki-status.md", ".claude/commands/wiki-status.md"),
    ("github/workflows/wiki-lint.yml", ".github/workflows/wiki-lint.yml"),
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "project"


def render(body: str, vars: dict[str, str]) -> str:
    out = body
    for sentinel, value in vars.items():
        out = out.replace(sentinel, value)
    unresolved = re.findall(r"__[A-Z][A-Z0-9_]*__", out)
    expected = {"__main__", "__init__", "__name__", "__file__"}
    real = [s for s in unresolved if s not in expected]
    if real:
        raise RuntimeError(f"unresolved sentinels: {set(real)}")
    return out


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a compile-first knowledge system into another repo.")
    parser.add_argument("target_dir", help="Target repository root")
    parser.add_argument("project_name", help="Human-readable project name")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files (a .bak.<timestamp> copy is kept unless --no-backup)")
    parser.add_argument("--no-backup", action="store_true", help="When overwriting, skip the .bak file (only effective with --force)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing anything")
    parser.add_argument("--raw-root-name", help="Folder name for the local raw root")
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    project_name = args.project_name.strip()
    slug = slugify(project_name)
    raw_root_name = args.raw_root_name or f"{slug}_raw"
    today = date.today().isoformat()

    vars = {
        "__PROJECT_NAME__": project_name,
        "__RAW_ROOT_NAME__": raw_root_name,
        "__TODAY__": today,
    }

    created: list[Path] = []
    overwritten: list[tuple[Path, Path | None]] = []
    skipped: list[Path] = []
    unchanged: list[Path] = []

    for tmpl_rel, target_rel in TEMPLATE_TO_TARGET:
        # v2.0.0: scripts/ under SKILL_ROOT/scripts/ (not templates/scripts/)
        if tmpl_rel.startswith("scripts/"):
            src = SKILL_ROOT / tmpl_rel
        else:
            src = TEMPLATES / tmpl_rel
        if not src.exists():
            print(f"WARNING: template missing: {tmpl_rel}")
            continue
        body = src.read_text(encoding="utf-8")
        try:
            content = render(body, vars)
        except RuntimeError as exc:
            raise SystemExit(f"template {tmpl_rel}: {exc}")

        dest = target / target_rel
        if not dest.exists():
            if not args.dry_run:
                write(dest, content)
            created.append(dest)
        elif dest.read_text(encoding="utf-8") == content:
            unchanged.append(dest)
        elif args.force:
            bak_path: Path | None = None
            if not args.dry_run:
                if not args.no_backup:
                    bak_path = backup(dest)
                write(dest, content)
            overwritten.append((dest, bak_path))
        else:
            skipped.append(dest)

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"{prefix}Bootstrapped knowledge system for: {project_name}")
    print(f"Target repo: {target}")
    print(f"Default raw root name: {raw_root_name}")

    if created:
        print(f"\nCreated ({len(created)}):")
        for p in created:
            print(f"  + {p.relative_to(target)}")
    if overwritten:
        print(f"\nOverwritten ({len(overwritten)}):")
        for p, bak in overwritten:
            suffix = f"  (backup: {bak.name})" if bak else "  (no backup)"
            print(f"  ! {p.relative_to(target)}{suffix}")
    if unchanged:
        print(f"\nUnchanged ({len(unchanged)})")
    if skipped:
        print(f"\nSkipped — already exists, content differs ({len(skipped)}):")
        print("Use --force to overwrite.")
        for p in skipped:
            print(f"  ~ {p.relative_to(target)}")

    if not args.dry_run and (created or overwritten):
        print("\nNext steps:")
        print(f"1. cd {target}")
        print("2. python3 scripts/init_raw_root.py")
        print("3. python3 scripts/wiki_check.py")
        print("4. python3 scripts/raw_manifest_check.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
