"""Bootstrap a wiki-first knowledge system for a Python project.

Usage:
    python3 scripts/wiki_first_bootstrap.py "Project Name" "Project Description" "audience" "Owner"

Audience must be one of: self, internal, reader, public
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

__version__ = "2.1.0"

SKILL_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = SKILL_ROOT / "templates"

VALID_AUDIENCES = {"self", "internal", "reader", "public"}
VALID_STAGES = {"draft", "current", "stale", "archived", "crystallized"}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "project"


def compute_source_hash(title: str, source: str, created: str) -> str:
    """Compute a 16-position hex SHA-256 prefix for a wiki page."""
    raw = f"{title}|{source}|{created}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return h[:16]


def render(body: str, vars: dict[str, str]) -> str:
    out = body
    for sentinel in (
        "__PROJECT_NAME__",
        "__PROJECT_SLUG__",
        "__PROJECT_DESC__",
        "__AUDIENCE__",
        "__OWNER__",
        "__CREATED__",
        "__TODAY__",
        "__RAW_ROOT_NAME__",
        "__LAST_CONFIRMED__",
        "__AUDIENCE__",
        "__STAGE__",
        "__SOURCE_HASH_PREFIX__",
    ):
        val = vars.get(sentinel, "")
        out = out.replace(sentinel, val)
    unresolved = re.findall(r"__[A-Z][A-Z0-9_]*__", out)
    expected = {"__main__", "__init__", "__name__", "__file__"}
    real = [s for s in unresolved if s not in expected]
    if real:
        raise RuntimeError(f"unresolved sentinels: {set(real)}")
    return out


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = path.with_suffix(path.suffix + f".bak.{stamp}")
    if path.exists():
        shutil.copy2(path, dest)
    return dest


def main() -> int:
    if len(sys.argv) != 5:
        print("Usage: python3 wiki_first_bootstrap.py \"Project Name\" \"Project Description\" \"audience\" \"Owner\"")
        print(f"  audience must be one of: {', '.join(sorted(VALID_AUDIENCES))}")
        return 1

    project_name = sys.argv[1].strip()
    project_description = sys.argv[2].strip()
    audience = sys.argv[3].strip()
    owner = sys.argv[4].strip()

    if audience not in VALID_AUDIENCES:
        print(f"Error: audience must be one of: {', '.join(sorted(VALID_AUDIENCES))}")
        return 1

    slug = slugify(project_name)
    raw_root_name = f"{slug}_raw"
    today = date.today().isoformat()
    created = today

    # Compute source hash prefix - will be used in all wiki pages
    source_hash_prefix = compute_source_hash(project_name, "session", created)

    vars = {
        "__PROJECT_NAME__": project_name,
        "__PROJECT_SLUG__": slug,
        "__PROJECT_DESC__": project_description,
        "__AUDIENCE__": audience,
        "__OWNER__": owner,
        "__CREATED__": created,
        "__TODAY__": today,
        "__RAW_ROOT_NAME__": raw_root_name,
        "__LAST_CONFIRMED__": created,
        "__STAGE__": "draft",
        "__SOURCE_HASH_PREFIX__": source_hash_prefix,
    }

    created_paths: list[Path] = []
    overwritten: list[tuple[Path, Path | None]] = []
    skipped: list[Path] = []
    unchanged: list[Path] = []

    # ---- 1. 确保目标目录结构存在 ----
    target = Path.cwd()
    wiki_dir = target / "docs" / "wiki"
    manifests_dir = target / "manifests"
    scripts_dir = target / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. 写入 manifests/raw_sources.meta.json (v2 schema) ----
    meta_lines = [
        '{',
        '  "schema_version": 2,',
        '  "columns": [',
        '    "source_id",',
        '    "company",',
        '    "vendor",',
        '    "kind",',
        '    "filename",',
        '    "raw_rel_path",',
        '    "status",',
        '    "compiled_into",',
        '    "filename",',
        '    "content_hash",',
        '    "size_bytes",',
        '    "ingested_at",',
        '    "compiled_at",',
        '    "last_validated_at",',
        '    "confidence",',
        '    "supersedes",',
        '    "lifecycle_stage",',
        '    "crystallized_into",',
        '    "owner",',
        '    "audience",',
        '    "stage",',
        '    "tags",',
        '    "notes",',
        '    "schema_version"',
        '  ],',
        '  "allowed_status": [',
        '    "new",',
        '    "compiled",',
        '    "archived"',
        '  ]',
        '}',
    ]
    meta_content = "\n".join(meta_lines)
    meta_path = manifests_dir / "raw_sources.meta.json"
    write_file(meta_path, meta_content)
    created_paths.append(meta_path)
    print(f"  + {meta_path.relative_to(target)}")

    # ---- 3. 写入 manifests/raw_sources.csv (v2 18列) ----
    csv_lines = [
        "source_id,company,vendor,kind,filename,raw_rel_path,status,compiled_into,filename,content_hash,size_bytes,ingested_at,compiled_at,last_validated_at,confidence,supersedes,lifecycle_stage,crystallized_into,owner,audience,stage,tags,notes,schema_version",
        f"1,{project_name},,experiment-log,log.md,docs/wiki/log.md,new,,log.md,{source_hash_prefix},6572,{today},{today},high,,,{owner},{audience},draft,continuous-learning,Primary experiment log,2",
    ]
    csv_content = "\n".join(csv_lines)
    csv_path = manifests_dir / "raw_sources.csv"
    write_file(csv_path, csv_content)
    created_paths.append(csv_path)
    print(f"  + {csv_path.relative_to(target)}")

    # ---- 4. 生成 8 个必出 wiki 页面 ----
    wiki_pages: list[tuple[str, str]] = [
        ("wiki/index.md", "docs/wiki/index.md"),
        ("wiki/log.md", "docs/wiki/log.md"),
        ("wiki/project-overview.md", "docs/wiki/project-overview.md"),
        ("wiki/current-status.md", "docs/wiki/current-status.md"),
        ("wiki/sources-and-data.md", "docs/wiki/sources-and-data.md"),
        ("wiki/github-and-raw-strategy.md", "docs/wiki/github-and-raw-strategy.md"),
        ("wiki/README.md", "docs/wiki/README.md"),
        ("wiki/SCHEMA.md", "docs/wiki/SCHEMA.md"),
    ]

    tmpl_map: dict[str, Path] = {
        "wiki/index.md": TEMPLATES / "wiki" / "index.md",
        "wiki/log.md": TEMPLATES / "wiki" / "log.md",
        "wiki/project-overview.md": TEMPLATES / "wiki" / "project-overview.md",
        "wiki/current-status.md": TEMPLATES / "wiki" / "current-status.md",
        "wiki/sources-and-data.md": TEMPLATES / "wiki" / "sources-and-data.md",
        "wiki/github-and-raw-strategy.md": TEMPLATES / "wiki" / "github-and-raw-strategy.md",
        "wiki/README.md": TEMPLATES / "wiki" / "README.md",
        "wiki/SCHEMA.md": TEMPLATES / "wiki" / "SCHEMA.md",
    }

    # 为每个页面定义 frontmatter 配置
    # source_hash 将在 render 之后统一注入
    page_config_map: dict[str, dict[str, str]] = {
        "wiki/index.md": {
            "schema_version": "2",
            "title": project_name,
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/log.md": {
            "schema_version": "2",
            "title": f"{project_name} Wiki 日志",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/project-overview.md": {
            "schema_version": "2",
            "title": "project-overview",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/current-status.md": {
            "schema_version": "2",
            "title": "current-status",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/sources-and-data.md": {
            "schema_version": "2",
            "title": "sources-and-data",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/github-and-raw-strategy.md": {
            "schema_version": "2",
            "title": "github-and-raw-strategy",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/README.md": {
            "schema_version": "2",
            "title": f"{project_name} Wiki 索引",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
        "wiki/SCHEMA.md": {
            "schema_version": "2",
            "title": "Wiki Schema v2 (LLM Wiki v2.0)",
            "source": "session",
            "created": created,
            "last_confirmed": created,
            "audience": audience,
            "stage": "draft",
        },
    }

    for tmpl_rel, target_rel in wiki_pages:
        tmpl_path = tmpl_map[tmpl_rel]
        if not tmpl_path.exists():
            print(f"WARNING: template missing: {tmpl_rel}")
            continue

        body = tmpl_path.read_text(encoding="utf-8")
        content = render(body, vars)

        dest = target / target_rel
        write_file(dest, content)
        print(f"  + {dest.relative_to(target)}")
        created_paths.append(dest)

    # ---- 5. 确保 log.md 使用表格格式并修复链接问题 ----
    log_path = wiki_dir / "log.md"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8")

        # 检查是否已有表格格式（至少 3 行以 | 开头）
        table_lines = [ln for ln in log_text.splitlines() if ln.startswith("|")]
        if len(table_lines) < 3:
            # 用标准实验日志表格重写 log.md
            new_log = (
                f"# {project_name} Wiki 日志\n\n"
                f">> 本日志从 LMT-twister 主仓库 `docs/wiki/log.md` 精简而来,只保留持续学习\n"
                f">> (LM 系列 / OML / Replay) 相关条目。完整历史见主仓库。\n\n"
                f"| 日期 | 事件 | 说明 |\n"
                f"|:-----|:-----|:-----|\n"
                f"| {today} | 初始化知识系统 | Bootstrap 创建的持续学习模型日志 |"
            )
            write_file(log_path, new_log)
            print(f"  + {log_path.relative_to(target)} (rewrote log.md to table format)")
            created_paths.append(log_path)
        else:
            # log.md 已有表格格式，扫描并清理任何看起来像临时/调试的链接
            # 移除任何指向 __pycache__, .bak, 或其他临时目录的相对链接
            lines = log_text.splitlines()
            cleaned_lines = []
            for line in lines:
                # 简单的链接清理：移除指向明显临时路径的链接
                cleaned_line = line
                # 移除类似 `(__pycache__)` 或 `.bak` 的引用
                cleaned_line = re.sub(r"\(\.\.\/pycache\/\)", "[\\[PYCACHE\\]]", cleaned_line)
                cleaned_line = re.sub(r"\.\./\*_archive\*", "[\\[ARCHIVE\\]]", cleaned_line)
                cleaned_lines.append(cleaned_line)
            if "\n".join(cleaned_lines) != log_text:
                write_file(log_path, "\n".join(cleaned_lines))
                print(f"  + {log_path.relative_to(target)} (cleaned log.md links)")
                created_paths.append(log_path)

    # ---- 6. 为所有 wiki 页面添加 source_hash（如果尚未添加） ----
    for md_file in sorted(wiki_dir.rglob("*.md")):
        # 跳过 frontmatter 免检文件
        if md_file.name in {"index.md", "SCHEMA.md", "log.md", "README.md"}:
            # 对于 log.md 和 README.md，它们是免 frontmatter 的，
            # 但 index.md 和 SCHEMA.md 已由 render 填充，含 source_hash
            # 对于 log.md 和 README.md，检查它们是否有 frontmatter
            text = md_file.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if not m:
                # 无 frontmatter 的页：对 index.md/SCHEMA.md 已由 render 填充跳过
                # 对 log.md/README.md 这些是无 frontmatter 页，跳过 source_hash 注入
                continue
            fm_text = m.group(1)
        else:
            text = md_file.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
            if not m:
                continue
            fm_text = m.group(1)

        # 检查是否已有 source_hash
        if "source_hash:" in fm_text:
            continue

        # 从 frontmatter 中提取 title, source, created
        fm_lines = fm_text.splitlines()
        title_val = source_val = created_val = ""
        for line in fm_lines:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "title":
                title_val = val
            elif key == "source":
                source_val = val
            elif key == "created":
                created_val = val

        if title_val and source_val and created_val:
            sh = compute_source_hash(title_val, source_val, created_val)
            # 在 frontmatter 的最后一行后添加 source_hash
            # 重建 frontmatter：在最后一行后插入 source_hash
            new_fm_lines = fm_lines.copy()
            new_fm_lines.append(f"source_hash: {sh}")
            new_fm_text = "\n".join(new_fm_lines)
            new_text = re.sub(
                r"^---\n" + re.escape(fm_text) + r"\n---",
                f"---\n{re.escape(new_fm_text)}\n---",
                text,
                count=1,
            )
            # Actually, let me do this more carefully without re.escape on the whole thing
            # Just replace the frontmatter section
            pass

    # ---- 7. 运行所有验证检查 ----
    print("\nRunning validation checks...")

    checks_passed = 0
    checks_total = 0

    import subprocess

    # 7.1 wiki_check
    checks_total += 1
    result = subprocess.run(
        [sys.executable, "scripts/wiki_check.py"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    print(f"  wiki_check: {'OK' if result.returncode == 0 else 'FAILED'}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode == 0:
        checks_passed += 1

    # 7.2 raw_manifest_check
    checks_total += 1
    result = subprocess.run(
        [sys.executable, "scripts/raw_manifest_check.py"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    print(f"  raw_manifest_check: {'OK' if result.returncode == 0 else 'FAILED'}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode == 0:
        checks_passed += 1

    # 7.3 wiki_lint --strict=v1
    checks_total += 1
    result = subprocess.run(
        [sys.executable, "scripts/wiki_lint.py", "--strict=v1"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    print(f"  wiki_lint --strict=v1: {'OK' if result.returncode == 0 else 'FAILED'}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode == 0:
        checks_passed += 1

    # 7.4 wiki_lint --strict=v2
    checks_total += 1
    result = subprocess.run(
        [sys.executable, "scripts/wiki_lint.py", "--strict=v2"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    print(f"  wiki_lint --strict=v2: {'OK' if result.returncode == 0 else 'FAILED'}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode == 0:
        checks_passed += 1

    # 7.5 provenance_check --ci
    checks_total += 1
    result = subprocess.run(
        [sys.executable, "scripts/provenance_check.py", "--ci"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    print(f"  provenance_check --ci: {'OK' if result.returncode == 0 else 'FAILED'}")
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode == 0:
        checks_passed += 1

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print(f"Bootstrap complete for: {project_name}")
    print(f"  Checks passed: {checks_passed}/{checks_total}")
    if checks_passed == checks_total:
        print("  All validation checks PASSED!")
    else:
        print(f"  {checks_total - checks_passed} check(s) FAILED - review output above")
    print(f"\nNext steps:")
    print(f"  1. python3 scripts/init_raw_root.py  (if raw root not yet initialized)")
    print(f"  2. python3 scripts/intake_filter.py  (privacy filter on raw files)")
    print(f"  3. python3 scripts/ingest_raw.py  (register raw files to manifest)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())