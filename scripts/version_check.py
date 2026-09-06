from __future__ import annotations
# llm-wiki-version: 2.0.0
# runtime: dev-only (hits GitHub API)

import json
import re
import sys
import urllib.request
from pathlib import Path

GITHUB_API = "https://api.github.com/repos/Ss1024sS/LLM-wiki/releases/latest"
VERSION_RE = re.compile(r"# llm-wiki-version:\s*(\S+)")
SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts[:3]) if parts else (0,)


def get_local_version() -> str:
    for script in ["wiki_check.py", "raw_manifest_check.py", "provenance_check.py"]:
        path = SCRIPTS_DIR / script
        if path.exists():
            m = VERSION_RE.search(path.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    return "unknown"


def get_remote_version() -> tuple[str, str]:
    try:
        req = urllib.request.Request(GITHUB_API, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            tag = data.get("tag_name", "").lstrip("v")
            url = data.get("html_url", "")
            return tag, url
    except Exception:
        return "", ""


def main() -> int:
    local = get_local_version()
    remote, release_url = get_remote_version()
    if not remote:
        return 0
    if local == "unknown":
        print(f"[llm-wiki] Could not detect local version. Latest is v{remote}")
        return 0
    if parse_version(remote) > parse_version(local):
        print(f"[llm-wiki] Update available: v{local} -> v{remote}")
        print(f"[llm-wiki] Run: bash scripts/upgrade.sh")
        if release_url:
            print(f"[llm-wiki] Release notes: {release_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
