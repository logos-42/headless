#!/usr/bin/env bash
# llm-wiki-version: 1.2.2
# Upgrade LLM-wiki scripts to latest version.
# Updates validation scripts and CI only. Never touches wiki content.
set -euo pipefail
REPO="https://github.com/Ss1024sS/LLM-wiki.git"
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
echo "Fetching latest LLM-wiki..."
git clone --depth 1 "$REPO" "$TMP/repo" 2>/dev/null
UPGRADE="$TMP/repo/scripts/upgrade_knowledge_system.py"
if [ -f "$UPGRADE" ]; then
  python3 "$UPGRADE" "$(pwd)"
else
  echo "Error: upgrade script not found in latest LLM-wiki"
  exit 1
fi
