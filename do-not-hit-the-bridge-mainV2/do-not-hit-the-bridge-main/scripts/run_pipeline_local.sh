#!/usr/bin/env bash
# Local cron replacement for .github/workflows/mirror.yml
set -euo pipefail

# --- config ---
REPO_DIR="${REPO_DIR:-$HOME/do-not-hit-the-bridge}"   # change if your clone lives elsewhere
BRANCH="${BRANCH:-main}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date -u +%Y%m%d).log"

exec >>"$LOG_FILE" 2>&1

echo "=============================================="
echo "Pipeline start: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
echo "Repo: $REPO_DIR"
echo "=============================================="

cd "$REPO_DIR"

# Optional: use a venv if you have one
if [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_DIR/.venv/bin/activate"
  PYTHON=python
fi

# Ensure dirs exist
mkdir -p data/raw data/processed

# Pull latest (avoids push rejection if you also use the Action sometimes)
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH" || {
  echo "WARN: git pull failed – continuing with local tree"
}

# Full pipeline (same as the Action)
$PYTHON scripts/fetch_all.py
$PYTHON scripts/train_xgboost.py
$PYTHON scripts/generate_forecast.py

# Commit + push data only
git config user.name "local-cron"
git config user.email "local-cron@localhost"

git add data/

if git diff --staged --quiet; then
  echo "No data changes to commit."
else
  git commit -m "Auto-update: $(date -u '+%Y-%m-%d %H:%M') UTC"
  git push origin "$BRANCH"
  echo "Pushed data update."
fi

echo "Pipeline end: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
echo
