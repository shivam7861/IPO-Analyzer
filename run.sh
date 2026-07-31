#!/usr/bin/env bash
# run.sh — Start IPO Lens locally
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "ipo_venv" ]; then
  source ipo_venv/bin/activate
fi

# Create uploads dir if missing
mkdir -p uploads

echo ""
echo "  ⚡ IPO Lens — AI-Powered RHP Analyzer"
echo "  ──────────────────────────────────────"
echo "  Open in browser: http://localhost:8000"
echo ""

# Launch FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
