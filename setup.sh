#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Self-contained environment setup for the lie_detection_binary project.
#   - creates a local venv
#   - installs pinned deps
#   - installs this project (editable)
#
# STT + visual features come from the Whissle gateway docker (see docs/GATEWAY.md);
# this repo needs no local CV/ASR models — only ffmpeg (for the prosody lane).
# Re-runnable / idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo "→ Using interpreter: $($PY --version)"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠  ffmpeg not found on PATH — the prosody lane (step 03) needs it."
  echo "   Install:  brew install ffmpeg   (macOS)  /  apt install ffmpeg  (Linux)"
fi

if [ ! -d ".venv" ]; then
  echo "→ Creating venv (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel >/dev/null
echo "→ Installing requirements.txt"
pip install -r requirements.txt

echo "→ Installing this project (editable)"
pip install -e .

echo ""
echo "✅ Setup complete. Activate with:  source .venv/bin/activate"
echo "   Then:  cp .env.example .env  &&  edit .env  (add WHISSLE_API_TOKEN)"
echo "   The Whissle gateway must be reachable — see docs/GATEWAY.md."
