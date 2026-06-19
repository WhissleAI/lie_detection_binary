#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Self-contained environment setup for the lie_detection_binary project.
#   - creates a local venv
#   - installs pinned deps (numpy<2 for mediapipe compatibility)
#   - installs the Whissle python client (editable, from the sibling repo)
#   - installs this project (editable)
# Re-runnable / idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
echo "→ Using interpreter: $($PY --version)"

if [ ! -d ".venv" ]; then
  echo "→ Creating venv (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel >/dev/null
echo "→ Installing requirements.txt"
pip install -r requirements.txt

# Whissle client from the sibling repo (kept DRY rather than vendored).
WHISSLE_DIR="${WHISSLE_PYTHON_API_DIR:-../whissle_python_api}"
if [ -d "$WHISSLE_DIR" ]; then
  echo "→ Installing Whissle client (editable) from $WHISSLE_DIR"
  pip install -e "$WHISSLE_DIR" || echo "  (whissle editable install failed — STT step will fall back to raw httpx)"
fi

echo "→ Installing this project (editable)"
pip install -e .

# Verify the audio-visual hybrid-intelligence models are present.
AV_DIR="${AV_UNDERSTANDING_DIR:-../Audio-visual-understanding}"
if [ ! -f "$AV_DIR/models/face_landmarker.task" ]; then
  echo "⚠  MediaPipe models not found under $AV_DIR/models/."
  echo "   Run: (cd \"$AV_DIR\" && ./scripts/download_models.sh)"
fi

echo ""
echo "✅ Setup complete. Activate with:  source .venv/bin/activate"
echo "   Then:  cp .env.example .env  &&  edit .env  (add WHISSLE_API_TOKEN)"
