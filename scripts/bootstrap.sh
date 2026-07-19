#!/usr/bin/env bash
# One-step setup for a fresh clone: virtualenv, pinned deps, binary checks, and a
# model prefetch so the first pipeline run does not stall on a download. Safe to
# re-run. For Make users, `make bootstrap` covers the first three steps.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv
PY="$VENV/bin/python"

echo "==> virtualenv"
[ -d "$VENV" ] || python3 -m venv "$VENV"

echo "==> dependencies"
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -r requirements.txt

echo "==> system binaries"
command -v tesseract >/dev/null 2>&1 \
  || echo "   WARN: tesseract missing (OCR).  macOS: brew install tesseract"
command -v ffmpeg >/dev/null 2>&1 \
  || echo "   WARN: ffmpeg missing (audio).    macOS: brew install ffmpeg"

echo "==> prefetch face models (downloads InsightFace buffalo_l once)"
"$PY" - <<'PYEOF' || echo "   (prefetch skipped; will download on first run)"
import detect_faces
detect_faces.get_app()
print("   face models ready")
PYEOF

echo "==> smoke test"
"$PY" -m pytest -q -k "integration" >/dev/null && echo "   integration tests pass"

echo "bootstrap complete. Next: '$PY run_pipeline.py' (or 'make pipeline')."
