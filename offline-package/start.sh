#!/usr/bin/env bash
# One-step launcher for Linux / macOS / Chromebook (Linux/crostini).
# Downloads the models the first time (needs internet), then serves the app.
set -e
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "Python 3 is required but was not found. Install Python 3 and try again." >&2
  echo "On a Chromebook: enable Linux (Settings > Advanced > Developers), then 'sudo apt install python3'." >&2
  exit 1
fi

# If no models are present yet, download them once.
if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null | grep -v '^README' || true)" ]; then
  echo "No models found yet — downloading them once (needs internet)…"
  "$PY" download-models.py
fi

exec "$PY" serve.py
