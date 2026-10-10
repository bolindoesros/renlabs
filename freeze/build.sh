#!/usr/bin/env bash
# Builds dist/Renlabs.app (macOS). Set PYTHON to use another interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" -m pip install -r freeze/requirements.txt
"$PYTHON" -m freeze.fetch_models
"$PYTHON" -m PyInstaller --noconfirm freeze/renlabs.spec
