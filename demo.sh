#!/usr/bin/env bash
# Runs the demo window with the project's own Python, no venv activation needed.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m ren.ui "$@"
