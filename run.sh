#!/usr/bin/env bash
# RockMap one-command launcher (Linux / macOS)
#   ./run.sh            install (first time), build the demo and open the dashboard
#   ./run.sh --real     also download a real 40 x 40 km Gilgit region (needs internet)
#   ./run.sh --gb       also map the whole of Gilgit-Baltistan at 100 m (needs internet, 30-60 min)
#   ./run.sh serve      just start the dashboard on existing data
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then echo "Python 3.9+ is required (https://www.python.org/downloads/)"; exit 1; fi
if [ ! -d .venv ]; then
  echo ">> Creating virtual environment (.venv)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import rockmap, torch" >/dev/null 2>&1; then
  echo ">> Installing RockMap and dependencies (first run only, a few minutes)"
  python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || python -m pip install torch
  else
    python -m pip install torch
  fi
  python -m pip install -e ".[dev]"
fi
if [ "${1:-}" = "serve" ]; then
  shift
  exec rockmap serve --open "$@"
fi
exec rockmap quickstart "$@"
