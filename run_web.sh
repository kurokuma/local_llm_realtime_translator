#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo ".venv was not found. Run: python -m venv .venv" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" web_translator.py "$@"
