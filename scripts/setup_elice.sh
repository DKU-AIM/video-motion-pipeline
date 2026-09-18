#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ELICE_SETUP_PYTHON:-python3}" "$PROJECT_ROOT/scripts/setup_elice.py" "$@"
