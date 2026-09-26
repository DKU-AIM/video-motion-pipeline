#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MOTION_WORKSPACE="${MOTION_WORKSPACE:-$HOME/aim-workspace}"
PY="${GROUNDING_PYTHON:-$MOTION_WORKSPACE/grounding_env/bin/python}"
export PYTHONUNBUFFERED=1
exec "$PY" "$ROOT/grounding/local_grounding_experiment.py" "$@"
