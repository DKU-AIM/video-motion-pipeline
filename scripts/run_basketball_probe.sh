#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MOTION_WORKSPACE="${MOTION_WORKSPACE:-$HOME/aim-workspace}"
PY="${GROUNDING_PYTHON:-$MOTION_WORKSPACE/grounding_env/bin/python}"
VIDEO="$MOTION_WORKSPACE/inputs/grounding/2026_AsianGames_men's_basketball_final.mp4"
OUT="$MOTION_WORKSPACE/results_grounding/basketball/$(date -u +%Y%m%dT%H%M%S)_$$"
export PYTHONUNBUFFERED=1
exec "$PY" "$ROOT/grounding/basketball_probe.py" --video "$VIDEO" --out "$OUT" "$@"
