#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_basketball_all.sh" --suite parameters \
  --models timelens2-4b timelens-8b --timeout-hours 0 --deadline-minutes 0 "$@"
