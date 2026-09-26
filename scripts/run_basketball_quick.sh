#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/run_basketball_all.sh" --suite quick --timeout-hours 0 --deadline-minutes 0 "$@"
