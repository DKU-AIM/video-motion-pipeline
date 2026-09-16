#!/usr/bin/env bash
# Run using the appropriate, already-installed model environment.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$COMMAND" in
  help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/elice.sh <command> [arguments]
  grounding [CLI args]  Run grounding using GROUNDING_PYTHON or python3
  pose                 Run prepared full/crop clips using POSE_PYTHON or python3
  export-motion        Export CoMotion tracks using POSE_PYTHON or python3
  motiongpt            Caption exported motion using CAPTION_PYTHON or python3
  mgllm                Caption MotionGPT-produced features using CAPTION_PYTHON
Set MOTION_WORKSPACE to the experiment directory for all except grounding.
This launcher does not install models or download licensed assets.
HELP
    exit 0;;
  grounding) exec "${GROUNDING_PYTHON:-python3}" "$PROJECT_ROOT/grounding/vtg_run.py" "$@";;
  pose) PYTHON_BIN="${POSE_PYTHON:-python3}"; SCRIPT="mesh_recovery/run_pose_batch.py";;
  export-motion) PYTHON_BIN="${POSE_PYTHON:-python3}"; SCRIPT="annotation/motion_captioning/export_motion_inputs.py";;
  motiongpt) PYTHON_BIN="${CAPTION_PYTHON:-python3}"; SCRIPT="annotation/motion_captioning/run_motiongpt.py";;
  mgllm) PYTHON_BIN="${CAPTION_PYTHON:-python3}"; SCRIPT="annotation/motion_captioning/run_mgllm.py";;
  *) echo "Unknown command: $COMMAND" >&2; exit 2;;
esac
: "${MOTION_WORKSPACE:?Set MOTION_WORKSPACE to your experiment directory}"
[[ -d "$MOTION_WORKSPACE" ]] || { echo "Workspace does not exist: $MOTION_WORKSPACE" >&2; exit 2; }
[[ $# -eq 0 ]] || { echo "This command takes no additional arguments" >&2; exit 2; }
exec "$PYTHON_BIN" "$PROJECT_ROOT/$SCRIPT"
