#!/usr/bin/env bash
# Run after cloning the repository on an Elice Linux GPU instance.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
skip_video=0
skip_model=0
sample=0
for arg in "$@"; do
  case "$arg" in
    --skip-video) skip_video=1 ;;
    --skip-model) skip_model=1 ;;
    --sample) sample=1 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/bootstrap_elice.sh [--sample | --skip-video] [--skip-model]
Default: install/check grounding environment, download TimeLens-8B and one video.
Video URL: VIDEO_URL environment variable, or hidden terminal prompt.
--sample: use the public TimeLens example instead of a private video.
--skip-video / --skip-model: prepare only the components you need.
Options via environment:
  VTG_ROOT          workspace root (default: ~/vtg-workspace)
  BOOTSTRAP_PYTHON   Python 3.10 binary (default: python3)
  VIDEO_NAME        output filename only (default: input-video.mov)
  VIDEO_SHA256      optional expected SHA-256 from the data provider
  MODEL_REPO        Hugging Face model (default: TencentARC/TimeLens-8B)
  MODEL_REVISION    model commit/tag (default: main; actual snapshot is recorded)
This script does not start inference, back up results, or manage cloud instances.
HELP
      exit 0 ;;
    *) echo "Unknown option. Use --help." >&2; exit 2 ;;
  esac
done
[[ "$sample" -eq 0 || "$skip_video" -eq 0 ]] || { echo "Choose --sample OR --skip-video." >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo "Run this script on the Elice Linux x86_64 GPU server, not your Mac." >&2; exit 1;
}
command -v nvidia-smi >/dev/null || { echo "NVIDIA GPU driver not found." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

VTG_ROOT="${VTG_ROOT:-$HOME/vtg-workspace}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
[[ "$VTG_ROOT" = /* ]] || { echo "VTG_ROOT must be an absolute path." >&2; exit 2; }
env_dir="$VTG_ROOT/env"
if [[ -e "$env_dir" && ! -f "$env_dir/pyvenv.cfg" ]]; then
  echo "Refusing to modify env/: it is not a Python venv. Choose another VTG_ROOT." >&2
  exit 1
fi
if [[ "$skip_video" -eq 0 ]]; then
  if [[ "$sample" -eq 1 ]]; then
    VIDEO_URL="https://huggingface.co/datasets/JungleGym/TimeLens-Assets/resolve/main/2Y8XQ.mp4"
    VIDEO_NAME="timelens-example.mp4"
  elif [[ -z "${VIDEO_URL:-}" ]]; then
    [[ -t 0 ]] || { echo "Set VIDEO_URL, or use --sample / --skip-video." >&2; exit 2; }
    read -r -s -p "Paste video download/share URL (hidden): " VIDEO_URL
    printf '\n'
  fi
  [[ -n "$VIDEO_URL" ]] || { echo "Video URL is empty." >&2; exit 2; }
  VIDEO_NAME="${VIDEO_NAME:-input-video.mov}"
  [[ "$VIDEO_NAME" != */* && "$VIDEO_NAME" != . && "$VIDEO_NAME" != .. ]] || {
    echo "VIDEO_NAME must be a filename, not a path." >&2; exit 2;
  }
  export VIDEO_URL
fi

step="Python environment"
trap 'printf "Setup stopped at: %s. Resolve the error and rerun the same command.\n" "$step" >&2' ERR
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Select Python 3.10 with BOOTSTRAP_PYTHON"'
mkdir -p "$VTG_ROOT/data" "$VTG_ROOT/results" "$VTG_ROOT/cache/huggingface" "$VTG_ROOT/manifests"
df -h "$VTG_ROOT"
if [[ ! -f "$env_dir/pyvenv.cfg" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$env_dir"
fi
py="$env_dir/bin/python"
"$py" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing venv is not Python 3.10; use another VTG_ROOT"'
step="Package installation"
"$py" -m pip install --upgrade pip
"$py" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
"$py" -m pip install -r "$PROJECT_ROOT/grounding/requirements-elice.txt"
# The previous decord wheel failed pip's platform check on the Elice instance.
"$py" -m pip uninstall -y decord
"$py" -m pip check
export FORCE_QWENVL_VIDEO_READER=torchvision
export HF_HOME="$VTG_ROOT/cache/huggingface"
export VTG_ROOT
step="GPU verification"
"$py" "$PROJECT_ROOT/scripts/check_grounding.py"
"$py" -m pip freeze > "$VTG_ROOT/manifests/requirements-resolved.txt"
nvidia-smi > "$VTG_ROOT/manifests/nvidia-smi.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$VTG_ROOT/manifests/code-revision.txt"

if [[ "$skip_model" -eq 0 ]]; then
  step="Model download"
  export MODEL_REPO="${MODEL_REPO:-TencentARC/TimeLens-8B}"
  export MODEL_REVISION="${MODEL_REVISION:-main}"
  "$py" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
repo = os.environ['MODEL_REPO']
snapshot = snapshot_download(repo_id=repo, revision=os.environ['MODEL_REVISION'], allow_patterns=[
    '*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.tiktoken',
])
(Path(os.environ['VTG_ROOT']) / 'manifests/model.json').write_text(
    json.dumps({'repo': repo, 'revision': Path(snapshot).name, 'snapshot': snapshot}, indent=2) + '\n')
print('Model cached:', repo)
PY
fi
if [[ "$skip_video" -eq 0 ]]; then
  step="Video download"
  video_path="$VTG_ROOT/data/$VIDEO_NAME"
  download_args=(--output "$video_path")
  [[ -z "${VIDEO_SHA256:-}" ]] || download_args+=(--sha256 "$VIDEO_SHA256")
  "$py" "$PROJECT_ROOT/scripts/download_video.py" "${download_args[@]}"
  step="Video decode verification"
  "$py" "$PROJECT_ROOT/scripts/check_grounding.py" --video "$video_path"
fi

step="Activation file"
{
  printf '# Generated by bootstrap_elice.sh; source this in each new terminal.\n'
  printf 'source %q\n' "$env_dir/bin/activate"
  printf 'export GROUNDING_PYTHON=%q\n' "$py"
  printf 'export FORCE_QWENVL_VIDEO_READER=torchvision\n'
  printf 'export HF_HOME=%q\n' "$HF_HOME"
  printf 'export VTG_ROOT=%q\n' "$VTG_ROOT"
  if [[ "$skip_video" -eq 0 ]]; then printf 'export VIDEO_PATH=%q\n' "$video_path"; fi
} > "$VTG_ROOT/activate.sh"
printf '\nREADY: grounding environment verified. No inference has been run.\n'
printf 'Activate in this terminal: source %q\n' "$VTG_ROOT/activate.sh"
printf 'Save experiment outputs under %q and copy them out BEFORE deleting the instance.\n' "$VTG_ROOT/results"
