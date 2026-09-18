#!/usr/bin/env bash
# Prepare isolated CoMotion and MotionGPT inference environments on Elice Linux.
# It deliberately never downloads, redistributes, or invents an SMPL body model.
set -Eeuo pipefail

COMOTION_REVISION=04b035af79a68267f6bb722b61bf5f13cb2b0068
MOTIONGPT_REVISION=001aaca8d0ee218fc17f8265d11ac124044fe42f
MOTIONGPT_MODEL_REVISION=a0a37a388137f15df8299c643a885a42b07772fe
HUMANML3D_REVISION=9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23
HUMANML3D_JOINTS_SHA256=739df5e2e025d21ae5ce01ea7e0a8d3038448a46e88865d95a638e6242ee8b6b
HUMANML3D_FEATURES_SHA256=6e6fc0204ea9e48abdf94f84e8be22b19a1f1e955c7b98e025b1d702f6fe96a0

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_motion_demo.sh [options]

Options:
  --assets-root PATH  Persistent path for source, virtualenvs and model assets.
                      Default: $MOTION_ASSETS_ROOT or ~/motion-workspace
  --smpl PATH         Authorized SMPL v1.1.0 neutral model .pkl. It is copied
                      only into the private assets root as SMPL_NEUTRAL.pkl.
  --prepare-only      Set up public code, Python environments, checkpoints, and
                      the public HumanML3D sanity sample. This is the default.
  --install-system    Install ffmpeg and OSMesa renderer libraries through
                      passwordless sudo. Elice's image must allow sudo -n.
  --check             Do not install or download. Report every prerequisite.
  -h, --help          Show this help.

The script needs Linux x86_64, Python 3.10, CUDA-visible GPU, ffmpeg/ffprobe,
and OSMesa for the repository's offscreen renderer.
HELP
}

ASSETS_ROOT="${MOTION_ASSETS_ROOT:-$HOME/motion-workspace}"
SMPL_SOURCE="${SMPL_NEUTRAL_PATH:-}"
CHECK_ONLY=0
INSTALL_SYSTEM=0

while (($#)); do
  case "$1" in
    --assets-root) ASSETS_ROOT=${2:?--assets-root needs a path}; shift 2 ;;
    --smpl) SMPL_SOURCE=${2:?--smpl needs a path}; shift 2 ;;
    --prepare-only) shift ;;
    --install-system) INSTALL_SYSTEM=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ASSETS_ROOT=$(python3 - "$ASSETS_ROOT" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)
case "$ASSETS_ROOT" in
  /|"") echo "Refusing an unsafe assets root: $ASSETS_ROOT" >&2; exit 2 ;;
esac

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; return 1; }
}
hash_file() { sha256sum "$1" | awk '{print $1}'; }
require_file_hash() {
  local path=$1 expected=$2 actual
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; return 1; }
  actual=$(hash_file "$path")
  [[ "$actual" == "$expected" ]] || { echo "SHA-256 mismatch: $path" >&2; return 1; }
}
check_python() {
  local py=$1 label=$2
  "$py" - "$label" <<'PY'
import sys
label = sys.argv[1]
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"{label} must use Python 3.10; found {sys.version.split()[0]}")
PY
}
check_venv() {
  local path=$1 label=$2
  [[ -f "$path/pyvenv.cfg" && -x "$path/bin/python" ]] || {
    echo "$label exists but is not a Python virtual environment: $path" >&2
    return 1
  }
}
check_cuda() {
  local py=$1 label=$2
  "$py" - "$label" <<'PY'
import sys, torch
label = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(f"{label}: torch.cuda.is_available() is false")
print(f"{label}: torch={torch.__version__}, gpu={torch.cuda.get_device_name(0)}")
PY
}
check_pinned_repo() {
  local path=$1 revision=$2 label=$3
  [[ -d "$path/.git" ]] || { echo "Missing $label source: $path" >&2; return 1; }
  [[ $(git -C "$path" rev-parse HEAD) == "$revision" ]] || {
    echo "$label is not at its pinned revision: $path" >&2
    return 1
  }
}
check_pose_imports() {
  local py=$1
  PYOPENGL_PLATFORM=osmesa "$py" - <<'PY'
import comotion_demo, pyrender, trimesh
from pyrender import OffscreenRenderer
renderer = OffscreenRenderer(16, 16)
renderer.delete()
print("pose imports and OSMesa renderer: OK")
PY
}
check_caption_imports() {
  local py=$1 repo=$2
  "$py" - "$repo" <<'PY'
import pathlib, sys, types
repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo))
for name in ['mGPT.data','mGPT.data.humanml','mGPT.data.humanml.common','mGPT.data.humanml.scripts','mGPT.data.humanml.utils']:
    module = types.ModuleType(name)
    module.__path__ = [str(repo / name.replace('.', '/'))]
    sys.modules[name] = module
from mGPT.archs.mgpt_vq import VQVae
from mGPT.archs.mgpt_lm import MLM
from mGPT.data.humanml.scripts import motion_process
from mGPT.data.humanml.common.skeleton import Skeleton
import scipy, tqdm, transformers
print("caption imports: OK")
PY
}
write_manifest() {
  local pose_versions caption_versions
  pose_versions=$("$POSE_ENV/bin/python" - <<'PY'
import importlib.metadata as md, json
names = ['torch', 'torchvision', 'PyOpenGL', 'pyrender', 'numpy', 'scenedetect', 'opencv-python']
print(json.dumps({name: md.version(name) for name in names}, sort_keys=True))
PY
)
  caption_versions=$("$CAPTION_ENV/bin/python" - <<'PY'
import importlib.metadata as md, json
names = ['torch', 'torchvision', 'numpy', 'scipy', 'transformers', 'sentencepiece', 'tqdm']
print(json.dumps({name: md.version(name) for name in names}, sort_keys=True))
PY
)
  "$CAPTION_ENV/bin/python" - "$ASSETS_ROOT/setup-manifest.json" \
    "$COMOTION_DIR" "$MOTIONGPT_DIR" "$ASSET_DIR/motiongpt_s3_h3d.tar" \
    "$ASSET_DIR/reference_joints.npy" "$ASSET_DIR/reference_features.npy" \
    "$pose_versions" "$caption_versions" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, comotion, motiongpt, checkpoint, joints, features = map(pathlib.Path, sys.argv[1:7])
pose_versions, caption_versions = sys.argv[7:]
def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
data = {
    'comotion_revision': revision(comotion),
    'motiongpt_revision': revision(motiongpt),
    'motiongpt_checkpoint_sha256': sha256(checkpoint),
    'humanml3d_reference_joints_sha256': sha256(joints),
    'humanml3d_reference_features_sha256': sha256(features),
    'pose_environment': json.loads(pose_versions),
    'caption_environment': json.loads(caption_versions),
    'nvidia_smi': subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout.strip(),
}
out.write_text(json.dumps(data, indent=2) + '\n')
PY
}
clone_at_revision() {
  local url=$1 dest=$2 revision=$3
  if [[ -e "$dest" ]]; then
    [[ -d "$dest/.git" ]] || { echo "Expected git repository: $dest" >&2; return 1; }
    [[ $(git -C "$dest" rev-parse HEAD) == "$revision" ]] || {
      echo "Existing source is not the pinned revision: $dest" >&2
      echo "Expected $revision, found $(git -C "$dest" rev-parse HEAD). Use a new --assets-root." >&2
      return 1
    }
    return
  fi
  git clone "$url" "$dest"
  git -C "$dest" checkout --detach "$revision"
}

POSE_ENV="$ASSETS_ROOT/pose-env"
CAPTION_ENV="$ASSETS_ROOT/caption-env"
COMOTION_DIR="$ASSETS_ROOT/ml-comotion"
MOTIONGPT_DIR="$ASSETS_ROOT/MotionGPT"
ASSET_DIR="$ASSETS_ROOT/assets"

if ((CHECK_ONLY)); then
  failures=0
  for command_name in ffmpeg ffprobe nvidia-smi sha256sum; do require_command "$command_name" || failures=1; done
  [[ -x "$POSE_ENV/bin/python" ]] || { echo "Missing pose environment: $POSE_ENV" >&2; failures=1; }
  [[ -x "$CAPTION_ENV/bin/python" ]] || { echo "Missing caption environment: $CAPTION_ENV" >&2; failures=1; }
  check_pinned_repo "$COMOTION_DIR" "$COMOTION_REVISION" CoMotion || failures=1
  check_pinned_repo "$MOTIONGPT_DIR" "$MOTIONGPT_REVISION" MotionGPT || failures=1
  [[ -f "$COMOTION_DIR/src/comotion_demo/data/comotion_detection_checkpoint.pt" ]] || { echo "Missing CoMotion detection checkpoint" >&2; failures=1; }
  [[ -f "$COMOTION_DIR/src/comotion_demo/data/comotion_refine_checkpoint.pt" ]] || { echo "Missing CoMotion refine checkpoint" >&2; failures=1; }
  [[ -f "$COMOTION_DIR/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl" ]] || { echo "Missing licensed SMPL model; see docs/MOTION_SETUP.md" >&2; failures=1; }
  [[ -f "$ASSET_DIR/motiongpt_s3_h3d.tar" ]] || { echo "Missing MotionGPT checkpoint" >&2; failures=1; }
  [[ -f "$MOTIONGPT_DIR/assets/meta/mean.npy" && -f "$MOTIONGPT_DIR/assets/meta/std.npy" ]] || { echo "Missing MotionGPT normalization metadata" >&2; failures=1; }
  require_file_hash "$ASSET_DIR/reference_joints.npy" "$HUMANML3D_JOINTS_SHA256" || failures=1
  require_file_hash "$ASSET_DIR/reference_features.npy" "$HUMANML3D_FEATURES_SHA256" || failures=1
  if [[ -x "$POSE_ENV/bin/python" ]]; then
    check_venv "$POSE_ENV" pose-env || failures=1
    check_python "$POSE_ENV/bin/python" pose || failures=1
    "$POSE_ENV/bin/python" -m pip check || failures=1
    check_cuda "$POSE_ENV/bin/python" pose || failures=1
    check_pose_imports "$POSE_ENV/bin/python" || failures=1
  fi
  if [[ -x "$CAPTION_ENV/bin/python" ]]; then
    check_venv "$CAPTION_ENV" caption-env || failures=1
    check_python "$CAPTION_ENV/bin/python" caption || failures=1
    "$CAPTION_ENV/bin/python" -m pip check || failures=1
    check_cuda "$CAPTION_ENV/bin/python" caption || failures=1
    check_caption_imports "$CAPTION_ENV/bin/python" "$MOTIONGPT_DIR" || failures=1
  fi
  [[ -f "$ASSETS_ROOT/setup-manifest.json" ]] || { echo "Missing setup manifest" >&2; failures=1; }
  ((failures == 0)) || exit 1
  echo "Motion demo prerequisites are ready: $ASSETS_ROOT"
  exit 0
fi

[[ $(uname -s) == Linux ]] || { echo "This setup is for an Elice Linux GPU instance." >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || { echo "This setup requires Linux x86_64." >&2; exit 2; }
if ((INSTALL_SYSTEM)); then
  require_command sudo
  sudo -n true || { echo "--install-system needs passwordless sudo on this Elice image." >&2; exit 1; }
  sudo -n apt-get update
  sudo -n apt-get install -y ffmpeg libosmesa6 libgl1-mesa-dri
fi
for command_name in python3.10 git curl tar sha256sum ffmpeg ffprobe nvidia-smi; do require_command "$command_name"; done
check_python python3.10 system-python
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ -n "$SMPL_SOURCE" && ! -f "$SMPL_SOURCE" ]]; then
  echo "--smpl path does not exist: $SMPL_SOURCE" >&2
  exit 2
fi

mkdir -p "$ASSETS_ROOT" "$ASSET_DIR"
clone_at_revision https://github.com/apple/ml-comotion.git "$COMOTION_DIR" "$COMOTION_REVISION"
clone_at_revision https://github.com/OpenMotionLab/MotionGPT.git "$MOTIONGPT_DIR" "$MOTIONGPT_REVISION"

if [[ -d "$POSE_ENV" ]]; then check_venv "$POSE_ENV" existing-pose-env; check_python "$POSE_ENV/bin/python" existing-pose-env; else python3.10 -m venv "$POSE_ENV"; fi
"$POSE_ENV/bin/python" -m pip install --upgrade pip
"$POSE_ENV/bin/python" -m pip install --upgrade setuptools wheel
"$POSE_ENV/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$POSE_ENV/bin/python" -m pip install -r "$(dirname "$0")/constraints-comotion-elice.txt"
# CoMotion pins chumpy from Git. Its legacy setup.py imports pip, so PEP 517's
# minimal isolated builder fails; install it in this already-isolated venv.
"$POSE_ENV/bin/python" -m pip install --no-build-isolation -c "$(dirname "$0")/constraints-comotion-elice.txt" -e "$COMOTION_DIR"
"$POSE_ENV/bin/python" -m pip install PyOpenGL==3.1.7 trimesh==3.23.5 Pillow==10.4.0
# PyPI pyrender 0.1.45 pins PyOpenGL==3.1.0, while the repository renderer
# requires OSMesa support released in later 3.1.x versions. This upstream
# pyrender commit changes that pin to ~=3.1.0, which accepts 3.1.7.
"$POSE_ENV/bin/python" -m pip install "git+https://github.com/mmatl/pyrender.git@7c613e8aed7142df9ff40767a8f10b7a19b6255c"
"$POSE_ENV/bin/python" -m pip check
check_cuda "$POSE_ENV/bin/python" pose

if [[ -d "$CAPTION_ENV" ]]; then check_venv "$CAPTION_ENV" existing-caption-env; check_python "$CAPTION_ENV/bin/python" existing-caption-env; else python3.10 -m venv "$CAPTION_ENV"; fi
"$CAPTION_ENV/bin/python" -m pip install --upgrade pip
"$CAPTION_ENV/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$CAPTION_ENV/bin/python" -m pip install -r "$(dirname "$0")/requirements-motiongpt-inference.txt"
"$CAPTION_ENV/bin/python" -m pip check
check_cuda "$CAPTION_ENV/bin/python" caption

# The Apple script has the authoritative URL. Extract in a temporary directory,
# inspect the two code-referenced paths, then atomically place them.
if [[ ! -f "$COMOTION_DIR/src/comotion_demo/data/comotion_detection_checkpoint.pt" || ! -f "$COMOTION_DIR/src/comotion_demo/data/comotion_refine_checkpoint.pt" ]]; then
  archive=$(mktemp "$ASSETS_ROOT/comotion-checkpoints.XXXXXX")
  extract=$(mktemp -d "$ASSETS_ROOT/comotion-extract.XXXXXX")
  trap 'rm -f "$archive"; rm -rf "$extract"' EXIT
  curl --fail --location --retry 3 --output "$archive" https://ml-site.cdn-apple.com/models/comotion/demo_checkpoints.tar.gz
  tar -xzf "$archive" -C "$extract"
  detection=$(find "$extract" -name comotion_detection_checkpoint.pt -type f -print -quit)
  refine=$(find "$extract" -name comotion_refine_checkpoint.pt -type f -print -quit)
  [[ -n "$detection" && -n "$refine" ]] || { echo "Official CoMotion archive did not contain the two required checkpoints." >&2; exit 1; }
  install -m 0644 "$detection" "$COMOTION_DIR/src/comotion_demo/data/comotion_detection_checkpoint.pt"
  install -m 0644 "$refine" "$COMOTION_DIR/src/comotion_demo/data/comotion_refine_checkpoint.pt"
  rm -f "$archive"; rm -rf "$extract"; trap - EXIT
fi

if [[ ! -f "$ASSET_DIR/motiongpt_s3_h3d.tar" ]]; then
  "$CAPTION_ENV/bin/python" - "$ASSET_DIR" "$MOTIONGPT_MODEL_REVISION" <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import sys
target = Path(sys.argv[1]) / "motiongpt_s3_h3d.tar"
downloaded = hf_hub_download(
    repo_id="OpenMotionLab/MotionGPT-base",
    filename="motiongpt_s3_h3d.tar",
    revision=sys.argv[2],
)
target.symlink_to(downloaded)
PY
fi

for sample in joints features; do
  case "$sample" in
    joints) destination="$ASSET_DIR/reference_joints.npy"; relative=HumanML3D/new_joints/012314.npy; expected="$HUMANML3D_JOINTS_SHA256" ;;
    features) destination="$ASSET_DIR/reference_features.npy"; relative=HumanML3D/new_joint_vecs/012314.npy; expected="$HUMANML3D_FEATURES_SHA256" ;;
  esac
  if [[ ! -f "$destination" ]]; then
    curl --fail --location --retry 3 --output "$destination.tmp" "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/$HUMANML3D_REVISION/$relative"
    mv "$destination.tmp" "$destination"
  fi
  require_file_hash "$destination" "$expected"
done

if [[ -n "$SMPL_SOURCE" ]]; then
  [[ -f "$SMPL_SOURCE" ]] || { echo "--smpl path does not exist: $SMPL_SOURCE" >&2; exit 2; }
  mkdir -p "$COMOTION_DIR/src/comotion_demo/data/smpl"
  install -m 0600 "$SMPL_SOURCE" "$COMOTION_DIR/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
fi

write_manifest

cat > "$ASSETS_ROOT/activate-motion.sh" <<EOF
# Source this file in the Elice terminal before running the demo.
export MOTION_ASSETS_ROOT=$(printf '%q' "$ASSETS_ROOT")
export MOTION_WORKSPACE="\${MOTION_WORKSPACE:-\$MOTION_ASSETS_ROOT/run-workspace}"
export POSE_PYTHON="\$MOTION_ASSETS_ROOT/pose-env/bin/python"
export CAPTION_PYTHON="\$MOTION_ASSETS_ROOT/caption-env/bin/python"
export PYOPENGL_PLATFORM=osmesa
EOF

echo
echo "Public CoMotion and MotionGPT setup finished: $ASSETS_ROOT"
if [[ ! -f "$COMOTION_DIR/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl" ]]; then
  echo "SMPL is intentionally still missing. Obtain the licensed neutral v1.1.0 model, then rerun:"
  echo "  bash scripts/setup_motion_demo.sh --assets-root $(printf '%q' "$ASSETS_ROOT") --smpl /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
fi
echo "Next check: bash scripts/setup_motion_demo.sh --assets-root $(printf '%q' "$ASSETS_ROOT") --check"
