"""Reconstruct reference joints from the team's existing HumanML3D reference features."""

from pathlib import Path
import sys
import types
import numpy as np
import torch

r = Path(sys.argv[1]).expanduser().resolve()
repo = r / "MotionGPT"
sys.path.insert(0, str(repo))
for name in [
    "mGPT.data",
    "mGPT.data.humanml",
    "mGPT.data.humanml.common",
    "mGPT.data.humanml.scripts",
    "mGPT.data.humanml.utils",
]:
    module = types.ModuleType(name)
    module.__path__ = [str(repo / name.replace(".", "/"))]
    sys.modules[name] = module
from mGPT.data.humanml.scripts.motion_process import recover_from_ric

features = np.load(r / "assets/reference_features.npy")
assert (
    features.ndim == 2
    and features.shape[1] == 263
    and len(features) > 0
    and np.isfinite(features).all()
)
dst = r / "assets/reference_joints.npy"
if dst.exists():
    print("Preserving existing", dst)
else:
    joints = recover_from_ric(torch.from_numpy(features).float(), 22).numpy()
    assert joints.shape == (len(features), 22, 3) and np.isfinite(joints).all()
    np.save(dst, joints)
    print("Recovered joints from supplied reference features:", dst)
