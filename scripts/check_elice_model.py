"""Import, CUDA and OSMesa smoke checks; does not claim inference quality."""

import os
import importlib
from pathlib import Path
import sys
import types

model, folder = sys.argv[1:]
r = Path(folder)
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.ones((16, 16), device="cuda")
assert (x @ x).sum().item() == 4096
print("CUDA OK:", torch.cuda.get_device_name(), torch.__version__)
if model == "grounding":
    for name in ("transformers", "accelerate", "qwen_vl_utils", "av", "decord"):
        importlib.import_module(name)
    transformers = importlib.import_module("transformers")
    assert transformers.Qwen2_5_VLForConditionalGeneration
    assert transformers.Qwen3VLForConditionalGeneration
elif model in ("comotion", "multihmr2"):
    import pyrender

    renderer = pyrender.OffscreenRenderer(32, 32)
    scene = pyrender.Scene()
    scene.add(pyrender.PerspectiveCamera(yfov=1.0))
    color, depth = renderer.render(scene)
    assert color.shape == (32, 32, 3)
    renderer.delete()
    print("OSMesa render OK")
    if model == "comotion":
        importlib.import_module("comotion_demo.models.comotion")
        from comotion_demo.utils.smpl_kinematics import SMPLKinematics

        SMPLKinematics().cpu().eval()
    else:
        importlib.import_module("multihmr2.api")
elif model == "motiongpt":
    repo = r / "MotionGPT"
    sys.path.insert(0, str(repo))
    for name in [
        "mGPT.data",
        "mGPT.data.humanml",
        "mGPT.data.humanml.common",
        "mGPT.data.humanml.scripts",
        "mGPT.data.humanml.utils",
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(repo / name.replace(".", "/"))]
        sys.modules[name] = mod
    assert importlib.import_module("mGPT.archs.mgpt_vq").VQVae
    assert importlib.import_module("mGPT.archs.mgpt_lm").MLM
    importlib.import_module("mGPT.data.humanml.scripts.motion_process")
    joints = np.load(r / "assets/reference_joints.npy")
    feats = np.load(r / "assets/reference_features.npy")
    assert joints.ndim == 3 and joints.shape[1:] == (22, 3) and len(joints) > 0
    assert feats.ndim == 2 and feats.shape[1] == 263 and len(feats) > 0
    assert np.isfinite(joints).all() and np.isfinite(feats).all()
    for name in ("mean", "std"):
        arr = np.load(repo / f"assets/meta/{name}.npy")
        assert arr.shape == (263,) and np.isfinite(arr).all()
        if name == "std":
            assert (arr > 0).all()
else:
    import zipfile

    sys.path.insert(0, str(r / "MG-MotionLLM"))
    assert importlib.import_module("models.vqvae").HumanVQVAE
    from transformers import T5Tokenizer

    with zipfile.ZipFile(r / "assets/t2m.zip") as archive:
        for name in ("mean", "std"):
            with archive.open(
                f"t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{name}.npy"
            ) as f:
                arr = np.load(f)
                assert arr.shape == (263,) and np.isfinite(arr).all()
    for task in ("m2t", "m2dt"):
        dst = r / f"assets/mgllm_{task}"
        T5Tokenizer.from_pretrained(dst, local_files_only=True)
        assert list(dst.glob("*.safetensors")) or list(
            dst.glob("pytorch_model*.bin")
        ), f"Missing weights: {dst}"
print(
    "Model imports / environment checks OK (full checkpoint inference still required)."
)
