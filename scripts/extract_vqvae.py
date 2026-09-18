"""Extract only the expected VQ-VAE checkpoint (never run upstream rm -rf scripts)."""

from pathlib import Path
import shutil
import sys
import zipfile

workspace = Path(sys.argv[1])
with zipfile.ZipFile(workspace / "assets/pretrained_vqvae.zip") as archive:
    matches = [
        n
        for n in archive.namelist()
        if n.rstrip("/").endswith("pretrained_vqvae/t2m.pth")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one pretrained_vqvae/t2m.pth in official archive"
        )
    dst = workspace / "MG-MotionLLM/checkpoints/pretrained_vqvae/t2m.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with archive.open(matches[0]) as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp.replace(dst)
