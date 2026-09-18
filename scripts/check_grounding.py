"""Check real CUDA execution, or decode the first frame of a downloaded video."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video")
    args = parser.parse_args()
    if args.video:
        import av
        with av.open(args.video) as container:
            if not container.streams.video:
                raise RuntimeError("Downloaded file has no video stream")
            stream = container.streams.video[0]
            frame = next(container.decode(video=0), None)
            if frame is None:
                raise RuntimeError("Cannot decode the first video frame")
            duration = (float(stream.duration * stream.time_base)
                        if stream.duration is not None else
                        container.duration / 1_000_000 if container.duration else None)
            print(json.dumps({"first_frame_decoded": True, "width": frame.width,
                              "height": frame.height, "duration_s": duration}))
        return
    import torch
    import torchvision
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
    from qwen_vl_utils import process_vision_info  # noqa: F401
    import av  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: confirm this is a GPU instance")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This grounding setup requires BF16-capable CUDA hardware")
    x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    if not torch.all((x @ x) == 16).item():
        raise RuntimeError("CUDA matrix computation failed")
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
                      "transformers": transformers.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0), "bf16_matmul": "passed"}))


if __name__ == "__main__":
    main()
