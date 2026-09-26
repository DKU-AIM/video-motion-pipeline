#!/usr/bin/env python3
"""Local dance-video temporal placement experiment. No external video downloads."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "A group of people are dancing in a room.",
    "Several people dance together indoors.",
    "People are practicing a group dance.",
]


def command(args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def duration(path):
    return float(command(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", path]))


def score(spans, target, length):
    """Union tIoU penalizes extra predictions; invalid/out-of-range spans fail."""
    if not spans:
        return {"valid": False, "tiou": 0.0, "start_error_s": None, "end_error_s": None}
    intervals = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return {"valid": False, "tiou": 0.0, "start_error_s": None, "end_error_s": None}
        a, b = span
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (a, b)) or not 0 <= a < b <= length + 0.1:
            return {"valid": False, "tiou": 0.0, "start_error_s": None, "end_error_s": None}
        intervals.append((a, min(b, length)))
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    intersection = sum(max(0, min(b, target[1]) - max(a, target[0])) for a, b in merged)
    union = sum(b - a for a, b in merged) + target[1] - target[0] - intersection
    return {"valid": True, "tiou": intersection / union,
            "start_error_s": abs(merged[0][0] - target[0]),
            "end_error_s": abs(merged[-1][1] - target[1])}


def prepare(out, source, args):
    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    normalized = out / "videos/source.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-an", "-vf", "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=25,setsar=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(normalized)], check=True)
    d = duration(normalized)
    cases = []
    for name, before, after in [("early", 4, 12), ("late", 12, 4)]:
        path = out / "videos" / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(normalized),
                        "-vf", f"tpad=start_duration={before}:stop_duration={after}:start_mode=add:stop_mode=add:color=black",
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
        cases.append({"id": name, "path": str(path), "duration": duration(path),
                      "target": [before, before + d]})
    try:
        commit = command(["git", "-C", ROOT, "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        commit = "unknown"
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reference_type": "synthetic source insertion interval; not human action-boundary ground truth",
                "queries": QUERIES, "cases": cases, "models": args.models,
                "settings": {"total_tokens": args.total_tokens, "fps": 2.0, "max_frames": 96,
                             "max_new_tokens": 256, "attn": "sdpa", "do_sample": False},
                "code_commit": commit}
    save(out / "manifest.json", manifest)
    return manifest


def report(out, manifest, rows):
    fields = ["model", "case", "query", "status", "valid", "tiou", "start_error_s",
              "end_error_s", "wall_s", "peak_vram_gb", "spans", "error"]
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Local grounding experiment", "",
             "Reference: synthetic insertion interval, NOT human-annotated action boundaries.",
             "Black padding makes this a simple diagnostic, not a real-world benchmark.",
             "Invalid/empty predictions and failed trials receive tIoU=0; extra spans are penalized via union IoU.",
             "wall_s includes video preprocessing + inference (excludes model load). VRAM includes loaded model.", "",
             "| Model | Completed / planned | Valid | Mean tIoU | tIoU >= 0.5 / planned |",
             "|---|---:|---:|---:|---:|"]
    for model in manifest["models"]:
        rs = [r for r in rows if r["model"] == model]
        n = len(manifest["cases"]) * len(manifest["queries"])
        lines.append(f"| {model} | {sum(r['status'] == 'ok' for r in rs)}/{n} | {sum(r['valid'] for r in rs)} | {sum(r['tiou'] for r in rs)/n:.3f} | {sum(r['tiou'] >= .5 for r in rs)}/{n} |")
    lines += ["", "Compare early vs late (expected shift: +8 s), and the three query paraphrases in summary.csv.",
              "Watch videos/early.mp4 and videos/late.mp4 to check the reference and predictions.",
              "Raw model responses, errors and timings: results.jsonl. Settings/provenance: manifest.json."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(out, manifest):
    import torch
    import vtg_run as v
    v.preflight()
    (out / "pip-freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]) + "\n")
    (out / "gpu.txt").write_text(command(["nvidia-smi"]) + "\n")
    rows = []
    failures = False
    with (out / "results.jsonl").open("w") as f:
        for model in manifest["models"]:
            g = None
            load_error = None
            try:
                g = v.Grounder(model, total_tokens=manifest["settings"]["total_tokens"],
                               attn="sdpa", fps=2.0, max_frames=96)
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
            try:
                for case in manifest["cases"]:
                    for query in manifest["queries"]:
                        row = {"model": model, "case": case["id"], "query": query,
                               "target": case["target"], "status": "error", "spans": [],
                               "raw": "", "error": "", "valid": False, "tiou": 0.0}
                        start = time.perf_counter()
                        try:
                            if load_error:
                                raise RuntimeError(load_error)
                            torch.cuda.reset_peak_memory_stats()
                            result = g(case["path"], query, max_new_tokens=256)
                            torch.cuda.synchronize()
                            row.update(result)
                            row.update(score(result["spans"], case["target"], case["duration"]))
                            row["status"] = "ok"
                        except Exception as exc:
                            failures = True
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            torch.cuda.empty_cache()
                        row["wall_s"] = round(time.perf_counter() - start, 3)
                        rows.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        report(out, manifest, rows)
                        print(f"{model} / {case['id']} / {query}: {row['status']} tIoU={row['tiou']:.3f} {row['error']}", flush=True)
            finally:
                if g is not None:
                    g.free()
    report(out, manifest, rows)
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path(os.environ.get("MOTION_WORKSPACE", Path.home() / "aim-workspace")))
    p.add_argument("--out", type=Path)
    p.add_argument("--models", nargs="+", choices=["timelens2-4b", "timelens2-8b"], default=["timelens2-4b", "timelens2-8b"])
    p.add_argument("--total-tokens", type=int, default=8192)
    p.add_argument("--prepare-only", action="store_true", help="Prepare videos and manifest without model download/GPU inference")
    args = p.parse_args()
    if args.total_tokens <= 0:
        p.error("--total-tokens must be positive")
    args.models = list(dict.fromkeys(args.models))
    source = args.workspace / "multi-hmr2/demo_data/sample_video.mp4"
    if not source.is_file():
        p.error(f"Missing source video: {source}")
    out = (args.out or args.workspace / "results_grounding" / time.strftime("local_dance_%Y%m%d_%H%M%S")).resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f"Output must be new or empty (existing results protected): {out}")
    manifest = prepare(out, source, args)
    print(f"Experiment: {out}", flush=True)
    if args.prepare_only:
        print("Prepared only; no model inference was performed.")
        return 0
    return run(out, manifest)


if __name__ == "__main__":
    sys.exit(main())
