#!/usr/bin/env python3
"""Bounded grounding → CoMotion → MotionGPT demo. Models run in separate processes."""
import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def select_span(spans, duration, max_seconds):
    """First chronological viable prediction; clip bounds do not imply correctness."""
    valid = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in span):
            continue
        start, end = max(0.0, float(span[0])), min(duration, float(span[1]))
        end = min(end, start + max_seconds)
        if end - start >= 2:
            valid.append((start, end))
    if not valid:
        raise ValueError('No valid grounding span of at least 2 seconds. Change the query/window; no motion was invented.')
    return min(valid)


def probe_duration(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams',
                             '-of', 'json', str(path)], check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    videos = [s for s in data['streams'] if s.get('codec_type') == 'video']
    if not videos:
        raise ValueError('Input has no video stream.')
    duration = float(videos[0].get('duration') or data['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid video duration.')
    return duration


def cut_video(source, output, start, duration, log):
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'warning', '-n', '-ss', str(start),
                        '-i', str(source), '-t', str(duration), '-map', '0:v:0', '-an', '-sn',
                        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-vsync', 'cfr',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(output)], check=True, stdout=stream, stderr=subprocess.STDOUT)
    if probe_duration(output) < 2:
        raise ValueError('Decoded clip is shorter than 2 seconds.')


def run_stage(name, cmd, run, env):
    log = run/'logs'/f'{name}.log'
    print(f'[{name}] Running; log: {log}', flush=True)
    with log.open('w') as stream:
        result = subprocess.run([str(x) for x in cmd], cwd=REPO, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed (exit {result.returncode}). See {log}')
    print(f'[{name}] Done', flush=True)


def preflight(args):
    missing = []
    for command in ('ffmpeg', 'ffprobe'):
        if not shutil.which(command):
            missing.append(command + ' (system package)')
    required = [args.assets_root/'ml-comotion/demo.py',
                args.assets_root/'ml-comotion/src/comotion_demo/data/comotion_detection_checkpoint.pt',
                args.assets_root/'ml-comotion/src/comotion_demo/data/comotion_refine_checkpoint.pt', args.assets_root/'MotionGPT/mGPT/archs/mgpt_lm.py',
                args.assets_root/'ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl',
                args.assets_root/'assets/motiongpt_s3_h3d.tar', args.assets_root/'assets/reference_joints.npy',
                args.assets_root/'assets/reference_features.npy', args.assets_root/'MotionGPT/assets/meta/mean.npy',
                args.assets_root/'MotionGPT/assets/meta/std.npy']
    for path in required:
        if not path.is_file():
            missing.append(str(path))
    for python in (args.grounding_python, args.pose_python, args.caption_python):
        if not Path(python).is_file() and not shutil.which(python):
            missing.append(python)
    if missing:
        raise RuntimeError('Missing prerequisites:\n  ' + '\n  '.join(missing) +
                           '\nRun scripts/setup_motion_demo.sh; see docs/MOTION_SETUP.md. SMPL is supplied separately.')
    checks = [
        (args.grounding_python, 'from transformers import AutoModelForImageTextToText, AutoProcessor; import qwen_vl_utils, av'),
        (args.pose_python, 'import comotion_demo, numpy, PIL; from smpl_eval.meshrender import MeshRenderer; r=MeshRenderer(32,32); r.close()'),
        (args.caption_python, 'import sys; sys.path.insert(0, ' + repr(str(args.assets_root/'MotionGPT')) + '); import numpy, scipy; from mGPT.archs.mgpt_vq import VQVae; from mGPT.archs.mgpt_lm import MLM'),
    ]
    env = {**os.environ, 'PYTHONPATH': str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'PYOPENGL_PLATFORM': 'osmesa'}
    for python, imports in checks:
        subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; ' + imports],
                       cwd=REPO, env=env, check=True)
    print('Prerequisites OK. Model checkpoint compatibility is verified during inference.', flush=True)


def write_report(run, metadata, captions):
    rows = [r for r in captions if not r.get('already_features')]
    if not rows:
        raise ValueError('No per-track captions to display.')
    cards = []
    for row in rows:
        # Reject unrelated/escaped outputs instead of linking another experiment accidentally.
        pose_dir = Path(row['source_parameters']).resolve().parent
        relative_video = (pose_dir/'comparison.mp4').relative_to(run.resolve())
        if not (run/relative_video).is_file():
            raise ValueError('Missing comparison video for caption.')
        fps = float(row['source_fps'])
        first, last = int(row['source_frame_start']), int(row['source_frame_end'])
        local_start, local_end = first/fps, (last+1)/fps
        start, end = metadata['clip_start']+local_start, metadata['clip_start']+local_end
        if not isinstance(row.get('caption'), str) or not row['caption'].strip():
            raise ValueError('MotionGPT returned an empty caption.')
        caption = html.escape(row['caption'])
        video = quote(relative_video.as_posix(), safe='/')
        cards.append(f'''<section><h2>Track {int(row['track_id'])}</h2>
<p>원본 기준 {start:.3f}–{end:.3f}초 · 클립 기준 {local_start:.3f}–{local_end:.3f}초</p>
<video controls preload="metadata" src="{video}#t={local_start:.3f},{local_end:.3f}"></video>
<p class="caption">{caption}</p><p class="note">설명은 이 트랙의 표시된 구간에 대한 모션 기반 예측입니다. 영상의 모든 사람에 대한 설명이 아닙니다.</p>
<a href="{video}">메시 비교 영상 전체</a></section>''')
    meta = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>AIM · Motion demo</title><style>
body{{max-width:1040px;margin:40px auto;padding:0 22px;background:#f4f5f7;color:#17212e;font:16px/1.65 system-ui,sans-serif}}
h1{{font-size:30px}}section{{background:white;padding:24px;border:1px solid #dde2e9;border-radius:14px;margin:24px 0}}
video{{width:100%;background:#101827;border-radius:8px}}.caption{{font-size:22px;font-weight:600}}.note{{color:#526173}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#2357b7}}.tag{{color:#2357b7;font-weight:bold}}
</style><body><p class="tag">AIM · Grounding → CoMotion → MotionGPT</p>
<h1>동작 복원과 설명 확인</h1><p>요청: <strong>{html.escape(metadata['query'])}</strong></p>
<p>원본에서 {metadata['clip_start']:.3f}–{metadata['clip_end']:.3f}초를 사용했습니다. 왼쪽은 입력, 오른쪽은 복원 메시입니다.</p>
<p class="note">자동 선택된 긴 연속 트랙입니다. 요청 행동을 한 인물인지 직접 확인하세요. MotionGPT는 RGB를 보지 않으며 설명은 정답 라벨이 아닙니다. 카메라 기준 모션으로, 월드 좌표 보정은 하지 않았습니다.</p>
{''.join(cards)}<section><h2>실험 기록</h2><a href="grounding.jsonl">그라운딩 출력</a> ·
<a href="results_motiongpt/captions.json">설명 JSON</a> · <a href="motion_inputs/manifest.json">트랙 정보</a>
<details><summary>영상·시간 구간·설정</summary><pre>{meta}</pre></details></section></body></html>'''
    report = run/'index.html'
    report.write_text(page, encoding='utf-8')
    return report


def package_review(run):
    """Portable visual review without model symlinks, frame dumps or private URLs."""
    members = [run/'index.html', run/'demo.json', run/'grounding.jsonl',
               run/'motion_inputs/manifest.json', run/'results_motiongpt/captions.json']
    members += list((run/'inputs').glob('*/*.mp4'))
    members += list((run/'results').glob('*/*/comotion/comparison.mp4'))
    members += list((run/'results').glob('*/*/comotion/metadata.json'))
    archive = run/'review.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as bundle:
        for path in members:
            bundle.write(path, path.relative_to(run).as_posix())
    return archive


def run_demo(args):
    if args.check:
        preflight(args)
        return None
    if not args.video or not args.query or not args.video.is_file():
        raise ValueError('Provide --video PATH (existing file) and --query TEXT.')
    preflight(args)  # Fail before creating outputs or using expensive model inference.
    if args.output:
        run = args.output
    else:
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
        run = args.assets_root/'runs'/name
    run.mkdir(parents=True, exist_ok=False)
    (run/'logs').mkdir()
    env = {**os.environ, 'MOTION_WORKSPACE': str(run), 'FORCE_QWENVL_VIDEO_READER': 'torchvision',
           'PYTHONPATH': str(REPO) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    stage = 'prepare'
    try:
        for name in ('ml-comotion', 'MotionGPT', 'assets'):
            (run/name).symlink_to(args.assets_root/name, target_is_directory=True)
        def status(name):
            save_json(run/'status.json', {'state':'running', 'stage':name, 'updated':datetime.now(timezone.utc).isoformat()})
        status(stage)
        duration = probe_duration(args.video)
        window_duration = min(args.window_seconds, duration-args.window_start)
        if window_duration < 2:
            raise ValueError('Source window is shorter than 2 seconds; adjust --window-start/--window-seconds.')
        grounding_video = run/'grounding_input.mp4'
        cut_video(args.video, grounding_video, args.window_start, window_duration, run/'logs/window.log')
        stage = 'grounding'; status(stage)
        run_stage(stage, [args.grounding_python, REPO/'grounding/vtg_run.py', '--model', args.model,
                         '--video', grounding_video, '--query', args.query, '--out', run/'grounding.jsonl'], run, env)
        predictions = [json.loads(line) for line in (run/'grounding.jsonl').read_text().splitlines() if line.strip()]
        if len(predictions) != 1:
            raise ValueError('Expected one grounding result; inference may have failed or exhausted VRAM.')
        start, end = select_span(predictions[0].get('spans', []), window_duration, args.max_clip_seconds)
        metadata = {'source_video': str(args.video), 'query':args.query, 'model':args.model,
                    'window_start':args.window_start, 'window_duration':window_duration,
                    'predicted_spans':predictions[0]['spans'], 'clip_start':args.window_start+start,
                    'clip_end':args.window_start+end, 'clip_limit_seconds':args.max_clip_seconds,
                    'python':{'grounding':args.grounding_python,'pose':args.pose_python,'caption':args.caption_python}}
        revision = subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'], text=True, capture_output=True)
        metadata['pipeline_revision'] = revision.stdout.strip() if revision.returncode == 0 else 'unknown'
        save_json(run/'demo.json', metadata)
        stage = 'clip'; status(stage)
        clip = run/'inputs/clip/full.mp4'
        cut_video(grounding_video, clip, start, end-start, run/'logs/clip.log')
        save_json(run/'clip_manifest.json', [{'id':'clip', 'source_start_seconds':metadata['clip_start'],
                  'source_video':str(args.video), 'category':'demo', 'roi_xywh':None}])
        commands = [
            ('pose', [args.pose_python, REPO/'mesh_recovery/run_pose_batch.py', '--models','comotion','--conditions','full']),
            ('export', [args.pose_python, REPO/'annotation/motion_captioning/export_motion_inputs.py']),
            ('caption', [args.caption_python, REPO/'annotation/motion_captioning/run_motiongpt.py']),
        ]
        for stage, command in commands:
            status(stage)
            run_stage(stage, command, run, env)
            if stage == 'pose' and not (run/'results/clip/full/comotion/verified.json').is_file():
                raise ValueError('Pose did not produce a verified comparison video.')
            if stage == 'export' and not json.loads((run/'motion_inputs/manifest.json').read_text()):
                raise ValueError('No continuous person track of at least 2 seconds. Try another clip/query.')
        stage = 'report'; status(stage)
        report = write_report(run, metadata, json.loads((run/'results_motiongpt/captions.json').read_text()))
        archive = package_review(run)
        save_json(run/'status.json', {'state':'complete','stage':'report','report':str(report),'review_archive':str(archive)})
        print(f'\nDONE: {report}\nDownload {archive}, unzip and open index.html to review.\nRaw motion data stays in {run}; back it up separately before deleting the instance.', flush=True)
        return report
    except Exception as exc:
        save_json(run/'status.json', {'state':'failed','stage':stage,'error':str(exc)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=lambda p:Path(p).expanduser().resolve())
    parser.add_argument('--query')
    parser.add_argument('--model', choices=['timelens-8b','timelens-7b','timelens2-8b','timelens2-4b','time-r1-7b','time-r1-3b'], default='timelens-8b')
    parser.add_argument('--window-start', type=float, default=0)
    parser.add_argument('--window-seconds', type=float, default=30)
    parser.add_argument('--max-clip-seconds', type=float, default=6)
    parser.add_argument('--assets-root', type=lambda p:Path(p).expanduser().resolve(),
                        default=Path(os.environ.get('MOTION_ASSETS_ROOT', '~/motion-workspace')).expanduser().resolve())
    parser.add_argument('--output', type=lambda p:Path(p).expanduser().resolve(), help='New output directory; never overwrite an existing run.')
    parser.add_argument('--grounding-python', default=os.environ.get('GROUNDING_PYTHON', str(Path.home()/'vtg-env/bin/python')))
    parser.add_argument('--pose-python', default=os.environ.get('POSE_PYTHON'))
    parser.add_argument('--caption-python', default=os.environ.get('CAPTION_PYTHON'))
    parser.add_argument('--check', action='store_true', help='Check prerequisites only; no model inference.')
    args = parser.parse_args(argv)
    args.pose_python = args.pose_python or str(args.assets_root/'pose-env/bin/python')
    args.caption_python = args.caption_python or str(args.assets_root/'caption-env/bin/python')
    for field in ('window_start','window_seconds','max_clip_seconds'):
        value = getattr(args, field)
        if not math.isfinite(value) or value < (0 if field == 'window_start' else 2):
            parser.error(f'--{field.replace("_","-")} must be finite and >= {0 if field == "window_start" else 2}')
    if args.max_clip_seconds > 6:
        parser.error('Demo limits motion inference to 6 seconds; use the batch runner for longer experiments.')
    if args.window_seconds > 60:
        parser.error('Demo limits grounding windows to 60 seconds; use --window-start to choose a different section.')
    return args


if __name__ == '__main__':
    try:
        run_demo(parse_args())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
