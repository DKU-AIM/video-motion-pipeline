"""Author 4D-Humans + PHALP, unchanged association, observation-only export."""
import argparse
from collections import Counter
import csv
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
for key, value in {'bool':bool, 'int':int, 'float':float, 'complex':complex,
                   'object':object, 'str':str, 'unicode':str}.items():
    if key not in np.__dict__: setattr(np, key, value)
if not hasattr(inspect, 'getargspec'): inspect.getargspec = inspect.getfullargspec
sys.path.insert(0, str(Path(os.environ['AIM_4DHUMANS_DIR']).expanduser().resolve()))
from track import HMR2_4dhuman, Human4DConfig
from omegaconf import OmegaConf
import cv2
import torch

p = argparse.ArgumentParser()
p.add_argument('--id', required=True)
p.add_argument('--video', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--frames', type=int, default=0)
args = p.parse_args()
video = args.video.resolve()
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out/'PASS.json').exists(), 'Already complete; no duplicate execution'
cap = cv2.VideoCapture(str(video))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
assert n > 0 and w > 0 and h > 0 and fps > 0
expected = min(n, args.frames) if args.frames else n
# Observe progress without changing the author's iterator or association logic.
native_module = importlib.import_module('phalp.trackers.PHALP')
native_progress = native_module.progress_bar
def logged_progress(iterable, *a, **kw):
    for item in native_progress(iterable, *a, **kw):
        yield item
        completed = int(item[0])+1
        if completed % 60 == 0 or completed == expected:
            (out/'progress.json').write_text(json.dumps({'processed_frames':completed,'total_frames':expected,'elapsed_seconds':time.monotonic()-start}))
native_module.progress_bar = logged_progress
cfg = OmegaConf.structured(Human4DConfig())
cfg.video.source = str(video); cfg.video.output_dir = str(out)
cfg.video.start_frame = 0; cfg.video.end_frame = args.frames or -1
cfg.render.enable = False; cfg.detect_shots = False; cfg.use_gt = False
cfg.overwrite = True
OmegaConf.save(cfg, out/'config.yaml')
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); torch.set_num_threads(4)
start = time.monotonic()
tracker = HMR2_4dhuman(cfg)
with torch.inference_mode(): result = tracker.track()
assert result is not None, 'Author tracker caught an error; inspect run log'
data, pkl_path = result
assert len(data) == expected, (len(data), expected)
frames = sorted(data.values(), key=lambda x:int(x['time']))
assert [int(x['time']) for x in frames] == list(range(expected))
counts = Counter(); stale = Counter(); exported = []; rows = []
for f in frames:
    t = int(f['time']); observations = []; seen = set()
    assert len(f['tid']) == len(f['tracked_time']) == len(f['bbox']) == len(f['smpl'])
    for ident, age, box, smpl in zip(f['tid'], f['tracked_time'], f['bbox'], f['smpl']):
        ident = int(ident)
        if age != 0: stale[ident] += 1; continue
        assert ident not in seen; seen.add(ident)
        xywh = [float(v) for v in box]
        assert len(xywh) == 4 and np.isfinite(xywh).all() and min(xywh[2:]) > 0
        assert all(np.isfinite(np.asarray(v)).all() for v in smpl.values())
        observations.append({'id':ident, 'bbox_xywh':xywh})
        rows.append([t, t/fps, ident, *xywh]); counts[ident] += 1
    exported.append({'frame':t, 'time_seconds':t/fps, 'observations':observations})
assert rows
with (out/'observed_tracks.csv').open('w') as f:
    wr = csv.writer(f); wr.writerow(['frame_zero_based','time_seconds','track_id','x','y','width','height']); wr.writerows(rows)
(out/'observed_tracks.json').write_text(json.dumps({'coordinate_convention':'0-based native input pixel xywh; observed associations only, no stale/predicted boxes', 'width':w,'height':h,'fps':fps,'frames':exported}, separators=(',',':')))
report = {'id':args.id, 'pipeline':'ViTDet-H + HMR 2.0 + PHALP', 'device':torch.cuda.get_device_name(0),
          'width':w,'height':h,'fps':fps,'frames':expected,'observed_instances':len(rows),
          'unique_observed_ids':len(counts),'observed_frames_by_id':dict(counts),
          'stale_prediction_frames_by_id':dict(stale),'elapsed_seconds':time.monotonic()-start,
          'native_pickle':str(pkl_path),'input':str(video),'input_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
          'notes':['Native model input preprocessing retained, including detector resizing and HMR person crops.',
                   'Input resolution and FPS preserved, no inference frame subsampling.',
                   'Every frame processed. No manual ID assignment or identity correction.',
                   'Counts and finite geometry are execution checks, not tracking accuracy.',
                   'Native model also detects referees, bench and spectators; IDs are not a player count.']}
(out/'PASS.json').write_text(json.dumps(report, indent=2))
print('TRACKING_PASS', json.dumps(report), flush=True)
