#!/usr/bin/env python3
"""Isolated CLI coverage for the CoMotion-only mesh recovery path."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'mesh_recovery' / 'run_pose_batch.py'


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fake_imports(root):
    write(root / 'numpy.py', '')
    write(root / 'torch.py', 'def set_num_threads(value): pass\n')
    write(root / 'PIL' / '__init__.py', 'class Image: pass\nclass ImageDraw: pass\n')
    write(root / 'smpl_eval' / '__init__.py', '')
    write(root / 'smpl_eval' / 'meshrender.py', 'class MeshRenderer: pass\ndef track_color(value): pass\ndef draw_labels(*args): pass\n')
    write(root / 'smpl_eval' / 'overlay.py', 'def _font(value): pass\n')
    write(root / 'comotion_demo' / '__init__.py', '')
    write(root / 'comotion_demo' / 'utils' / '__init__.py', '')
    write(root / 'comotion_demo' / 'utils' / 'smpl_kinematics.py', '')


class PoseDemoOptionsTest(unittest.TestCase):
    def run_runner(self, workspace, fake_modules, *args, fake_bin=None):
        env = os.environ | {
            'MOTION_WORKSPACE': str(workspace),
            'PYTHONPATH': str(fake_modules),
        }
        if fake_bin:
            env['PATH'] = f'{fake_bin}:{env["PATH"]}'
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

    def test_help_does_not_require_model_dependencies(self):
        completed = subprocess.run(
            [sys.executable, str(RUNNER), '--help'],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--models', completed.stdout)
        self.assertIn('--conditions', completed.stdout)

    def test_comotion_only_runs_without_multihmr2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_modules = root / 'fake_modules'
            fake_imports(fake_modules)
            workspace = root / 'workspace'
            workspace.mkdir()
            (workspace / 'clip_manifest.json').write_text('[]')

            completed = self.run_runner(workspace, fake_modules, '--models', 'comotion', '--conditions', 'full')

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((workspace / 'results' / 'status.json').exists())

    def test_captured_comotion_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_modules = root / 'fake_modules'
            fake_imports(fake_modules)
            workspace = root / 'workspace'
            input_dir = workspace / 'inputs' / 'demo'
            input_dir.mkdir(parents=True)
            (input_dir / 'full.mp4').touch()
            (workspace / 'clip_manifest.json').write_text(json.dumps([{
                'id': 'demo', 'source_start_seconds': 0, 'category': 'test', 'roi_xywh': [0, 0, 1, 1],
            }]))
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            ffprobe = fake_bin / 'ffprobe'
            ffprobe.write_text('#!/bin/sh\nprintf \'{"streams":[{"avg_frame_rate":"1/1","nb_frames":"1","width":1,"height":1}]}\'\n')
            ffmpeg = fake_bin / 'ffmpeg'
            ffmpeg.write_text('#!/bin/sh\nexit 1\n')
            ffprobe.chmod(ffprobe.stat().st_mode | stat.S_IXUSR)
            ffmpeg.chmod(ffmpeg.stat().st_mode | stat.S_IXUSR)

            completed = self.run_runner(workspace, fake_modules, '--models', 'comotion', '--conditions', 'full', fake_bin=fake_bin)

            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertTrue((workspace / 'results' / 'demo' / 'full' / 'comotion' / 'error.json').exists())


if __name__ == '__main__':
    unittest.main()
