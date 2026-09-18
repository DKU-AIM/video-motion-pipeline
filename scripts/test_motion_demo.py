"""CPU tests: orchestration fixtures are NOT model quality/GPU validation."""
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('demo', Path(__file__).with_name('run_motion_demo.py'))
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class SpanTests(unittest.TestCase):
    def test_orders_clamps_and_caps(self):
        self.assertEqual(demo.select_span([[12, 22], [-2, 5]], 20, 4), (0.0, 4.0))
        self.assertEqual(demo.select_span([[18, 50]], 20, 6), (18.0, 20.0))

    def test_rejects_invalid_or_too_short(self):
        for spans in ([], [[4, 3]], [[0, float('nan')]], [[0, float('inf')]], [[3, 3.9]], [[30, 40]], [['x', 4]], [[True, 5]]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                demo.select_span(spans, 20, 6)

    def test_skips_short_span_for_viable_one(self):
        self.assertEqual(demo.select_span([[0, 1], [4, 8]], 20, 6), (4.0, 8.0))


class PreflightTests(unittest.TestCase):
    def test_missing_comotion_checkpoint_stops_before_python(self):
        for missing in ('comotion_detection_checkpoint.pt', 'comotion_refine_checkpoint.pt'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as root:
                base = Path(root)
                files = ['ml-comotion/demo.py', 'MotionGPT/mGPT/archs/mgpt_lm.py',
                         'ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl',
                         'ml-comotion/src/comotion_demo/data/comotion_detection_checkpoint.pt',
                         'ml-comotion/src/comotion_demo/data/comotion_refine_checkpoint.pt',
                         'assets/motiongpt_s3_h3d.tar', 'assets/reference_joints.npy',
                         'assets/reference_features.npy', 'MotionGPT/assets/meta/mean.npy',
                         'MotionGPT/assets/meta/std.npy']
                for name in files:
                    if Path(name).name != missing:
                        path = base/name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                args = demo.parse_args(['--assets-root', str(base), '--check'])
                with patch.object(demo.shutil, 'which', return_value='/fake/bin'), patch.object(demo.subprocess, 'run') as run:
                    with self.assertRaisesRegex(RuntimeError, missing):
                        demo.preflight(args)
                    run.assert_not_called()


class ReportTests(unittest.TestCase):
    def test_escapes_caption_and_preserves_track_interval(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            pose = run/'results/clip/full/comotion'
            pose.mkdir(parents=True)
            (pose/'comparison.mp4').touch()
            report = demo.write_report(run, {
                'source_video': '/video.mov', 'query': '<script>bad</script>',
                'model': 'timelens-8b', 'window_start': 30, 'window_duration': 30,
                'predicted_spans': [[3, 9]], 'clip_start': 33, 'clip_end': 39,
            }, [{
                'id': 'track1', 'caption': '<img src=x onerror=bad()>', 'track_id': 7,
                'source_parameters': str(pose/'motion_tracks.pt'),
                'source_fps': 25, 'source_frame_start': 25, 'source_frame_end': 74,
            }])
            text = report.read_text()
            self.assertNotIn('<script>bad</script>', text)
            self.assertIn('&lt;img', text)
            self.assertIn('34.000', text)
            self.assertIn('36.000', text)
            self.assertIn('Track 7', text)
            self.assertIn('results/clip/full/comotion/comparison.mp4', text)

    def test_empty_caption_is_not_a_success_report(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            pose = run/'results/clip/full/comotion'
            pose.mkdir(parents=True)
            (pose/'comparison.mp4').touch()
            with self.assertRaisesRegex(ValueError, 'empty caption'):
                demo.write_report(run, {'clip_start':0}, [{
                    'source_parameters':str(pose/'motion_tracks.pt'),
                    'source_fps':30, 'source_frame_start':0, 'source_frame_end':59, 'caption':'  ',
                }])

    def test_empty_or_external_output_fails(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            with self.assertRaises(ValueError):
                demo.write_report(run, {}, [])
            with self.assertRaises(ValueError):
                demo.write_report(run, {}, [{'source_parameters': '/outside/motion_tracks.pt'}])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root/'video.mp4'
        self.video.touch()
        self.assets = self.root/'models'
        for name in ('ml-comotion', 'MotionGPT', 'assets'):
            (self.assets/name).mkdir(parents=True)
        self.args = demo.parse_args(['--video', str(self.video), '--query', 'drink water',
                                     '--assets-root', str(self.assets), '--output', str(self.root/'run')])
        self.calls = []

    def fake_stage(self, name, cmd, run, env):
        self.calls.append(name)
        if name == 'grounding':
            (run/'grounding.jsonl').write_text(json.dumps({'model':'timelens-8b','query':'drink water','spans':[[16,20]]})+'\n')
        elif name == 'pose':
            p = run/'results/clip/full/comotion'
            p.mkdir(parents=True)
            (p/'comparison.mp4').touch()
            (p/'verified.json').write_text('{}')
        elif name == 'export':
            (run/'motion_inputs').mkdir()
            (run/'motion_inputs/manifest.json').write_text('[{"id":"track1"}]')
        elif name == 'caption':
            (run/'results_motiongpt').mkdir()
            (run/'results_motiongpt/captions.json').write_text(json.dumps([{
                'id':'track1', 'caption':'A person raises an arm.', 'track_id':1,
                'source_parameters':str(run/'results/clip/full/comotion/motion_tracks.pt'),
                'source_fps':30, 'source_frame_start':0, 'source_frame_end':119,
            }]))

    def test_pipeline_order_manifest_and_result(self):
        with patch.object(demo, 'preflight'), patch.object(demo, 'probe_duration', return_value=30), \
             patch.object(demo, 'cut_video'), patch.object(demo, 'run_stage', side_effect=self.fake_stage):
            report = demo.run_demo(self.args)
        self.assertEqual(self.calls, ['grounding', 'pose', 'export', 'caption'])
        self.assertTrue(report.is_file())
        with zipfile.ZipFile(report.parent/'review.zip') as bundle:
            self.assertIn('results/clip/full/comotion/comparison.mp4', bundle.namelist())
            self.assertNotIn('ml-comotion', bundle.namelist())
        manifest = json.loads((report.parent/'clip_manifest.json').read_text())[0]
        self.assertEqual(manifest['source_start_seconds'], 16)
        self.assertEqual(json.loads((report.parent/'status.json').read_text())['state'], 'complete')

    def test_no_grounding_span_stops_before_pose(self):
        def no_span(name, cmd, run, env):
            self.calls.append(name)
            (run/'grounding.jsonl').write_text('{"spans": []}\n')
        with patch.object(demo, 'preflight'), patch.object(demo, 'probe_duration', return_value=30), \
             patch.object(demo, 'cut_video'), patch.object(demo, 'run_stage', side_effect=no_span):
            with self.assertRaisesRegex(ValueError, 'span'):
                demo.run_demo(self.args)
        self.assertEqual(self.calls, ['grounding'])
        self.assertEqual(json.loads((self.root/'run/status.json').read_text())['state'], 'failed')
        self.assertFalse((self.root/'run/index.html').exists())

    def test_does_not_overwrite_existing_run(self):
        (self.root/'run').mkdir()
        sentinel = self.root/'run/user.txt'
        sentinel.write_text('keep')
        with patch.object(demo, 'preflight'), self.assertRaises(FileExistsError):
            demo.run_demo(self.args)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_failed_caption_leaves_logs_and_no_success_report(self):
        def fail_caption(name, cmd, run, env):
            if name == 'caption':
                raise RuntimeError('caption failed')
            self.fake_stage(name, cmd, run, env)
        with patch.object(demo, 'preflight'), patch.object(demo, 'probe_duration', return_value=30), \
             patch.object(demo, 'cut_video'), patch.object(demo, 'run_stage', side_effect=fail_caption):
            with self.assertRaisesRegex(RuntimeError, 'caption failed'):
                demo.run_demo(self.args)
        status = json.loads((self.root/'run/status.json').read_text())
        self.assertEqual(status['stage'], 'caption')
        self.assertFalse((self.root/'run/index.html').exists())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg integration requires system binaries')
class FFmpegTests(unittest.TestCase):
    def test_real_clip_can_decode_and_has_requested_duration(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            source, output = folder/'source.mp4', folder/'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=25',
                            '-t', '4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
            demo.cut_video(source, output, 1, 2.4, folder/'ffmpeg.log')
            self.assertAlmostEqual(demo.probe_duration(output), 2.4, places=1)
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], check=True)


if __name__ == '__main__':
    unittest.main()
