import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch', ROOT / 'grounding/basketball_batch.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BatchTests(unittest.TestCase):
    def test_full_matrix(self):
        jobs = b.matrix(b.MODELS)
        self.assertEqual(len(jobs), 72)
        self.assertEqual(len({j['id'] for j in jobs}), 72)
        self.assertEqual(sum(j['repeat'] * 16 for j in jobs), 1664)
        self.assertEqual({j['model'] for j in jobs[:8]}, set(b.MODELS))

    def test_quick_matrix_and_command(self):
        jobs = b.quick_matrix(b.MODELS)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(sum(j['repeat'] * 4 for j in jobs), 32)
        for job in jobs:
            self.assertEqual(job['prompt'], 'all-json')
            self.assertEqual(job['variants'], 'first')
        cmd = b.command(jobs[0], 'python', 'video', 'queries', ['shot','bench','closeup','football'], 'out')
        self.assertEqual(cmd[cmd.index('--variants') + 1], 'first')
        wrapper = (ROOT / 'scripts/run_basketball_quick.sh').read_text()
        self.assertIn('--timeout-hours 0 --deadline-minutes 0', wrapper)

    def test_v2_query_scope(self):
        jobs = b.v2_matrix(b.MODELS)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(sum(len(j['query_ids']) for j in jobs), 48)
        self.assertTrue(all(len(j['query_ids']) == 5 for j in jobs[:8]))
        self.assertTrue(all(j['query_ids'] == ['shot'] for j in jobs[8:]))
        self.assertTrue(all(j['max_new_tokens'] == 2048 for j in jobs))
        cmd = b.command(jobs[-1], 'python', 'video', 'queries', ['shot','bench'], 'out')
        self.assertEqual(cmd[cmd.index('--ids')+1:cmd.index('--variants')], ['shot'])

    def test_parameter_matrix_and_report(self):
        models = ['timelens2-4b','timelens-8b']
        jobs = b.parameter_matrix(models)
        self.assertEqual(len(jobs), 8)
        self.assertEqual(sum(len(j['query_ids']) for j in jobs), 40)
        self.assertEqual({j['model'] for j in jobs}, set(models))
        for model in models:
            selected = [j for j in jobs if j['model'] == model]
            self.assertEqual([(j['fps'],j['total_tokens']) for j in selected],
                             [(.5,32768),(1,32768),(2,32768),(1,16384)])
            self.assertTrue(all(j['max_frames'] == 512 and j['max_new_tokens'] == 2048 for j in selected))
        for j in jobs: j['expected_trials'] = 5
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            b.summarize(out, {'suite':'parameters','jobs':jobs,'planned_trials':40}, {})
            self.assertIn('A_fps05_tokens32768', (out/'parameter_summary.md').read_text())
        wrapper = (ROOT/'scripts/run_basketball_parameters.sh').read_text()
        self.assertIn('--timeout-hours 0 --deadline-minutes 0', wrapper)

    def test_failure_resume_retry_with_real_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'grounding').mkdir()
            (root / 'experiments/basketball').mkdir(parents=True)
            for name in ['basketball_probe.py','vtg_run.py','basketball_batch.py']:
                (root / 'grounding' / name).write_text('test code')
            (root / 'experiments/basketball/queries.json').write_text(json.dumps([{'id':'shot','queries':['q1','q2']}]))
            video = root / "inputs/grounding/2026_AsianGames_men's_basketball_final.mp4"
            video.parent.mkdir(parents=True); video.write_bytes(b'fixture')
            worker = root / 'worker.py'
            worker.write_text('''import json, sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True)
failed = 'native' in str(out) and out.name == 'attempt_001'
rows = [{'trial':i,'status':'error' if failed else 'ok'} for i in range(2)]
(out/'results.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
raise SystemExit(1 if failed else 0)
''')
            jobs = b.matrix(['timelens2-4b'], repeats=1)[:2]
            def command(job, python, video, queries, ids, out):
                return [sys.executable, str(worker), str(out)]
            args = ['batch', '--workspace', str(root), '--baseline-repeats', '1', '--models','timelens2-4b']
            with patch.object(b, 'ROOT', root), patch.object(b, 'matrix', return_value=jobs), patch.object(b, 'command', side_effect=command), patch.dict(os.environ, {'GROUNDING_PYTHON':sys.executable}), contextlib.redirect_stdout(io.StringIO()):
                # Short poll for tiny fixture subprocesses (no GPU/model involved).
                original_sleep = b.time.sleep
                with patch.object(b.time, 'sleep', side_effect=lambda _: original_sleep(.02)):
                    with patch.object(sys, 'argv', args):
                        self.assertEqual(b.main(), 1)
                        self.assertEqual(b.main(), 1)  # Completed/failed jobs are not rerun by default.
                    with patch.object(sys, 'argv', args + ['--retry-failed']):
                        self.assertEqual(b.main(), 0)
            out = root / 'results_grounding/basketball_all'
            state = json.loads((out/'state.json').read_text())
            self.assertEqual(state['timelens2-4b__native']['attempt'], 2)
            self.assertEqual(state['timelens2-4b__all_json']['attempt'], 1)
            self.assertTrue((out/'all_trials.csv').is_file())
            self.assertTrue((out/'timelens2-4b__native/attempt_001/results.jsonl').is_file())

if __name__ == '__main__':
    unittest.main()
