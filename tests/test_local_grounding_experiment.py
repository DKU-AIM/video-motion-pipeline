import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT / 'grounding/local_grounding_experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class ExperimentTest(unittest.TestCase):
    def test_union_and_invalid_predictions(self):
        self.assertEqual(e.score([[4, 10], [8, 14]], [4, 14], 20)['tiou'], 1)
        self.assertAlmostEqual(e.score([[0, 2], [4, 14]], [4, 14], 20)['tiou'], 10/12)
        for spans in ([], [[-1, 10]], [[4, 30]], [[10, 4]], [[float('nan'), 12]]):
            self.assertFalse(e.score(spans, [4, 14], 20)['valid'])

    def test_complete_and_failed_runs_are_reported(self):
        class Grounder:
            def __init__(self, model, **kwargs):
                self.model = model
                if model == 'bad':
                    raise RuntimeError('simulated load failure')
            def __call__(self, *args, **kwargs):
                return {'spans': [[4, 14]], 'raw': '4 to 14'}
            def free(self):
                pass
        cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda: None,
                                     synchronize=lambda: None, empty_cache=lambda: None)
        modules = {'torch': types.SimpleNamespace(cuda=cuda),
                   'vtg_run': types.SimpleNamespace(preflight=lambda: None, Grounder=Grounder)}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), patch.object(e, 'command', return_value='test'):
            out = Path(temp)
            manifest = {'models': ['good', 'bad'], 'queries': ['dance'], 'settings': {'total_tokens': 8192},
                        'cases': [{'id': 'early', 'path': 'test.mp4', 'target': [4, 14], 'duration': 20}]}
            self.assertEqual(e.run(out, manifest), 1)
            rows = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['tiou'], 1)
            self.assertEqual(rows[1]['status'], 'error')
            self.assertIn('| bad | 0/1', (out / 'report.md').read_text())


if __name__ == '__main__':
    unittest.main()
