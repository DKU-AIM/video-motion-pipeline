import importlib.util
from pathlib import Path
import unittest
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]

def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'grounding' / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

p = module('probe', 'basketball_probe.py')
e = module('evaluate', 'evaluate_basketball.py')

class BasketballTests(unittest.TestCase):
    def test_token_progress_excludes_prompt(self):
        class Tokens:
            def __init__(self, n): self.n = n
            def numel(self): return self.n
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.jsonl'
            stream = p.TokenProgress(path, 1)
            stream.put(Tokens(10000))
            stream.put(Tokens(1))
            stream.put(Tokens(1))
            stream.end()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['generated_tokens'], 0)
            self.assertEqual(rows[-1]['generated_tokens'], 2)
            self.assertEqual(rows[-1]['event'], 'generation_end')

    def test_multiple_answers_and_thinking(self):
        self.assertEqual(p.extract_spans('<think>2 to 8</think><answer>10 to 12, 20 to 24</answer>')[0], [[10.,12.],[20.,24.]])
        self.assertEqual(p.extract_spans('<think>2 to 8')[1], 'unparsed')
        self.assertEqual(p.extract_spans('```json\n[[1,2],[4,6]]\n```')[0], [[1,2],[4,6]])

    def test_noncanonical_multi_span_formats(self):
        self.assertEqual(p.extract_spans('[1,2], [4,6]')[0], [[1.,2.],[4.,6.]])
        self.assertEqual(p.extract_spans('[{"start_seconds":"1.5","end_seconds":"3"}]'), ([[1.5,3.]], 'json_objects'))
        self.assertEqual(p.extract_spans('[[2:15, 2:17], [3:29, 3:31]]')[0], [[135.,137.],[209.,211.]])
        spans, status = p.extract_spans('[[1,2], [4,6], [8')
        self.assertEqual(spans, [[1.,2.],[4.,6.]])
        self.assertTrue(status.startswith('partial_'))

    def test_output_limit_respects_eos(self):
        self.assertFalse(p.hit_generation_limit(2048, 2048, 2, [1,2]))
        self.assertTrue(p.hit_generation_limit(2048, 2048, 7, [1,2]))
        self.assertFalse(p.hit_generation_limit(100, 2048, 7, None))

    def test_empty_not_parse_error(self):
        self.assertEqual(p.extract_spans('[]'), ([], 'json_empty'))
        self.assertEqual(p.extract_spans('I cannot determine this.'), ([], 'unparsed'))

    def test_invalid_spans(self):
        self.assertFalse(p.valid_span([-1,3], 20))
        self.assertFalse(p.valid_span([2,30], 20))
        self.assertFalse(p.valid_span([3,3], 20))

    def test_duplicates_do_not_improve_recall(self):
        label = {'spans': [[1,3],[10,12]]}
        row = {'status': 'ok','spans': [[1,3],[1,3]],'parse_status':'json'}
        result = e.evaluate(row, label, 20, .5)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1,1,1))
        self.assertEqual(result['event_recall'], .5)

    def test_bipartite_matching_can_reassign(self):
        # The broad first prediction can match either GT; second can match only first.
        self.assertEqual(e.match_events([[0,10],[0,4]], [[0,4],[6,10]], 20, .3), 2)

    def test_no_target_requires_successful_explicit_empty(self):
        for status, parsed, expected in [('ok','json_empty',True),('ok','unparsed',False),('error','json_empty',False)]:
            result = e.evaluate({'status': status, 'parse_status': parsed, 'spans': []}, {'spans': []}, 20, .5)
            self.assertEqual(result['negative_correct'], expected)

if __name__ == '__main__':
    unittest.main()
