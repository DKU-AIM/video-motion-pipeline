import importlib.util
import json
import unittest
from pathlib import Path

PATH=Path(__file__).with_name('build_analysis.py')
spec=importlib.util.spec_from_file_location('analysis',PATH)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AnalysisTests(unittest.TestCase):
    def test_overlap_adjacent_and_union(self):
        self.assertEqual(a.union([[4,6],[0,2],[1,4],[10,12]]),[[0,6],[10,12]])
        self.assertEqual(a.length([[0,3],[2,5]]),5)
        self.assertEqual(a.intersect([[0,3],[2,5]],[[1,4]]),3)
        self.assertAlmostEqual(a.jaccard([[0,2]],[[1,3]]),1/3)
        self.assertIsNone(a.jaccard([],[]))

    def test_strict_json_rejects_recovered_formats(self):
        self.assertTrue(a.strict_json('[]'))
        self.assertTrue(a.strict_json('[[1,2],[3,4]]'))
        self.assertFalse(a.strict_json('```json\n[[1,2]]\n```'))
        self.assertFalse(a.strict_json('[[true,2]]'))
        self.assertFalse(a.strict_json('[[NaN,2]]'))

    def test_dataset_and_paired_baseline(self):
        rows=[json.loads(l) for l in PATH.with_name('response_snapshot.jsonl').read_text().splitlines()]
        d=json.loads(PATH.with_name('derived_metrics.json').read_text())
        self.assertEqual(len(rows),40)
        self.assertEqual(len({(r['model'],r['letter'],r['query_id']) for r in rows}),40)
        self.assertTrue(all(r['strict_json'] for r in rows))
        self.assertTrue(all(r['same_raw'] and r['same_grid'] and r['same_input_tokens'] for r in d['bridge']))
        for r in rows:
            self.assertAlmostEqual(a.length(r['spans']),r['union_s'])
        shot=next(r for r in rows if r['model']=='timelens-8b' and r['letter']=='C' and r['query_id']=='shot')
        self.assertEqual((shot['span_count'],shot['components'],shot['union_s']),(30,29,130))
        self.assertAlmostEqual(sum(r['wall_s'] for r in rows)/60,d['query_wall_minutes'])

if __name__=='__main__':unittest.main()
