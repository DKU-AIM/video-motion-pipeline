#!/usr/bin/env python3
"""Reparse saved responses without inference; preserve original results.jsonl."""
import argparse
import json
from pathlib import Path
from basketball_probe import extract_spans, valid_span
from basketball_batch import summarize


def normalize(out):
    plan = json.loads((out/'plan.json').read_text())
    state = json.loads((out/'state.json').read_text())
    for job, entry in state.items():
        if entry['status'] == 'running':
            raise RuntimeError('Normalize after the batch stops to keep reports consistent')
        directory = Path(entry['attempt_dir'])
        source = directory/'results.jsonl'
        if not source.exists():
            continue
        duration = json.loads((directory/'settings.json').read_text())['video_duration_s']
        rows = []
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row['status'] == 'ok':
                row['original_spans'] = row.get('spans')
                row['original_parse_status'] = row.get('parse_status')
                row['spans'], row['parse_status'] = extract_spans(row['raw'])
                row['span_count'] = len(row['spans'])
                row['invalid_span_count'] = sum(not valid_span(s,duration) for s in row['spans'])
                row['incomplete_output'] = bool(row.get('hit_output_limit') or row['parse_status'].startswith('partial_'))
                row['normalization_version'] = 2
            rows.append(row)
        tmp = directory/'results.normalized.tmp'
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        tmp.replace(directory/'results.normalized.jsonl')
        print(job, [(r['query_id'],r.get('span_count'),r.get('parse_status')) for r in rows])
    summarize(out,plan,state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    normalize(parser.parse_args().run)
