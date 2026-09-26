#!/usr/bin/env python3
"""Evaluate only fully reviewed query annotations with one-to-one event matching."""
import argparse
import csv
import json
import math
from pathlib import Path


def iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / (a[1] - a[0] + b[1] - b[0] - intersection)


def valid(span, duration):
    return (isinstance(span, (list, tuple)) and len(span) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in span)
            and 0 <= span[0] < span[1] <= duration + .1)


def match_events(predictions, truth, duration, threshold):
    """Maximum-cardinality bipartite matching; each GT can be recovered at most once."""
    graph = [[j for j, gt in enumerate(truth) if valid(pred, duration) and iou(pred, gt) >= threshold]
             for pred in predictions]
    owners = {}
    def assign(i, seen):
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False
    for i in range(len(predictions)):
        assign(i, set())
    return len(owners)


def evaluate(row, label, duration, threshold):
    preds = row.get('spans', []) if row.get('status') == 'ok' else []
    gt = label['spans']
    tp = match_events(preds, gt, duration, threshold)
    precision = tp / len(preds) if preds else (0.0 if gt else None)
    recall = tp / len(gt) if gt else None
    f1 = 2 * tp / (len(preds) + len(gt)) if preds or gt else None
    return {'tp': tp, 'fp': len(preds) - tp, 'fn': len(gt) - tp,
            'event_precision': precision, 'event_recall': recall, 'event_f1': f1,
            'count_absolute_error': abs(len(preds) - len(gt)),
            'gt_count': len(gt), 'pred_count': len(preds),
            'negative_correct': (row.get('status') == 'ok' and row.get('parse_status') == 'json_empty') if not gt else None,
            'all_events_recovered': tp == len(gt) if gt else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    args = p.parse_args()
    annotation = json.loads(args.annotations.read_text())
    settings = json.loads((args.run / 'settings.json').read_text())
    if Path(annotation['video']).resolve() != Path(settings['video']).resolve():
        p.error('Annotation video and inference video differ; do not evaluate clip-relative times as global times')
    if annotation.get('video_sha256') != settings.get('video_sha256'):
        p.error('Annotation and inference video hashes differ')
    duration = settings['video_duration_s']
    if abs(annotation['duration_s'] - duration) > .2:
        p.error('Annotation and inference durations differ')
    labels = {q['id']: q for q in annotation['queries'] if q['reviewed']}
    if not labels:
        p.error('No reviewed labels. Review the entire video and mark each completed query reviewed=true first.')
    for label in labels.values():
        if not all(valid(s, duration) for s in label['spans']):
            p.error(f"Invalid ground truth span: {label['id']}")
    results_file = args.run / 'results.normalized.jsonl'
    if not results_file.exists():
        results_file = args.run / 'results.jsonl'
    rows = [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()]
    output = []
    for row in rows:
        if row['query_id'] not in labels:
            continue
        for threshold in (.3, .5, .7):
            output.append(dict(trial=row['trial'], query_id=row['query_id'], variant=row['variant'], repeat=row['repeat'],
                               status=row['status'], parse_status=row.get('parse_status'), iou_threshold=threshold,
                               **evaluate(row, labels[row['query_id']], duration, threshold)))
    if not output:
        p.error('No trial corresponds to the reviewed query IDs')
    path = args.run / 'evaluation.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(output[0]))
        writer.writeheader(); writer.writerows(output)
    (args.run / 'annotations-used.json').write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
