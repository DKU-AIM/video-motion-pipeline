#!/usr/bin/env python3
"""Ground-truth-free audit. Diagnostics below must not be presented as accuracy."""
import json
import math
from pathlib import Path
import hashlib
import csv

BASE=Path(__file__).resolve().parent

def strict_pairs(raw):
    try:
        obj=json.loads(raw.strip())
    except (ValueError,TypeError):
        return False
    return isinstance(obj,list) and all(isinstance(p,list) and len(p)==2 and
        all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in p) for p in obj)


def union(spans, duration, gap=0):
    merged=[]
    for a,b in sorted(spans):
        if not all(math.isfinite(x) for x in (a,b)) or not 0<=a<b<=duration:
            continue
        if merged and a<=merged[-1][1]+gap+1e-8:
            merged[-1][1]=max(b,merged[-1][1])
        else:
            merged.append([a,b])
    return merged


def length(spans):
    return sum(b-a for a,b in spans)


def intersection(a,b):
    return sum(max(0,min(y,v)-max(x,u)) for x,y in a for u,v in b)


def audit(rows,duration):
    output=[]
    for model in dict.fromkeys(r['model'] for r in rows):
        selected=[r for r in rows if r['model']==model and r['condition']=='all_json_5q']
        byquery={r['query_id']:r for r in selected}
        general=union(byquery['shot']['spans'],duration)
        attribute=union(byquery['dark_shot']['spans'],duration)
        shot_count=len(byquery['shot']['spans'])
        output.append({'model':model,
            'strict_format_all5':sum(strict_pairs(r['raw']) for r in selected),
            'strict_format_nonnegative4':sum(strict_pairs(r['raw']) for r in selected if r['query_id']!='football'),
            'recoverable_all5':sum(r['parse_status']!='unparsed' for r in selected),
            'incomplete_all5':sum(r['incomplete_output'] for r in selected),
            'shot_raw_count':shot_count,
            'shot_components_gap0':len(general),
            'shot_components_gap05':len(union(byquery['shot']['spans'],duration,.5)),
            'shot_components_gap1':len(union(byquery['shot']['spans'],duration,1)),
            'shot_completeness':not byquery['shot']['incomplete_output'],
            'shot_invalid_count':byquery['shot']['invalid_span_count'],
            'attribute_outside_general_percent':100*(1-intersection(general,attribute)/length(attribute)) if attribute else None,
            'attribute_diagnostic_usable':not any(byquery[q]['incomplete_output'] or byquery[q]['invalid_span_count'] for q in ['shot','dark_shot'])})
    return output


def main():
    metrics=json.loads((BASE/'derived_metrics.json').read_text())
    rows=[json.loads(l) for l in (BASE/'response_snapshot.jsonl').read_text().splitlines()]
    results=audit(rows,metrics['duration_s'])
    with (BASE/'critical_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(results[0]));writer.writeheader();writer.writerows(results)
    plan=json.loads((Path(metrics['source_run'])/'plan.json').read_text())
    repo=BASE.parents[2]
    provenance={name:{'at_run':sha,'current':hashlib.sha256((repo/name).read_bytes()).hexdigest(),
                     'matches':hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha} for name,sha in plan['code_sha256'].items()}
    payload={'definitions':{'strict_format':'Bare JSON array of finite numeric [start,end] arrays. [] passes syntax only; code fences/objects/partial arrays do not.',
         'gap_components':'Counts on valid in-video intervals; gaps <= threshold merge. Not GT events; gap>0 count does not define temporal coverage.',
         'attribute_outside_general':'Length(attribute_union minus shot_union)/length(attribute_union); descriptive inconsistency, NOT semantic error rate.'},
         'diagnostics':results,'current_source_hash_check':provenance,
         'video_sha256_at_run':plan['video_sha256'],
         'strict_total':sum(x['strict_format_all5'] for x in results),
         'strict_nonnegative':sum(x['strict_format_nonnegative4'] for x in results)}
    (BASE/'critical_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for r in results:
        print(r)
    print('STRICT',payload['strict_total'],'/40; excluding football',payload['strict_nonnegative'],'/32')
    print('SOURCE_HASHES',provenance)


if __name__=='__main__':
    main()
