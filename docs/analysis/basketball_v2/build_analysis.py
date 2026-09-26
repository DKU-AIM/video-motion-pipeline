#!/usr/bin/env python3
"""Rebuild presentation tables/plots from the completed v2 records; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=Path.home()/'aim-workspace/results_grounding/basketball_v2')
    args=p.parse_args()
    out=Path(__file__).resolve().parent
    root=args.run.resolve()
    plan=json.loads((root/'plan.json').read_text())
    state=json.loads((root/'state.json').read_text())
    rows=[]
    snapshots={}
    for job in plan['jobs']:
        directory=Path(state[job['id']]['attempt_dir'])
        settings=json.loads((directory/'settings.json').read_text())
        duration=settings['video_duration_s']
        load=json.loads((directory/'load.json').read_text())
        snapshots[job['model']]={'repository':settings['model_repo'],'model_commit':load.get('model_commit')}
        for line in (directory/'results.jsonl').read_text().splitlines():
            row=json.loads(line)
            row.update(model=job['model'],condition=job['condition'])
            rows.append(row)
    assert len(rows)==48 and all(r['status']=='ok' for r in rows)
    models=list(dict.fromkeys(r['model'] for r in rows))
    summaries=[]
    def union(spans):
        merged=[]
        for a,b in sorted(spans):
            if not 0<=a<b<=duration+.1:
                continue
            if merged and a<=merged[-1][1]+1e-8:
                merged[-1][1]=max(b,merged[-1][1])
            else:
                merged.append([a,b])
        return merged
    for model in models:
        selected=[r for r in rows if r['model']==model and r['condition']=='all_json_5q']
        shot=next(r for r in selected if r['query_id']=='shot')
        default=next(r for r in rows if r['model']==model and r['condition']=='default_shot_control')
        merged=union(shot['spans'])
        summary={'model':model, **{r['query_id']+'_count':len(r['spans']) for r in selected},
                 'mean_wall_s':statistics.mean(r['wall_s'] for r in selected),
                 'min_wall_s':min(r['wall_s'] for r in selected),'max_wall_s':max(r['wall_s'] for r in selected),
                 'mean_preprocess_s':statistics.mean(r['preprocess_s'] for r in selected),
                 'mean_generate_s':statistics.mean(r['generate_s'] for r in selected),
                 'peak_allocated_gib':max(r['peak_allocated_gib'] for r in selected),
                 'input_tokens_min':min(r['input_tokens'] for r in selected),
                 'input_tokens_max':max(r['input_tokens'] for r in selected),
                 'incomplete_count':sum(r['incomplete_output'] for r in selected),
                 'invalid_span_count':sum(r['invalid_span_count'] for r in selected),
                 'default_shot_count':len(default['spans']),
                 'legacy_shot_count':shot['legacy_span_count'],
                 'shot_connected_components':len(merged),
                 'shot_union_s':sum(b-a for a,b in merged),
                 'shot_union_percent':sum(b-a for a,b in merged)/duration*100}
        summaries.append(summary)
    with (out/'model_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summaries[0]));w.writeheader();w.writerows(summaries)
    (out/'response_snapshot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    derived={'source_run':str(root),'duration_s':duration,
             'wall_minutes':(max(s['finished_unix_s'] for s in state.values())-min(s['started_unix_s'] for s in state.values()))/60,
             'query_wall_minutes':sum(r['wall_s'] for r in rows)/60,
             'models':snapshots,'summary':summaries,
             'source_plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()}
    (out/'derived_metrics.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(12,5.4))
    ax.axvspan(duration,405,color='#fff0ed')
    for y,model in enumerate(models):
        r=next(r for r in rows if r['model']==model and r['condition']=='all_json_5q' and r['query_id']=='shot')
        for a,b in r['spans']:
            ax.barh(y,b-a,left=a,height=.56,color='#b8462e' if b>duration else '#3e729c',edgecolor='white',linewidth=.35)
    ax.axvline(duration,color='#b8462e',linestyle='--',linewidth=1)
    ax.text(duration+3,-.6,f'Video end: {duration:.2f}s',color='#b8462e',fontsize=9)
    ax.set_yticks(range(len(models)),models);ax.invert_yaxis()
    ax.set_xlim(0,405);ax.set_xlabel('Time from start of uploaded video (seconds)')
    ax.set_title('Shot query: raw predicted intervals (NOT ground truth)')
    fig.tight_layout();fig.savefig(out/'shot_timeline.png',dpi=180);plt.close(fig)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={'width_ratios':[2,1]})
    ys=list(range(len(models)))
    pre=[s['mean_preprocess_s'] for s in summaries];gen=[s['mean_generate_s'] for s in summaries]
    ax.barh(ys,pre,label='Preprocessing',color='#97b9d2')
    ax.barh(ys,gen,left=pre,label='Generation',color='#3e729c')
    ax.set_yticks(ys,models);ax.invert_yaxis();ax.set_xlabel('Mean seconds per query (5 queries)');ax.legend(loc='lower right')
    bx.barh(ys,[s['peak_allocated_gib'] for s in summaries],color='#63866a')
    bx.set_yticks(ys,[]);bx.invert_yaxis();bx.set_xlabel('Peak allocated GPU memory (GiB)')
    fig.suptitle('All-json condition: processing time and GPU memory')
    fig.tight_layout();fig.savefig(out/'time_memory.png',dpi=180);plt.close(fig)
    print('Wrote',out)


if __name__=='__main__':
    main()
