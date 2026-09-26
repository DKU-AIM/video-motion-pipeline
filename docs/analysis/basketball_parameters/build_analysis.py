#!/usr/bin/env python3
"""Rebuild experiment 2 analysis from immutable completed records, without inference."""
import argparse
import base64
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DOCS = ROOT/'docs'
MODELS = ['timelens2-4b','timelens-8b']
NAMES = {'timelens2-4b':'TimeLens2-4B','timelens-8b':'TimeLens-8B'}
QUERIES = ['shot','dark_shot','bench','closeup','football']
QLABEL = {'shot':'슛','dark_shot':'어두운 유니폼 슛','bench':'벤치 환호','closeup':'얼굴 클로즈업','football':'축구 골 후보'}
CNAMES = ['A','B','C','D']
COLORS = ['#2274a5','#008577','#d28a27','#865ba6']

def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def union(spans, gap=0):
    merged=[]
    for a,b in sorted(spans):
        if merged and a<=merged[-1][1]+gap:
            merged[-1][1]=max(b,merged[-1][1])
        else:
            merged.append([a,b])
    return merged

def length(spans):
    return sum(b-a for a,b in union(spans))

def intersect(a,b):
    return sum(max(0,min(y,v)-max(x,u)) for x,y in union(a) for u,v in union(b))

def jaccard(a,b):
    i=intersect(a,b);u=length(a)+length(b)-i
    return i/u if u else None

def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |\n' for row in rows)+'\n'

def csv_write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def strict_json(raw):
    try:
        a=json.loads(raw)
        return isinstance(a,list) and all(isinstance(p,list) and len(p)==2 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in p) for p in a)
    except (ValueError,TypeError):
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path.home()/'aim-workspace/results_grounding/basketball_parameters')
    parser.add_argument('--experiment1',type=Path,default=Path.home()/'aim-workspace/results_grounding/basketball_v2')
    args=parser.parse_args();run=args.run.resolve()
    plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'state.json').read_text())
    assert len(plan['jobs'])==8 and plan['planned_trials']==40
    assert all(x['status']=='success' for x in state.values()), 'Incomplete experiment'
    source=OUT/'source';source.mkdir(exist_ok=True)
    query_source = ROOT/'experiments/basketball/queries.json'
    assert sha(query_source) == plan['queries_sha256'], 'Queries changed since run'
    shutil.copy2(query_source,source/'queries.json')

    provenance={};rows=[];sessions=[];settings_by_job={};load_by_job={};resources=[]
    for filename in ['plan.json','state.json']:
        shutil.copy2(run/filename,source/filename);provenance[filename]=sha(run/filename)
    for job in plan['jobs']:
        path=Path(state[job['id']]['attempt_dir']);dest=source/job['id'];dest.mkdir(exist_ok=True)
        for name in ['results.jsonl','settings.json','load.json','prompt-template.txt','pip-freeze.txt','gpu.txt','code-commit.txt','resources.jsonl','generation_progress.jsonl']:
            if (path/name).exists():
                shutil.copy2(path/name,dest/name);provenance[f"{job['id']}/{name}"]=sha(path/name)
        settings=json.loads((path/'settings.json').read_text());settings_by_job[job['id']]=settings
        load=json.loads((path/'load.json').read_text());load_by_job[job['id']]=load
        config_path = (Path.home()/'.cache/huggingface/hub'/('models--'+settings['model_repo'].replace('/','--'))/'snapshots'/load['model_commit']/'config.json')
        if not config_path.exists():
            config_path = dest/'model-config.json'
        config = json.loads(config_path.read_text())
        vision = config['vision_config']
        assert (vision['temporal_patch_size'],vision['patch_size'],vision['spatial_merge_size']) == (2,16,2)
        if config_path != dest/'model-config.json':
            shutil.copy2(config_path,dest/'model-config.json')
        provenance[f"{job['id']}/model-config.json"] = sha(dest/'model-config.json')

        duration=settings['video_duration_s']
        for line in (path/'results.jsonl').read_text().splitlines():
            r=json.loads(line);r.update(model=job['model'],condition=job['condition'],letter=job['condition'][0],
                                     fps=job['fps'],budget=job['total_tokens'],job_id=job['id'])
            assert r['status']=='ok'
            assert all(0<=a<b<=duration for a,b in r['spans'])
            t,h,w=r['video_grid_thw'][0]
            r.update(union_s=length(r['spans']),coverage_pct=100*length(r['spans'])/duration,
                     components=len(union(r['spans'])),strict_json=strict_json(r['raw']),
                     grid_temporal=t,grid_h=h,grid_w=w,
                     grid_frames=2*t,derived_height=16*h,derived_width=16*w,
                     derived_visual_tokens=t*h*w//4)
            rows.append(r)
        res=[json.loads(x) for x in (path/'resources.jsonl').read_text().splitlines()]
        for r in res:r.update(job_id=job['id'],model=job['model'],letter=job['condition'][0])
        resources.extend(res)
        trial_res=[r for r in res if r['phase'].startswith('trial_')]
        sessions.append(dict(model=job['model'],letter=job['condition'][0],job_id=job['id'],
                             session_s=state[job['id']]['elapsed_s'],load_s=load['seconds'],
                             samples=len(res),gpu_query_failures=sum(r.get('gpu_rows') is None for r in res),
                             peak_sampled_rss_gib=max(r.get('process_rss_bytes') or 0 for r in trial_res)/1024**3,
                             gpu_util_numeric_samples=sum(any(len(v)>1 and re.fullmatch(r'\s*\d+(?:\.\d+)?\s*',v[1]) for v in r.get('gpu_rows') or []) for r in res)))
    assert len(rows)==40 and len({(r['model'],r['letter'],r['query_id']) for r in rows})==40
    (OUT/'response_snapshot.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    by={(r['model'],r['letter'],r['query_id']):r for r in rows}
    summary=[]
    for c in CNAMES:
        for m in MODELS:
            rr=[by[m,c,q] for q in QUERIES];ss=next(s for s in sessions if s['model']==m and s['letter']==c)
            summary.append(dict(model=m,condition=c,**{q+'_count':by[m,c,q]['span_count'] for q in QUERIES},
                                mean_wall_s=st.mean(r['wall_s'] for r in rr),mean_preprocess_s=st.mean(r['preprocess_s'] for r in rr),
                                mean_generate_s=st.mean(r['generate_s'] for r in rr),median_wall_s=st.median(r['wall_s'] for r in rr),
                                mean_positive_wall_s=st.mean(r['wall_s'] for r in rr if r['query_id']!='football'),
                                mean_nonfirst_wall_s=st.mean(r['wall_s'] for r in rr if not r['cold_query']),
                                min_wall_s=min(r['wall_s'] for r in rr),max_wall_s=max(r['wall_s'] for r in rr),
                                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rr),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rr),
                                mean_input_tokens=st.mean(r['input_tokens'] for r in rr),mean_output_tokens=st.mean(r['output_tokens'] for r in rr),
                                peak_sampled_rss_gib=ss['peak_sampled_rss_gib'],session_s=ss['session_s'],load_s=ss['load_s']))
    sm={(r['model'],r['condition']):r for r in summary}
    csv_write('condition_summary.csv',summary)
    qmetrics=[{k:r[k] for k in ['model','letter','query_id','span_count','components','union_s','coverage_pct','wall_s','preprocess_s','generate_s','input_tokens','output_tokens','grid_frames','derived_width','derived_height','derived_visual_tokens','peak_allocated_gib','peak_reserved_gib','strict_json','incomplete_output']} for r in rows]
    csv_write('query_metrics.csv',qmetrics);csv_write('session_resources.csv',sessions)
    agreement=[];attributes=[];ratios=[]
    for m in MODELS:
        for c in CNAMES:
            a=by[m,c,'shot']['spans'];b=by[m,c,'dark_shot']['spans']
            attributes.append(dict(model=m,condition=c,dark_outside_shot_pct=100*(length(b)-intersect(a,b))/length(b) if b else None,
                                   shot_dark_jaccard=jaccard(a,b)))
        for left,right in [('A','B'),('B','C'),('D','B')]:
            a,b=sm[m,left],sm[m,right]
            ratios.append(dict(model=m,comparison=left+'→'+right,
                               wall_change_pct=100*(b['mean_wall_s']/a['mean_wall_s']-1),
                               preprocess_delta_s=b['mean_preprocess_s']-a['mean_preprocess_s'],
                               generation_delta_s=b['mean_generate_s']-a['mean_generate_s'],
                               memory_delta_gib=b['peak_allocated_gib']-a['peak_allocated_gib']))
            for q in QUERIES[:-1]:
                agreement.append(dict(model=m,comparison=left+'→'+right,query=q,
                                      temporal_jaccard=jaccard(by[m,left,q]['spans'],by[m,right,q]['spans'])))
    csv_write('prediction_agreement.csv',agreement);csv_write('attribute_consistency.csv',attributes);csv_write('paired_cost_changes.csv',ratios)
    # Use experiment 1 raw source, not previous quick/debug runs.
    p1=json.loads((args.experiment1/'plan.json').read_text());s1=json.loads((args.experiment1/'state.json').read_text())
    bridges=[];oldrows=[]
    for job in p1['jobs']:
        if job['model'] not in MODELS or job['condition']!='all_json_5q':continue
        d=Path(s1[job['id']]['attempt_dir']);setting1=json.loads((d/'settings.json').read_text());load1=json.loads((d/'load.json').read_text())
        newjob=next(j for j in plan['jobs'] if j['model']==job['model'] and j['condition'].startswith('D_'))
        assert setting1['video_sha256']==settings_by_job[newjob['id']]['video_sha256']
        assert (d/'prompt-template.txt').read_text()==(source/newjob['id']/'prompt-template.txt').read_text()
        assert load1['model_commit']==load_by_job[newjob['id']]['model_commit']
        for line in (d/'results.jsonl').read_text().splitlines():
            r=json.loads(line);r.update(model=job['model']);oldrows.append(r);new=by[job['model'],'D',r['query_id']]
            bridges.append(dict(model=job['model'],query=r['query_id'],same_raw=r['raw']==new['raw'],same_spans=r['spans']==new['spans'],
                                same_grid=r['video_grid_thw']==new['video_grid_thw'],same_input_tokens=r['input_tokens']==new['input_tokens'],
                                experiment1_wall_s=r['wall_s'],experiment2_D_wall_s=new['wall_s']))
    csv_write('experiment1_bridge.csv',bridges)
    (source/'experiment1_selected_responses.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in oldrows))
    # Preserve only source files matching run-time digests. Record missing/mismatched evidence honestly.
    verified=OUT/'verified_source_snapshot';verified.mkdir(exist_ok=True)
    code=[]
    for name,expected in plan['code_sha256'].items():
        current=ROOT/name;match=current.exists() and sha(current)==expected
        code.append(dict(file=name,expected_sha256=expected,current_sha256=sha(current) if current.exists() else None,match=match))
        if match:shutil.copy2(current,verified/current.name)
    write_json(verified/'verification.json',code)
    # Verify snapshot sources themselves and duration/condition consistency.
    manifest=dict(source_run=str(run),hashes=provenance,code_verification=code,video_sha256=plan['video_sha256'])
    write_json(OUT/'source_manifest.json',manifest)
    elapsed=(max(v['finished_unix_s'] for v in state.values())-min(v['started_unix_s'] for v in state.values()))/60
    derived=dict(duration_s=duration,session_wall_minutes=elapsed,query_wall_minutes=sum(r['wall_s'] for r in rows)/60,
                 load_minutes=sum(s['load_s'] for s in sessions)/60,strict_count=sum(r['strict_json'] for r in rows),
                 incomplete_count=sum(r['incomplete_output'] for r in rows),invalid_count=sum(r['invalid_span_count'] for r in rows),
                 max_output_tokens=max(r['output_tokens'] for r in rows),summary=summary,ratios=ratios,agreement=agreement,
                 attributes=attributes,bridge=bridges,resource_sessions=sessions)
    write_json(OUT/'derived_metrics.json',derived)
    print(json.dumps({'elapsed_min':elapsed,'strict':derived['strict_count'],'max_output':derived['max_output_tokens'],'ratios':ratios,'bridge_equal':sum(b['same_spans'] for b in bridges),'code':code,'attributes':attributes},ensure_ascii=False,indent=2))
    make_figures(rows,summary,resources,agreement,by,duration)
    make_report(rows,summary,sessions,derived,plan,state,by,sm,code)


def make_figures(rows,summary,resources,agreement,by,duration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
        fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    # Actual input allocation: identical shapes in both models.
    fig,axes=plt.subplots(1,3,figsize=(12.4,3.8))
    r=[by[MODELS[0],c,'shot'] for c in CNAMES]
    for ax,vals,title,ylabel in [(axes[0],[x['grid_frames'] for x in r],'Temporal representation','Frames represented by grid (derived)'),
                                 (axes[1],[x['derived_height']*x['derived_width']/1000 for x in r],'Spatial representation','Pixels per frame (thousands, derived)'),
                                 (axes[2],[x['input_tokens'] for x in r],'Actual total input','Input tokens (shot query)')]:
        ax.bar(CNAMES,vals,color=COLORS);ax.set_title(title);ax.set_ylabel(ylabel)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:,.0f}',ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,max(vals)*1.18)
    fig.tight_layout();save(fig,'01_input_allocation')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,m in zip(axes,MODELS):
        a=np.array([[by[m,c,q]['span_count'] for q in QUERIES] for c in CNAMES])
        im=ax.imshow(a,cmap='Blues',vmin=0,vmax=30,aspect='auto')
        ax.set_xticks(range(5),['Shot','Dark shot','Bench','Close-up','Football'],rotation=25,ha='right');ax.set_yticks(range(4),CNAMES);ax.set_title(NAMES[m])
        for y in range(4):
            for x in range(5):ax.text(x,y,str(a[y,x]),ha='center',va='center',color='white' if a[y,x]>17 else '#183d59')
    fig.suptitle('Returned span counts (NOT true event counts)',y=1.02);fig.tight_layout();save(fig,'02_count_matrix')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for ax,m in zip(axes,MODELS):
        r=[next(s for s in summary if s['model']==m and s['condition']==c) for c in CNAMES]
        pre=[s['mean_preprocess_s'] for s in r];gen=[s['mean_generate_s'] for s in r]
        ax.bar(CNAMES,pre,color='#a9c7db',label='Preprocessing')
        ax.bar(CNAMES,gen,bottom=pre,color='#246594',label='Generation')
        for i,c in enumerate(CNAMES):
            vals=[by[m,c,q]['wall_s'] for q in QUERIES]
            ax.scatter(np.arange(5)*.055+i-.11,vals,s=14,c='#b47724',alpha=.8,zorder=4)
            ax.text(i,pre[i]+gen[i]+2,f"{r[i]['mean_wall_s']:.1f}",ha='center')
        ax.set_title(NAMES[m]);ax.set_ylabel('Seconds per query (mean of 5)');ax.set_ylim(0,95);ax.legend(loc='upper left',fontsize=8)
    fig.text(.5,-.015,'Dots: five individual query times; these are not repeated trials or confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'03_latency_breakdown')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for m,color in zip(MODELS,['#246594','#d28a27']):
        r=[next(s for s in summary if s['model']==m and s['condition']==c) for c in CNAMES]
        axes[0].plot(CNAMES,[s['peak_allocated_gib'] for s in r],'-o',label=NAMES[m],color=color)
        axes[0].plot(CNAMES,[s['peak_reserved_gib'] for s in r],'--',color=color,alpha=.6)
        axes[1].scatter([s['mean_wall_s'] for s in r],[s['peak_allocated_gib'] for s in r],s=55,label=NAMES[m],color=color)
        for s,c in zip(r,CNAMES):axes[1].annotate(c,(s['mean_wall_s'],s['peak_allocated_gib']),xytext=(5,4),textcoords='offset points')
    axes[0].set_ylabel('GiB (max across 5 queries)');axes[0].set_title('GPU memory: allocated (solid), reserved (dashed)');axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Mean seconds per query');axes[1].set_ylabel('Peak allocated GiB');axes[1].set_title('Cost view only — no accuracy axis');axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'04_memory_cost')
    for query,name,title in [('shot','05_shot_timeline','Shot predictions'),('bench','06_bench_timeline','Bench celebration predictions')]:
        fig,ax=plt.subplots(figsize=(12.6,5.2))
        labels=[]
        for mi,m in enumerate(MODELS):
            for ci,c in enumerate(CNAMES):
                y=mi*5+ci;row=by[m,c,query];labels.append((y,NAMES[m]+' / '+c))
                ax.barh(y,duration,color='#f1f4f7',height=.66)
                for a,b in row['spans']:ax.barh(y,b-a,left=a,height=.60,color=COLORS[ci],edgecolor='white',linewidth=.5)
        ax.set_yticks([y for y,_ in labels],[s for _,s in labels]);ax.invert_yaxis();ax.set_xlim(0,duration)
        ax.set_xlabel('Seconds from video start');ax.set_title(title+' — raw spans, not ground truth');ax.grid(axis='x',alpha=.2)
        fig.tight_layout();save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(12,3.7))
    for ax,m in zip(axes,MODELS):
        pairs=['A→B','B→C','D→B'];a=np.array([[next(r['temporal_jaccard'] for r in agreement if r['model']==m and r['comparison']==p and r['query']==q) for q in QUERIES[:-1]] for p in pairs])
        ax.imshow(a,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_title(NAMES[m]);ax.set_xticks(range(4),['Shot','Dark shot','Bench','Close-up']);ax.set_yticks(range(3),['A vs B','B vs C','D vs B'])
        for y in range(3):
            for x in range(4):ax.text(x,y,f'{a[y,x]:.2f}',ha='center',va='center',color='white' if a[y,x]>.6 else '#183d59')
    fig.suptitle('Temporal overlap between predictions — NOT IoU against ground truth',y=1.04);fig.tight_layout();save(fig,'07_prediction_agreement')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,m in zip(axes,MODELS):
        for c,color in zip(CNAMES,COLORS):
            rs=[r for r in resources if r['model']==m and r['letter']==c];t0=rs[0]['unix_s']
            ax.plot([(r['unix_s']-t0)/60 for r in rs],[(r.get('process_rss_bytes') or 0)/1024**3 for r in rs],label=c,color=color,linewidth=.9)
        ax.set_title(NAMES[m]);ax.set_xlabel('Minutes since sampler start (separate sessions)');ax.set_ylabel('Sampled process RAM (GiB)');ax.legend(ncol=4,fontsize=8)
    fig.tight_layout();save(fig,'08_host_memory')


# Report construction is appended below.

def make_report(rows,summary,sessions,d,plan,state,by,sm,code):
    duration=d['duration_s']
    def fig(name,alt):return f'![{alt}](analysis/basketball_parameters/{name}.png)\n\n'
    def simple(title,items):return '<blockquote class="easy-summary"><p><strong>'+title+'</strong></p><ol>'+''.join('<li>'+x+'</li>' for x in items)+'</ol></blockquote>\n\n'
    parts=['# 실험2 · FPS와 비주얼 예산에 따른 농구 영상 그라운딩의 변화\n\n',
           '**TimeLens2-4B · TimeLens-8B / 2026-09-26 / 완료된 40회 기록의 분석**\n\n',
           '> 사용자 정의에 따라 앞선 `basketball_v2` 48회 비교를 **실험1**, 이번 `basketball_parameters` 40회 비교를 **실험2**로 부른다. 과거 디버깅 실행과 32회 비교는 본 통계에 포함하지 않았다.\n\n',
           '<div class="visual-grid"><div class="visual-card">분석한 추론<strong>40 / 40</strong><small>2모델 × 4조건 × 5쿼리</small></div><div class="visual-card">총 실행 시간<strong>'+f"{d['session_wall_minutes']:.1f}분"+'</strong><small>로드·기타 오버헤드 포함</small></div><div class="visual-card">실행 실패 / 잘림<strong>0 / 0</strong><small>엄격한 숫자 쌍 배열 40/40</small></div><div class="visual-card pending">의미 정확도<strong>미평가</strong><small>사람이 검수한 정답 구간 없음</small></div></div>\n\n',
           '## 먼저 읽는 용어와 핵심 결론\n\n']
    parts.append(table(['용어','뜻','이 보고서에서 주의할 점'],[
        ['요청 FPS','영상 1초에서 뽑도록 요청한 프레임 수','원본 약 29.97FPS와 다르며, 실제 처리 개수는 정렬 규칙의 영향을 받는다.'],
        ['비주얼 예산','전처리에서 영상 정보량에 배분하는 예산','전체 입력 토큰과 같지 않다. 시간·공간에 나눠 배분된다.'],
        ['실제 입력 토큰','전처리 후 모델에 들어간 전체 토큰 수','영상 토큰뿐 아니라 텍스트·시간표시 등을 포함한다.'],
        ['시간 합집합','예측 구간을 겹치거나 맞닿은 부분까지 병합한 시간','중복을 두 번 세지 않는다. 실제 사건 수나 정답 시간은 아니다.'],
        ['예측 간 Jaccard','두 예측의 시간 교집합 ÷ 합집합','정답과의 IoU가 아니다. 두 조건이 얼마나 다르게 답했는지만 본다.'],
        ['GPU allocated / reserved','PyTorch의 실제 할당 / 캐싱 할당자가 확보한 메모리','프로세스 RAM, 물리 GPU 전체 사용량, 전력과 다르다.'],
    ]))
    parts.append(simple('핵심은 세 가지입니다.',[
        'FPS를 높일수록 이번 실행의 평균 지연은 늘었다. <strong>주된 증가는 전처리</strong>에서 관측됐다.',
        '프레임 수를 늘리면 같은 예산에서 프레임당 해상도가 작아졌다. 이번 실험은 <strong>시간·공간 배분의 비교</strong>이지 순수 FPS 효과의 분리는 아니다.',
        '비주얼 예산이나 FPS 증가가 더 많은 구간 반환을 보장하지 않았다. <strong>정확도 향상 여부는 아직 모른다.</strong>',
    ]))
    parts+=['## 1. 실험1에서 실험2로: 무엇을 이어받았나\n\n',
            '[실험1 발표 문서](BASKETBALL_V2_PRESENTATION_REPORT.html)는 여러 모델의 출력 형식·다중 시간 쌍·프롬프트 영향을 관찰했다. 실험2는 그중 사용자가 선택한 두 모델에서 입력 정보 배분과 비용을 비교한다. 두 모델이 정확도 1·2위라서 선정됐다고 해석하지 않는다.\n\n',
            '### 1.1 기준 조건 D는 실험1과 어떻게 연결되는가\n\n',
            'D는 실험1의 본실험과 FPS 1, 비주얼 예산 16,384, all-json 지시, 출력 상한 2,048이 같다. 프레임 상한은 256→512로 늘었으나 실제 비주얼 격자는 동일했다. 원본 영상 SHA256, 프롬프트 전문, 두 모델의 체크포인트 commit을 대조했다.\n\n']
    parts.append(table(['비교 항목','실험1 선택 모델 본실험 ↔ 실험2 D'],[
        ['대응 응답','2개 모델 × 5개 쿼리 = 10쌍'],['반환 구간','10/10 완전 일치'],['모델 원문','10/10 문자열 일치'],
        ['실제 비주얼 격자·입력 토큰','10/10 일치'],['변경된 상한','max_frames 256→512, 실제 처리 격자는 변화 없음'],
        ['시간 비교','다른 시점의 실행이므로 동일 출력이어도 지연의 차이가 생길 수 있음'],
    ]))
    parts.append('이는 D가 실험1과 연결되는 <strong>일관된 관측 기준</strong>임을 뒷받침한다. 한 번씩의 일치로 모든 GPU·라이브러리 환경에서의 결정성이나 일반적인 재현성을 입증하지는 않는다. [대응 비교 CSV](analysis/basketball_parameters/experiment1_bridge.csv).\n\n')
    parts+=['## 2. 연구 질문과 통제 조건\n\n',table(['질문','이번 자료로 확인 가능한 것','아직 확인할 수 없는 것'],[
        ['RQ1. FPS를 바꾸면?','실제 비주얼 격자·응답·비용의 변화','공간 해상도를 고정한 순수 FPS 인과 효과'],
        ['RQ2. 예산을 두 배로 늘리면?','B와 D의 대응 출력·시간·메모리 차이','정답 회수율 개선'],
        ['RQ3. 어떤 쿼리가 민감한가?','구간 수·경계·시간 겹침·속성 포함 관계','어느 조건의 경계가 정답인지'],
        ['RQ4. 어떤 설정을 채택할까?','현 서버의 비용과 검수 우선순위','GT 기반 최적 모델·설정 순위'],
    ]),'### 2.1 실험 구성\n\n',table(['조건','요청 FPS','비주얼 예산','최대 프레임','출력 상한','비교 관계'],[
        ['A','0.5','32,768','512','2,048','A/B/C: 고정 예산에서 요청 FPS 변화'],
        ['B','1','32,768','512','2,048','FPS 기준 및 예산 비교의 고예산 조건'],
        ['C','2','32,768','512','2,048','FPS 증가 조건'],
        ['D','1','16,384','512','2,048','B/D: 같은 FPS에서 예산 변화'],
    ]),f'입력은 동일한 약 **{duration:.2f}초**, 1920×1080 농구 편집 영상이다. GPU는 NVIDIA A100X MIG 3g.40gb, PyTorch 2.6.0+cu124 환경이다. all-json 지시, BF16, SDPA, greedy, `use_cache=True`를 고정했다. 원문·파서·토큰 진행·자원 시계열을 보존했다.\n\n',
    '**실행 순서:** A의 두 모델 → B의 두 모델 → C의 두 모델 → D의 두 모델. 세션마다 모델을 새로 로드하고, 쿼리는 아래 순서로 한 번씩 실행했다. 시간 제한은 적용하지 않았다. 무작위 순서나 반복 평균 실험이 아니다.\n\n']
    qspec=json.loads((ROOT/'experiments/basketball/queries.json').read_text());qmap={q['id']:q for q in qspec}
    parts.append(table(['ID','검색 문장','관찰 의도'],[[q,qmap[q]['queries'][0],QLABEL[q]] for q in QUERIES]))
    parts.append('모든 쿼리는 사람이 정답을 확인한 라벨이 아니다. 특히 축구 골은 **부재 후보**이고, 다른 행동도 실제 개수와 경계를 아직 전수 검수하지 않았다. 일반 슛과 유니폼 속성 슛은 서로 관련된 문장이므로 독립 표본으로 취급하지 않는다.\n\n')
    parts+=['## 3. 실제로 모델에 들어간 정보\n\n',
            '### 3.1 같은 예산이어도 입력은 같지 않았다\n\n']
    gridrows=[]
    for c in CNAMES:
        r=by[MODELS[0],c,'shot'];mean=sm[MODELS[0],c]['mean_input_tokens']
        gridrows.append([c,str(r['video_grid_thw'][0]),str(r['grid_frames']),f"{r['derived_width']}×{r['derived_height']}",f"{r['derived_visual_tokens']:,}",f'{mean:,.0f}'])
    parts.append(table(['조건','기록된 격자 [T,H,W]','격자 환산 프레임','격자 환산 크기 W×H','병합 후 영상 토큰 추정','실제 전체 입력 토큰 평균'],gridrows))
    parts.append(fig('01_input_allocation','조건별 격자 환산 프레임 수, 공간 픽셀 수, 실제 슛 입력 토큰'))
    parts.append('두 모델 모두 같은 조건에서는 동일한 격자를 기록했다. Qwen3-VL 구조의 temporal patch 2, spatial patch 16, spatial merge 2를 사용해 **프레임=2T**, **크기=16W×16H**, **영상 토큰=T×H×W÷4**로 환산했다. 이는 저장된 격자에서의 계산이며, 실제 샘플 시각·패딩 전 프레임 목록을 직접 기록한 값은 아니다.\n\n')
    parts.append('A→C에서 격자 환산 프레임은 **110→444**로 늘고 프레임당 픽셀 수는 **589,824→147,456**, 즉 4분의 1로 줄었다. 따라서 움직임의 시간 간격을 줄이는 동시에 작은 공·유니폼을 표현하는 공간 정보도 바뀐다. B의 704×384와 D의 512×288도 서로 다르다.\n\n')
    parts.append('또한 예산 32,768인 A/B/C의 실제 전체 입력 토큰 평균은 **32,255 / 30,412 / 34,131**이다. 프레임·격자 정렬과 텍스트·시간표시 등이 포함되므로 설정 숫자가 엄격한 전체 입력 상한이 아니다. C가 32,768을 넘었다는 사실만으로 예산 설정 오류라고 판단하면 안 된다.\n\n')
    parts.append(simple('입력을 이해하는 요약',[
        'FPS만 설정에서 바꿨더라도 모델이 본 공간 해상도도 달라졌다.',
        'C는 A보다 시간적으로 촘촘하지만 프레임 한 장은 더 작다.',
        '따라서 아래 결과를 “FPS만 높여서 생긴 순수 효과”라고 부르지 않는다.',
    ]))
    parts+=['## 4. 실행 안정성과 출력 형식\n\n',table(['점검 항목','결과','의미'],[
        ['실행 완료','40/40; 실패 0','이번 조건에서 응답 생성이 완료됨'],
        ['엄격한 JSON 숫자 쌍 배열','40/40; 부재 후보 제외 32/32','원문 자체의 구문 검사; 파서 복구만의 결과가 아님'],
        ['잘림 / 범위 밖 구간','0 / 0','완결된 출력이고 유효한 시간 범위임'],
        ['최대 생성 토큰',str(d['max_output_tokens']),'이번 두 모델의 모든 출력은 512토큰보다도 짧았음'],
        ['축구 골 응답','8/8 빈 배열','두 모델·네 조건의 이 쿼리에서 빈 배열; 일반 환각 억제 능력은 미평가'],
        ['의미 정확도·Recall·F1','산출하지 않음','검수된 정답 구간이 없음'],
    ]),'출력 상한 2,048이 필요해서 성공했다고 해석할 근거는 없다. 이번 최장 출력은 422토큰이므로 **상한 증가의 효과를 이 40회만으로 입증하지 못한다.** 프롬프트에 맞는 숫자 배열도 영상 내용에 틀릴 수 있다.\n\n',
    '## 5. 구간 결과: 개수와 시간 구조를 함께 보기\n\n',
    '### 5.1 모델 × 조건 × 쿼리의 반환 개수\n\n',fig('02_count_matrix','각 모델의 조건별 쿼리 반환 수 열지도')]
    parts.append(table(['모델','조건','슛','어두운 유니폼 슛','벤치','클로즈업','축구'],[[NAMES[s['model']],s['condition']]+[s[q+'_count'] for q in QUERIES] for s in summary]))
    parts.append('TimeLens2-4B의 슛은 A/B/C/D에서 **10/5/5/14개**, TimeLens-8B는 **22/27/30/30개**다. 증가하는 방향도 모델·쿼리에 따라 다르다. 예를 들어 TimeLens-8B의 클로즈업은 FPS 0.5→1→2에서 **21→10→12개**로 변한다. 개수의 많고 적음은 현재 정답 회수율과 연결할 수 없다.\n\n')
    parts+=['### 5.2 슛: 같은 30개라도 같은 결과가 아니다\n\n',fig('05_shot_timeline','두 모델의 A B C D 슛 예측 구간 시간선')]
    parts.append(table(['모델','조건','원래 구간 수','연결 성분 수','예측 시간 합집합','영상 덮음 비율'],[[NAMES[r['model']],r['letter'],r['span_count'],r['components'],f"{r['union_s']:.0f}초",f"{r['coverage_pct']:.1f}%"] for r in rows if r['query_id']=='shot']))
    parts.append('**TimeLens2-4B:** B와 C의 슛 예측은 모두 영상 시작 후 **36초 이내**에만 있다. D와 A는 후반 시간대에도 구간을 낸다. 이는 반환 위치가 앞부분에 집중됐다는 관찰이며, B/C 입력 영상 자체가 36초에서 잘렸다거나 나머지 슛을 놓쳤다는 결론은 아직 아니다. GT와 실제 샘플 시각 검수가 필요하다.\n\n')
    parts.append('**TimeLens-8B:** C와 D는 모두 30개를 반환하지만 합집합 길이는 **130초와 95초**로 다르다. C는 맞닿은 두 구간을 병합하면 29개 연결 성분이다. 더 큰 덮음은 더 많은 정답 회수일 수도, 더 넓은 오탐일 수도 있어 개수만으로 C가 낫다고 판단할 수 없다.\n\n')
    parts+=['### 5.3 벤치: 같은 2개가 전혀 다른 경계를 뜻한다\n\n',fig('06_bench_timeline','두 모델의 조건별 벤치 환호 예측 시간선'),
            'TimeLens2-4B는 D에서 **[134,160], [208,217]**을, B에서는 **[134,137], [157,160]**을 반환한다. 둘 다 2개지만 D의 긴 구간 하나가 B에서는 두 부분으로 나뉘고, 뒤쪽 구간도 달라졌다. 이 비교의 예측 간 시간 Jaccard는 **0.171**이다. 정확한 분할인지 과도한 분할인지는 영상 대조가 필요하다.\n\n',
            'TimeLens-8B의 B/D 벤치 예측은 각각 3개이며 시간 Jaccard **0.846**으로 상대적으로 비슷했다. 그러나 C에서는 두 모델 모두 59~62초 부근과 94~98초 부근의 구간을 추가했다. 같은 추가가 정답을 뜻하지는 않으므로 우선 검수할 사례다.\n\n',
            '### 5.4 조건 변화에 대한 민감도\n\n',fig('07_prediction_agreement','조건 쌍별 예측 시간 합집합의 Jaccard; 정답 IoU가 아님'),
            '위 값은 **J = 길이(예측₁∩예측₂) / 길이(예측₁∪예측₂)**이다. 두 빈 배열의 J는 정의하지 않고 축구 골 쿼리를 그림에서 제외했다. 높은 값은 비슷한 답, 낮은 값은 다른 답을 뜻하며 어느 답이 맞는지 알려주지 않는다.\n\n',
            'TimeLens2-4B의 B/C 슛은 J=0.906으로 비슷하지만, 두 답 모두 앞 36초에 집중됐다. **안정적인 출력도 같은 방식으로 틀릴 수 있다.** 반대로 낮은 일치도가 개선인지 악화인지는 GT 없이 구분할 수 없다.\n\n',
            '### 5.5 유니폼 속성의 포함 관계 진단\n\n']
    parts.append(table(['모델','조건','속성 슛 예측 중 일반 슛 예측 밖 시간 비율'],[[NAMES[a['model']],a['condition'],f"{a['dark_outside_shot_pct']:.1f}%"] for a in d['attributes']]))
    parts.append('일반 슛보다 좁은 조건인 “어두운 유니폼 슛”이 일반 슛 예측 밖에 놓인 시간을 계산했다. 예컨대 TimeLens2-4B의 D는 **33.3%**, B/C는 **0%**다. 이 값이 줄어도 정확도 개선이라 할 수 없다. 일반 슛의 누락, 속성 슛의 오탐, 경계 차이를 분리하지 못하고 일반 슛을 영상 전체로 반환해도 0%가 되기 때문이다.\n\n')
    parts.append(simple('출력 구조에서 읽을 수 있는 것',[
        'FPS·예산을 늘리면 답이 달라진다. 하지만 반환 구간 수는 단조롭게 늘지 않는다.',
        '개수뿐 아니라 어느 시간대를 얼마나 덮는지, 긴 구간으로 묶는지 함께 봐야 한다.',
        '예측 간 일치도와 속성 포함 관계는 검수할 사례를 고르는 진단값이지 정확도 점수가 아니다.',
    ]))
    parts+=['## 6. 시간·GPU·호스트 자원 분석\n\n','### 6.1 평균 지연은 FPS와 함께 늘었다\n\n',fig('03_latency_breakdown','전처리와 생성 시간의 분해 및 쿼리별 관측점')]
    parts.append(table(['모델','조건','전처리 평균','생성 평균','전체 평균','쿼리별 범위','최대 allocated'],[[NAMES[s['model']],s['condition'],f"{s['mean_preprocess_s']:.2f}초",f"{s['mean_generate_s']:.2f}초",f"{s['mean_wall_s']:.2f}초",f"{s['min_wall_s']:.1f}~{s['max_wall_s']:.1f}초",f"{s['peak_allocated_gib']:.2f} GiB"] for s in summary]))
    parts.append('전체 시간은 영상 전처리·입력 전송·GPU 동기화 후 생성·텍스트 해석을 포함하며 모델 로드와 결과 파일 기록은 제외한다. 막대는 다섯 쿼리의 산술 평균이고 점은 서로 다른 쿼리의 관측값이다. **반복 실험 오차막대나 신뢰구간이 아니다.** 첫 쿼리는 항상 슛이므로 워밍업과 쿼리 효과가 얽혀 있다.\n\n')
    parts.append(table(['모델','비교','전체 지연 변화','전처리 차이','생성 차이','최대 allocated 차이'],[[NAMES[r['model']],r['comparison'],f"{r['wall_change_pct']:+.1f}%",f"{r['preprocess_delta_s']:+.2f}초",f"{r['generation_delta_s']:+.2f}초",f"{r['memory_delta_gib']:+.2f} GiB"] for r in d['ratios']]))
    parts.append('A→B에서 TimeLens2-4B의 생성 시간은 오히려 **2.78초 줄었지만**, 전처리가 **11.57초 늘어** 전체는 느려졌다. TimeLens-8B도 전처리가 **12.95초 증가**했고 생성 평균은 거의 같았다. B→C의 지연 증가 역시 주로 전처리 차이에서 관측된다. 따라서 현재 경로에서는 모델 생성만 최적화해도 영상 디코딩·샘플링 비용이 남는다.\n\n')
    parts.append('다만 생성 시간은 출력 길이·입력 길이·첫 호출 효과의 영향을 함께 받는다. 한 세션 평균이 모델의 고유 토큰 생성 효율이나 인과적인 연산량을 뜻하지 않는다. 쿼리별 입력/출력 토큰과 시간은 CSV에 함께 보관했다. 빈 배열 쿼리를 제외한 평균과 첫 호출을 제외한 평균도 `condition_summary.csv`에 별도로 기록했다.\n\n')
    parts+=['### 6.2 예산 증가의 메모리 비용\n\n',fig('04_memory_cost','GPU allocated reserved 메모리 및 지연과 메모리의 비용 비교'),
            'D→B는 요청 FPS를 1로 유지하며 비주얼 예산을 두 배로 늘린 비교다. TimeLens2-4B의 최대 allocated는 **12.56→15.94 GiB**, TimeLens-8B는 **21.24→25.11 GiB**로 늘었다. 전체 지연은 각각 **17.1%·29.6% 증가**했다. 이 비용에 비례하는 정확도 이득은 아직 증명되지 않았다.\n\n',
            'A/B/C에서는 GPU 메모리가 FPS에 따라 단조롭게 늘지 않는다. A의 실제 입력 토큰 수가 B보다 많고 최대 allocated도 조금 높다. C가 가장 높지만, 단순히 FPS만으로 GPU 메모리를 예측하기보다 **실제 격자와 토큰 수**를 함께 봐야 한다.\n\n',
            '### 6.3 호스트 RAM과 계측 한계\n\n',fig('08_host_memory','조건별 세션 시간에 따른 프로세스 RAM 샘플 추이')]
    parts.append(table(['모델','A RAM 피크','B RAM 피크','C RAM 피크','D RAM 피크'],[[NAMES[m]]+[f"{sm[m,c]['peak_sampled_rss_gib']:.2f} GiB" for c in CNAMES] for m in MODELS]))
    parts.append('프로세스 RSS는 추론 단계 샘플 중 최대값이다. 약 1초 간격의 관측이라 짧은 피크를 놓칠 수 있고 PyTorch GPU 피크와 측정 방식이 다르다. C에서는 두 모델 모두 약 **18 GiB RAM**까지 관측돼 A의 약 7 GiB보다 높았다. 그래프는 각 세션 시작을 0으로 맞춘 개별 실행이며 동시 실행 그래프가 아니다.\n\n')
    total_samples=sum(s['samples'] for s in sessions);numeric=sum(s['gpu_util_numeric_samples'] for s in sessions)
    parts.append(f'자원 로그는 총 **{total_samples:,}개** 샘플이다. NVIDIA 조회 명령 자체의 실패는 {sum(s["gpu_query_failures"] for s in sessions)}건이지만, **수치 GPU 사용률 샘플은 {numeric}개**다. MIG 환경의 N/A·권한 제한을 0%로 처리하지 않았다. 물리 보드 전력은 다른 MIG 작업을 포함할 수 있어 이번 모델의 에너지나 비용으로 환산하지 않았다.\n\n')
    parts.append(f'실험 전체는 **{d["session_wall_minutes"]:.2f}분**, 쿼리 지연 합은 **{d["query_wall_minutes"]:.2f}분**, 모델 로드 기록 합은 **{d["load_minutes"]:.2f}분**이다. 나머지는 환경 점검·기록·해제·세션 전환·폴링 등의 오버헤드다. 특정 모델 로드가 오래 걸린 이유를 다운로드나 GPU 경쟁으로 단정하지 않았다.\n\n')
    parts+=['## 7. 직접 살펴보기: 쿼리·조건·원문 탐색\n\n',
            '아래 탐색기는 이 HTML에 포함된 40개 응답만 사용한다. 모델과 쿼리를 바꾸면 A~D 시간선을 비교할 수 있다. 구간을 선택하면 해당 조건의 원문이 표시된다. 선택적으로 **같은 원본 영상 파일**을 열면 그 구간부터 재생할 수 있다. 영상은 보고서에 포함하지 않으며 브라우저 안에서만 사용한다.\n\n',
            '<!--EXPLORER-->\n\n',
            '## 8. 해석의 경계와 다음 검수\n\n',
            '### 8.1 말할 수 있는 것과 없는 것\n\n']
    parts.append(table(['주장','판정'],[
        ['FPS 상승은 이번 실행에서 평균 지연을 늘렸다','관측으로 지지. 전처리의 증가가 주된 구성요소였다.'],
        ['FPS 상승으로 정확도가 높아졌다','판정 불가. GT가 없고 공간 해상도도 달라졌다.'],
        ['예산을 두 배로 늘리면 더 많은 장면을 찾는다','보편적 주장으로 지지하지 않음. TL2 슛 D 14개→B 5개처럼 감소 사례 존재.'],
        ['구간 수가 줄었으므로 성능이 나빠졌다','판정 불가. 줄어든 것이 오탐인지 정답인지 모름.'],
        ['C가 더 비싸므로 가장 좋다','근거 없음. 비용과 정확도는 별개.'],
        ['D는 실험1과 연결되는 기준 조건이다','10개 대응 원문·구간·격자 일치로 뒷받침. 일반 재현성 주장은 아님.'],
        ['완전한 JSON 40개이므로 시스템이 정확하다','구문·실행 안정성만 확인. 의미 검증은 남음.'],
    ]))
    parts.append('### 8.2 먼저 사람이 확인할 사례\n\n')
    parts.append(table(['검수 우선순위','비교','확인할 질문'],[
        ['1','TL2 슛 B/C의 0~36초 집중 vs A/D 후반 예측','후반의 실제 슛을 누락한 것인가? 앞부분 예측 중 오탐은 없는가?'],
        ['2','TL2 벤치 D [134,160] vs B [134,137], [157,160]','중간에 무관한 장면이 있어 분할이 필요한가?'],
        ['3','C의 59~62초 및 94~98초 벤치 예측','벤치 환호인지 경기 장면인지 실제 영상으로 판정한다.'],
        ['4','TimeLens-8B 클로즈업 A 21개 vs B 10개','더 많은 구간이 정답 증가인가, 넓게 잡은 오탐인가?'],
        ['5','dark_shot이 일반 shot 밖인 구간','유니폼 조건을 혼동했는가, 일반 슛 예측이 빠졌는가?'],
    ]))
    parts.append('정답 라벨은 전체 영상을 보며 쿼리별 모든 사건·경계를 기록하고 리플레이 포함 규칙을 고정해야 한다. 가능하면 두 명이 모델 결과를 보지 않고 독립 검수한다. 이후 tIoU 0.3/0.5/0.7에서 일대일 사건 매칭으로 Precision/Recall/F1을 계산한다. 현재 예측 간 Jaccard를 그 점수로 대신하지 않는다.\n\n')
    parts.append('### 8.3 후속 실행 전에 필요한 결정\n\n')
    parts.append('**당장 기본값을 비싼 C로 올릴 근거는 없다.** D는 실험1과 일치하는 저예산 기준으로 유지하고, A는 낮은 지연과 높은 프레임당 공간 정보가 필요한 경우의 검수 후보로 둔다. B/C는 경계 분할과 짧은 동작의 누락 여부를 GT로 확인할 비교군이다. 이는 운영·검수 우선순위이며 정확도 최적 설정 선정이 아니다.\n\n')
    parts.append('순수한 FPS 효과를 분리하려면 별도 실험에서 프레임당 해상도를 고정하고 실제 샘플 시각을 저장해야 한다. 비용 비교를 강화하려면 동일 조건을 반복하고 순서를 교차·무작위화하며 전처리 캐시 유무를 별도 조건으로 분리한다. 현재 한 영상·연관된 쿼리 5개·각 1회이므로 통계적 유의성, 일반화 성능, 실험 간 인과 결론은 주장하지 않는다.\n\n')
    parts.append(simple('발표 마무리',[
        '실험1은 여러 시간 쌍을 반환하는 행동을 확인했고, 실험2는 그 행동이 입력 배분에 민감함을 보였다.',
        '더 높은 FPS와 예산에는 관측 가능한 비용이 들지만 정확도 이득은 아직 검증되지 않았다.',
        '다음 단계는 설정을 더 늘리는 것이 아니라, 대표 차이를 실제 영상·정답과 대조하는 것이다.',
    ]))
    parts+=['## 9. 자료·계산 정의·재현 범위\n\n',
            '- 원자료: `~/aim-workspace/results_grounding/basketball_parameters/`의 최신 완료 attempt만 사용. 실패한 예전 배치나 다른 모델의 결과는 포함하지 않았다.\n',
            '- [원문 40회 스냅샷](analysis/basketball_parameters/response_snapshot.jsonl), [계산된 지표 JSON](analysis/basketball_parameters/derived_metrics.json), [조건별 CSV](analysis/basketball_parameters/condition_summary.csv), [쿼리별 CSV](analysis/basketball_parameters/query_metrics.csv).\n',
            '- [예측 간 일치도](analysis/basketball_parameters/prediction_agreement.csv), [속성 포함 관계](analysis/basketball_parameters/attribute_consistency.csv), [세션 계측](analysis/basketball_parameters/session_resources.csv), [원자료 SHA256 목록](analysis/basketball_parameters/source_manifest.json).\n',
            '- [분석·보고서 생성 코드](analysis/basketball_parameters/build_analysis.py). 재생성: `~/aim-workspace/grounding_env/bin/python docs/analysis/basketball_parameters/build_analysis.py`. 추가 모델 추론 없이 파일을 다시 읽는다.\n',
            '- [주장·한계 자체 검토](BASKETBALL_EXPERIMENT2_CRITICAL_REVIEW.md).\n',
            '- PNG와 SVG 그림은 `docs/analysis/basketball_parameters/`에 별도로 저장했고 이 HTML에는 PNG를 내장했다. 브라우저 인쇄로 PDF 저장이 가능하다.\n\n']
    parts.append(table(['실행 코드','기록된 SHA256과 일치','보존'],[[r['file'],'예' if r['match'] else '아니오','일치 사본 저장' if r['match'] else '당시 소스라고 복사하지 않음'] for r in code]))
    parts.append('세 코드 파일의 현재 내용은 이번 실행 계획의 해시와 일치해 사본을 보존했다. 모델 commit, pip freeze, GPU 정보와 원문 설정도 세션별 사본에 포함했다. 그렇더라도 컨테이너 이미지·GPU 드라이버·모든 외부 의존성이 고정된 완전한 실행 패키지는 아니다.\n\n')
    parts.append('계산 정의: 구간 합집합은 겹치거나 정확히 맞닿은 구간만 병합(허용 간격 0초), 길이는 종료−시작, 덮음 비율은 합집합 길이÷영상 길이. 속성 밖 비율은 길이(dark_shot−shot)÷길이(dark_shot). 이번 40개 원문에는 범위 밖 구간이 없어 지표 계산을 위한 경계 보정은 하지 않았다. 성능 평균은 성공한 다섯 쿼리의 산술 평균이며, GPU 피크는 그 다섯 호출 중 최대값이다.\n\n')
    md=''.join(parts)
    source_md=DOCS/'BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.md';source_md.write_text(md)
    render(md,rows,duration)


def render(md,rows,duration):
    from markdown_it import MarkdownIt
    reference=(DOCS/'BASKETBALL_V2_PRESENTATION_REPORT.html').read_text()
    css=re.search(r'<style>(.*?)</style>',reference,re.S)[1]
    css+='''
    figure{margin:24px 0}.table-wrap{overflow-x:auto}details{margin:16px 0}summary{cursor:pointer;color:#245d87;font-weight:bold}
    .explorer{border:1px solid #cbdbe5;border-radius:12px;padding:20px;background:#f7fafc}.explorer select,.explorer input{max-width:100%;padding:8px;margin:5px 12px 5px 3px}.explorer label{display:inline-block}.explorer svg{width:100%;height:auto;background:white}.explorer pre{max-height:280px;overflow:auto}.explorer rect[role=button]{cursor:pointer}.explorer rect[role=button]:focus{outline:2px solid #111}.explorer video{width:100%;max-height:440px}.caption{font-size:13px;color:#526779}.nowrap{white-space:nowrap}
    @media print{.explorer{display:none}h2{break-before:auto;margin-top:28px}.visual-panel{break-inside:auto}table{font-size:8pt}h2,h3{break-after:avoid}main{width:auto}}
    '''
    body=MarkdownIt('commonmark',{'html':True}).enable('table').render(md)
    body=body.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
    def embed(m):
        path=DOCS/m[1]
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    body=re.sub(r'src="(analysis/basketball_parameters/[^\"]+\.png)"',embed,body)
    headings=[]
    def heading(m):
        key='section-'+str(len(headings)+1);headings.append((key,m[1]));return f'<h2 id="{key}">{m[1]}</h2>'
    body=re.sub(r'<h2>(.*?)</h2>',heading,body)
    explorer='''<section class="explorer" aria-label="예측 구간 대화형 탐색"><div><label for="model">모델</label><select id="model"><option value="timelens2-4b">TimeLens2-4B</option><option value="timelens-8b">TimeLens-8B</option></select><label for="query">쿼리</label><select id="query"><option value="shot">슛</option><option value="dark_shot">어두운 유니폼 슛</option><option value="bench">벤치 환호</option><option value="closeup">얼굴 클로즈업</option><option value="football">축구 골 후보</option></select><label for="condition">원문 조건</label><select id="condition"><option>A</option><option>B</option><option>C</option><option>D</option></select></div><p id="query-text"></p><svg id="timeline" viewBox="0 0 960 270" role="img" aria-label="선택한 모델과 쿼리의 네 조건 예측 시간선"></svg><p id="selection" aria-live="polite"></p><label for="video-file">원본 영상 파일 선택(선택사항)</label><input id="video-file" type="file" accept="video/*"><video id="video" controls hidden preload="metadata"></video><p class="caption">색 막대는 예측이며 정답이 아닙니다. 선택한 파일이 같은 원본인지 확인하세요. 구간 클릭 또는 키보드 Enter로 해당 구간을 재생합니다.</p><h4>선택 조건의 원문·계측</h4><p id="metrics"></p><pre id="raw"></pre></section><noscript>스크립트가 꺼져 있습니다. 위 정적 그림과 표에서 모든 핵심 결과를 확인할 수 있습니다.</noscript>'''
    body=body.replace('<!--EXPLORER-->',explorer)
    dataset=json.dumps(rows,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    script=r'''
    const data=JSON.parse(document.getElementById('report-data').textContent);
    const duration=DURATION;
    const $=id=>document.getElementById(id), colors=['#2274a5','#008577','#d28a27','#865ba6'];
    let localURL=null, endTime=null;
    function rawView(){const r=data.find(x=>x.model===$('model').value&&x.query_id===$('query').value&&x.letter===$('condition').value);$('raw').textContent=r.raw;$('metrics').textContent=`${r.letter} | 반환 ${r.span_count}개 | 예측 덮음 ${r.coverage_pct.toFixed(1)}% | ${r.wall_s.toFixed(2)}초 | GPU ${r.peak_allocated_gib.toFixed(2)} GiB | 입력 ${r.input_tokens.toLocaleString()} 토큰`;}
    function draw(){const rr=data.filter(x=>x.model===$('model').value&&x.query_id===$('query').value).sort((a,b)=>a.letter.localeCompare(b.letter));$('query-text').textContent=rr[0].query;const svg=$('timeline');svg.replaceChildren();const NS='http://www.w3.org/2000/svg';function el(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;}for(let t=0;t<=220;t+=20){let x=65+t/duration*865;el('line',{x1:x,x2:x,y1:20,y2:220,stroke:'#dce3e8'});el('text',{x:x,y:248,'text-anchor':'middle','font-size':13,fill:'#526779'},t+'s');}rr.forEach((r,i)=>{let y=30+i*48;el('text',{x:15,y:y+20,'font-size':16,fill:colors[i]},r.letter);el('rect',{x:65,y,width:865,height:28,fill:'#f0f3f5'});r.spans.forEach(([a,b])=>{const n=el('rect',{x:65+a/duration*865,y:y+1,width:Math.max(.7,(b-a)/duration*865),height:26,fill:colors[i],tabindex:0,role:'button','aria-label':`${r.letter}: ${a}초에서 ${b}초 예측`});const title=document.createElementNS(NS,'title');title.textContent=`${a}–${b}초`;n.append(title);const select=()=>{$('condition').value=r.letter;rawView();$('selection').textContent=`${r.letter} 예측: ${a}–${b}초 (정답 여부 미검수)`;if(localURL){endTime=b;$('video').currentTime=a;$('video').play().catch(()=>{});}};n.addEventListener('click',select);n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});});});rawView();}
    $('model').addEventListener('change',draw);$('query').addEventListener('change',draw);$('condition').addEventListener('change',rawView);
    $('video-file').addEventListener('change',e=>{if(localURL)URL.revokeObjectURL(localURL);const f=e.target.files[0];if(!f){localURL=null;$('video').hidden=true;return;}localURL=URL.createObjectURL(f);$('video').src=localURL;$('video').hidden=false;endTime=null;});
    $('video').addEventListener('timeupdate',()=>{if(endTime!==null&&$('video').currentTime>=endTime){$('video').pause();endTime=null;}});
    draw();
    '''.replace('DURATION',str(duration))
    nav='<nav>'+''.join(f'<a href="#{key}">{label}</a>' for key,label in headings)+'</nav>'
    page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험2 · FPS·비주얼 예산 비교 분석</title><style>'+css+'</style></head><body><main><p class="print-note">실험1 후속 발표용 보고서 · 그림 내장 단일 HTML · 브라우저 인쇄 → PDF 저장 가능</p>'+nav+body+'</main><script id="report-data" type="application/json">'+dataset+'</script><script>'+script+'</script></body></html>'
    target=DOCS/'BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.html';target.write_text(page)
    (OUT/'explorer.js').write_text(script)
    assert page.count('src="data:image/png;base64,')==8
    print(target)


if __name__=='__main__':main()
