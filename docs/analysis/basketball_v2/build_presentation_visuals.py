#!/usr/bin/env python3
"""Presentation figures from saved v2 responses; never launches inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from critical_audit import union

BASE=Path(__file__).resolve().parent
ROWS=[json.loads(x) for x in (BASE/'response_snapshot.jsonl').read_text().splitlines()]
META=json.loads((BASE/'derived_metrics.json').read_text())
MODELS=list(META['models'])
LABELS=['TimeLens2-4B','TimeLens2-8B','TimeLens-8B','TimeLens-7B','Time-R1-7B','Time-R1-3B','Qwen3-VL-8B','Qwen2.5-VL-7B']
plt.rcParams.update({'font.family':'NanumGothic','font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white'})
BLUE='#246594';TEAL='#008577';RED='#b54835';INK='#182f45'
def save(fig,name):
 for ext in ('png','svg'):fig.savefig(BASE/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def row(m,q,condition='all_json_5q'):
 return next(r for r in ROWS if r['model']==m and r['query_id']==q and r['condition']==condition)

# No magnitude heatmap: large counts must not read as better performance.
fig,ax=plt.subplots(figsize=(13,7));ax.set_xlim(-2.2,5);ax.set_ylim(8.6,-1.35);ax.axis('off')
for j,title in enumerate(['Q1 일반 슛','Q2 속성 슛','Q3 벤치 환호','Q4 얼굴 클로즈업','Q5 축구 골']):ax.text(j+.5,-.65,title,ha='center',weight='bold',fontsize=11)
for i,(m,label) in enumerate(zip(MODELS,LABELS)):
 ax.text(-.12,i+.5,label,ha='right',va='center',fontsize=12)
 for j,q in enumerate(['shot','dark_shot','bench','closeup','football']):
  r=row(m,q);bad=r['incomplete_output'] or r['invalid_span_count'];n=len(r['spans'])
  ax.add_patch(Rectangle((j+.035,i+.07),.93,.86,facecolor='#fff0e9' if bad else '#eef4f8',edgecolor='white'))
  suffix=' †‡' if r['invalid_span_count'] else ' †' if r['incomplete_output'] else ''
  ax.text(j+.5,i+.49,str(n)+suffix,ha='center',va='center',fontsize=18,color=RED if bad else INK,weight='bold')
fig.suptitle('모델과 질문에 따라 반환 개수가 크게 달라진다',fontsize=20,weight='bold',y=1.01)
ax.text(-2.05,8.45,'숫자 = 예측 시간 쌍 수, 정답 수 아님   |   † 잘린 응답의 부분 복구   ‡ 영상 밖 구간 포함',fontsize=11,color=RED)
save(fig,'query_count_matrix')

fig,ax=plt.subplots(figsize=(12,6.8))
for i,m in enumerate(MODELS):
 native=len(row(m,'shot','default_shot_control')['spans']);allr=row(m,'shot');count=len(allr['spans'])
 ax.plot([native,count],[i,i],color='#b9cbd8',lw=3,zorder=1)
 ax.scatter(native,i,s=90,facecolor='white',edgecolor=BLUE,lw=2,zorder=3,label='기본 지시' if i==0 else None)
 ax.scatter(count,i,s=95,marker='D',color=RED if allr['incomplete_output'] else TEAL,zorder=2,label='all-json 지시' if i==0 else None)
 ax.annotate(str(native),(native,i),xytext=(-8,9),textcoords='offset points',ha='right',fontsize=11,color=BLUE)
 ax.annotate(str(count)+(' †‡' if allr['invalid_span_count'] else ''),(count,i),xytext=(8,-14),textcoords='offset points',ha='left',fontsize=12,color=RED if allr['incomplete_output'] else TEAL)
ax.set_yticks(range(8),LABELS);ax.invert_yaxis();ax.set_xlim(-4,111);ax.set_xlabel('반환 시간 쌍 수 · Q1 슛 · 동일 영상');ax.grid(axis='x',alpha=.2)
ax.spines[['top','right','left']].set_visible(False);ax.legend(loc='upper right');ax.set_title('기본 지시의 한 구간 ≠ 다중 출력 불가능',fontsize=20,weight='bold',pad=24)
fig.text(.13,-.015,'프롬프트 전체 묶음의 비교이며 ALL만의 인과 효과는 아님. TimeLens2 기본 지시도 ALL 요청.\n†‡ Qwen2.5의 91개는 잘림·영상 밖 구간을 포함하며 유효 검출 수가 아님.',fontsize=11,color=INK)
save(fig,'prompt_count_comparison')

r=row('qwen3-vl-8b','shot');spans=r['spans'];merged=union(spans,META['duration_s'])
assert len(spans)==98 and len(merged)==2
fig,ax=plt.subplots(figsize=(13,4.6))
for a,b in spans:ax.broken_barh([(a,b-a)],(1.2,.4),facecolors=BLUE,edgecolors='white',linewidth=.7)
for a,b in merged:
 ax.broken_barh([(a,b-a)],(.3,.4),facecolors=TEAL)
 ax.text((a+b)/2,.08,f'{a:g}–{b:g}초',ha='center',fontsize=12,color=TEAL)
ax.set_xlim(0,META['duration_s']);ax.set_ylim(-.25,2);ax.set_yticks([1.4,.5],['원문: 98개 시간 쌍','맞닿은 구간 병합: 2개']);ax.set_xticks([0,40,80,120,160,200,222.56]);ax.set_xlabel('영상 재생 시간 (초)');ax.spines[['left','top','right']].set_visible(False)
ax.set_title('98개가 98개의 서로 다른 슛을 뜻하지는 않는다',fontsize=20,weight='bold',pad=24)
fig.text(.125,-.17,'Qwen3-VL-8B · Q1  |  예측 합집합 198.5초 / 영상 222.56초 = 89.2%\n겹치거나 맞닿은 구간만 병합(허용 간격 0초). 2개 역시 정답 사건 수가 아님. GT 미작성.',fontsize=12,color=INK)
save(fig,'fragmentation_explainer')
print('Created 3 presentation figures in PNG and SVG.')
