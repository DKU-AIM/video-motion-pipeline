"""Accessible HTML summary panels; quantities derived from saved response data."""
import json
from pathlib import Path
from critical_audit import strict_pairs
BASE=Path(__file__).resolve().parent
CSS='''
.visual-panel{border:1px solid #cbdbe5;border-radius:14px;padding:25px;margin:28px 0;background:#f7fafc;break-inside:avoid}
.visual-panel h4{margin:0 0 18px;font-size:22px;line-height:1.4;color:#183d59}.visual-panel p{margin:12px 0 0}.visual-kicker{font-size:12px;font-weight:bold;letter-spacing:.08em;color:#49677e;margin-bottom:8px}
.visual-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.visual-grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.visual-card{padding:18px;background:white;border:1px solid #d6e2e9;border-radius:8px}.visual-card strong{display:block;font-size:29px;color:#246594;line-height:1.3;margin:8px 0}.visual-card small{display:block;font-size:12px;color:#526779}.visual-card.pending{border:2px dashed #b57631;background:#fffbf3}.visual-card.pending strong{color:#98611d;font-size:24px}
.visual-pill{display:inline-block;background:#e7eff5;border-radius:5px;padding:4px 9px;margin:4px 4px 0 0;font-size:13px}.visual-note{font-size:13px;color:#526779}.visual-equation{font-size:22px;text-align:center;font-weight:bold;margin:18px 0;color:#183d59}.visual-panel ol{padding-left:22px}.visual-panel li{margin:9px 0}
@media(max-width:760px){.visual-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.visual-grid.two{grid-template-columns:1fr}.visual-panel{padding:16px}.visual-card strong{font-size:24px}}
@media print{.visual-panel{padding:14px}.visual-card{padding:10px}.visual-panel h4{font-size:15pt}.visual-card strong{font-size:20pt}.visual-note,.visual-card small{font-size:9pt}}
'''
def inject(body):
 rows=[json.loads(l) for l in (BASE/'response_snapshot.jsonl').read_text().splitlines()]
 main=[r for r in rows if r['condition']=='all_json_5q'];native=[r for r in rows if r['condition']=='default_shot_control']
 n=len(main);complete=sum(not r['incomplete_output'] for r in main);strict=sum(strict_pairs(r['raw']) for r in main);ok=sum(r['status']=='ok' for r in main)
 assert (n,len(native),len(rows))==(40,8,48)
 design=f'''<section class="visual-panel" aria-label="실험 구성 요약"><div class="visual-kicker">실험을 한눈에</div><h4>같은 영상, 모델 8개, 질문 5개</h4><div class="visual-grid"><div class="visual-card">공통 입력<strong>영상 1개</strong><small>222.56초 농구 편집 영상<br>독립 영상 표본은 1개</small></div><div class="visual-card">본실험<strong>8 × 5 = {n}</strong><small>모델 × 쿼리<br>모두 all-json 지시</small></div><div class="visual-card">기본 지시 대조<strong>8 × 1 = {len(native)}</strong><small>모델 × 일반 슛 질문<br>나머지 4개 질문은 대조 없음</small></div><div class="visual-card">총 호출<strong>{len(rows)}회</strong><small>16개 모델 세션<br>각 조건 반복 1회</small></div></div><p><span class="visual-pill">요청 FPS 1</span><span class="visual-pill">max_frames 256</span><span class="visual-pill">비주얼 예산 16,384</span><span class="visual-pill">출력 상한 2,048</span><span class="visual-pill">BF16 · SDPA · greedy</span></p><p class="visual-note">통제한 설정값의 요약이다. 모델마다 프로세서와 실제 입력 토큰 수는 다르다.</p></section>'''
 quality=f'''<section class="visual-panel" aria-label="성공의 서로 다른 의미"><div class="visual-kicker">본실험 40회 기준 · 서로 다른 지표</div><h4>응답이 나왔다는 것과 정답을 찾았다는 것은 다르다</h4><div class="visual-grid"><div class="visual-card">응답 생성<strong>{ok}/{n}</strong><small>실행 예외 없이 생성</small></div><div class="visual-card">출력 잘림 없음<strong>{complete}/{n}</strong><small>3회는 출력 상한 도달</small></div><div class="visual-card">엄격한 형식 준수<strong>{strict}/{n}</strong><small>숫자 쌍의 JSON 배열<br>빈 배열도 구문상 통과</small></div><div class="visual-card pending">실제 검색 정확도<strong>아직 미측정</strong><small>사람의 정답 구간 필요<br>0%라는 뜻이 아님</small></div></div><p class="visual-note">모든 카드의 분모는 같은 40회다. 단계별 탈락률이 아니며, 형식 준수도 의미상 정답을 보장하지 않는다. 기본 지시 대조 8회는 제외했다.</p></section>'''
 conclusion='''<section class="visual-panel" aria-label="현재 증거와 다음 검증"><div class="visual-kicker">결론의 경계</div><h4>확인한 것은 다중 출력 행동, 다음 목표는 다중 사건 회수</h4><div class="visual-grid two"><div class="visual-card"><strong>이번에 확인</strong><ul><li>TimeLens2 이외 모델도 여러 시간 쌍 생성</li><li>모델·쿼리·프롬프트에 따른 출력 차이</li><li>잘림·범위 오류·파서 호환성 및 실행 비용</li></ul></div><div class="visual-card pending"><strong>다음에 검증</strong><ol><li>원본 영상 전체 검수 → 모든 사건의 GT</li><li>일대일 매칭 → 누락·오탐·경계 평가</li><li>FPS·예산 비교 → 정확도와 비용</li><li>별도 영상 → 일반화 확인</li></ol></div></div><p class="visual-note">후속 FPS 비교는 고정 비주얼 예산에서 시간·공간 정보 배분을 바꿀 수 있다. 현재 결과로 정확도 우승 모델을 선정하지 않는다.</p></section>'''
 for anchor,panel in [('<h3>3.1 입력 영상과 실행 환경</h3>',design),('<h3>4.1.1 실행 성공·형식 준수·복구 가능성의 분리</h3>',quality),('<h3>6.1 발표 결론</h3>',conclusion)]:
  assert body.count(anchor)==1,anchor
  body=body.replace(anchor,anchor+panel)
 return body
