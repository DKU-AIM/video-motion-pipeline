# 발표용 시각화 활용 안내

추가 자료는 완료된 v2의 응답 사본에서 생성했다. 새로운 추론이나 정답 정확도 계산은 포함하지 않는다. HTML에는 기존 그림 2개와 추가 차트 3개를 이미지 데이터로 내장하고, 요약 패널 3개를 HTML/CSS로 배치했다. 외부 CDN이나 네트워크 연결 없이 그림을 볼 수 있다.

| 자료 | 발표에서 전달할 메시지 | 위치 / 내보내기 |
|---|---|---|
| 실험 구성 요약 | 독립 영상은 1개, 본실험 40회 + 대조 8회 | HTML 3.1 |
| 성공 지표 카드 | 생성 성공·잘림·형식 준수·정확도는 다른 개념 | HTML 4.1.1 |
| 모델×쿼리 반환 수 | 모델과 질문에 따라 출력 행동이 달라짐 | 4.2 · [PNG](query_count_matrix.png) · [SVG](query_count_matrix.svg) |
| 프롬프트 대응 비교 | 기본 지시의 단일 출력으로 다중 출력 불가능을 단정할 수 없음 | 4.3 · [PNG](prompt_count_comparison.png) · [SVG](prompt_count_comparison.svg) |
| 98개 → 2개 시간 구조 | 시간 쌍 개수와 별도 사건 개수는 다름 | 4.4 · [PNG](fragmentation_explainer.png) · [SVG](fragmentation_explainer.svg) |
| 증거와 다음 검증 | 현재는 출력 행동 진단, GT 기반 사건 회수 평가는 다음 단계 | HTML 6.1 |

PNG는 슬라이드에 바로 삽입할 수 있고 SVG는 확대해도 선명하다. HTML 패널은 브라우저 인쇄/PDF 저장 시에도 나타난다. 요약 패널 외의 상세 표와 원문 분석은 근거 확인용으로 유지했다.

추천 발표 흐름: 실험 구성 → 반환 수 지도 → 프롬프트 비교 → 98개/2개 사례 → 성공 지표와 정확도 공백 → 기존 시간·메모리 차트 → 다음 검증. 발표 시간이 짧으면 반환 수 지도 대신 프롬프트 비교와 98개/2개 사례만 사용한다.

주의: 주황색은 잘림·범위 오류 표시이며 낮은 정확도 등급이 아니다. 0개는 파싱 실패 대체값이 아닌 빈 응답이다. 모든 시간 그림은 예측이며 GT가 아니다. 지표 카드는 같은 40회를 서로 다른 기준으로 센 것이므로 단계별 탈락 퍼널로 설명하지 않는다.

재생성 (저장소 루트):

```bash
/home/elicer/aim-workspace/grounding_env/bin/python docs/analysis/basketball_v2/build_presentation_visuals.py
/home/elicer/aim-workspace/grounding_env/bin/python docs/analysis/basketball_v2/render_report.py
```

데이터: `response_snapshot.jsonl`, `derived_metrics.json`. 형식 검사와 시간 합집합은 `critical_audit.py`를 재사용한다. 요약 패널의 응답·완결·형식 준수 수는 렌더링 시 원문에서 계산한다.
