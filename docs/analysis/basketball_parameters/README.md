# 실험2 분석 산출물

주 보고서: [HTML](../../BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.html), [Markdown](../../BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.md). 자체 검토: [주장과 한계](../../BASKETBALL_EXPERIMENT2_CRITICAL_REVIEW.md).

완료된 2모델 × 4조건 × 5쿼리 = 40회만 분석했다. 추가 추론은 실행하지 않았다. 보고서의 예측 간 일치도와 반환 개수는 정답 정확도 지표가 아니다.

- `source/`: 실행 계획·상태, 세션 설정·자원 로그·환경 정보 사본과 실험1 대응 응답. 모델 config는 실행에 기록된 commit의 캐시에서 보존했다.
- `source_manifest.json`: 원자료 출처와 SHA256. `verified_source_snapshot/`에는 기록된 해시와 일치하는 실행 코드 사본이 있다.
- `response_snapshot.jsonl`, `derived_metrics.json`, CSV들: 원문과 계산 결과.
- `01_`부터 `08_` 그림: PNG 및 편집·확대용 SVG. HTML에는 PNG가 내장되어 있다.
- `build_analysis.py`: 원자료에서 지표·그림·보고서를 다시 생성한다. matplotlib와 markdown-it-py를 이용한다. 현재 서버의 원자료 경로를 사용하므로 다른 서버에서는 실행 경로 조정이 필요하다.
- `test_analysis.py`: 구간 집합 연산·엄격한 출력 형식 판별·데이터 대응 검증.
- `check_browser.py`, `browser_validation.json`: Chromium에서 그림 8개, 선택 UI, 구간 선택, 링크, 모바일 가로 넘침과 JS 오류를 검증한 코드·결과. 선택한 로컬 영상의 실제 디코딩·재생은 이 검증에 포함하지 않는다.

저장소 루트에서 재생성:

```bash
/home/elicer/aim-workspace/grounding_env/bin/python docs/analysis/basketball_parameters/build_analysis.py
/home/elicer/aim-workspace/grounding_env/bin/python docs/analysis/basketball_parameters/test_analysis.py -v
```

HTML 자체의 그래프·탐색기는 인터넷 연결 없이 열린다. 부속 CSV·소스 링크를 함께 공유하려면 이 폴더도 보존한다. 영상은 내장하지 않았으며 사용자가 같은 원본을 선택할 때만 브라우저에서 로컬로 연다. HEVC 재생 지원은 브라우저 환경에 따라 다르다.
