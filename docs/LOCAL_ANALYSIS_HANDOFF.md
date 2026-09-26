# GPU 없는 로컬 Codex 인수인계

## 작업 목적과 범위

농구 영상에서 자연어 쿼리에 맞는 시간 구간을 찾는 temporal video grounding 실험이다. 핵심 질문은 (1) 반환 구간이 실제로 쿼리와 맞는가, (2) 동일 사건이 여러 번 등장할 때 여러 구간을 반환하는가, (3) 모델·프롬프트·입력 설정에 따라 결과와 비용이 어떻게 달라지는가이다.

GPU 서버 실험은 끝났고 서버를 삭제할 예정이다. 앞으로는 **보관된 결과를 CPU 로컬에서 재분석하고 문서·그래프를 개선**한다. 요청 없이 모델 추론, 가중치 다운로드, CUDA 설치, `setup_elice.sh install all`, 배치 실험 재실행을 하지 말 것. 기존 결과를 정답으로 간주하지 말 것.

저장소: `https://github.com/DKU-AIM/video-motion-pipeline`

작업 브랜치: `experiments/basketball-grounding-study` (main이 아님).

```bash
git clone --branch experiments/basketball-grounding-study https://github.com/DKU-AIM/video-motion-pipeline.git
cd video-motion-pipeline
```

인수인계 직전 커밋: `b81180e` 실험 코드·초기 분석, `fb36f4e` 원자료 보관, `cf55eaa` 최신 보고서·파라미터 분석. 이 문서 작성 직전 원격 브랜치와 로컬 HEAD가 `cf55eaa`로 같았음을 확인했다. 이 문서는 이후 추가된 커밋이므로 별도 push가 필요하다.

## 먼저 읽을 자료

1. [보관 자료 안내](../experiments/basketball/artifacts/README.md): 원자료와 실패 이력, 파일 의미, 무결성 확인.
2. [실험1 보고서](BASKETBALL_V2_PRESENTATION_REPORT.md), [실험1 비판적 검토](BASKETBALL_V2_CRITICAL_REVIEW.md).
3. [실험2 보고서](BASKETBALL_EXPERIMENT2_PRESENTATION_REPORT.md), [실험2 비판적 검토](BASKETBALL_EXPERIMENT2_CRITICAL_REVIEW.md).
4. [다중 구간 출력 문헌 조사](GROUNDING_MULTI_SPAN_LITERATURE.md): 공식 논문·코드·모델 카드 출처가 있다. 문헌 주장을 추가/변경할 때 원문을 다시 검증할 것.
5. [실험2 분석 폴더 안내](analysis/basketball_parameters/README.md).

보고서와 같은 이름의 `.html`을 로컬 브라우저에서 열면 그림과 보고서를 볼 수 있다. 실험2 HTML에는 모델·쿼리·조건별 예측 탐색기가 있고 영상은 내장되어 있지 않다. HTML 자체의 그림과 탐색기는 오프라인 사용 가능하다. GitHub 웹에서는 Markdown과 PNG로 읽고, HTML은 내려받아 연다.

## 실험 명칭과 설계

| 명칭 | 폴더 | 설계 |
|---|---|---|
| 초기 quick 비교 | `basketball_quick` | 8모델 × 4쿼리, 최종 8작업 성공. 실패·수정·재시도 포함 |
| 실험1 (v2) | `basketball_v2` | 8모델 × all-json 5쿼리 = 40회 + 모델별 native 슛 쿼리 대조 8회, 총 48회 |
| 실험2 (parameters) | `basketball_parameters` | TimeLens2-4B·TimeLens-8B × 4조건 × 5쿼리 = 40회 |
| 과거 진단 | `basketball_all*`, `basketball_timing_check` | 초기 중단·경로 오류·시간 측정 기록. 실험1/2 정상 통계에 합치지 않음 |

실험1 모델은 TimeLens2-4B/8B, TimeLens-8B/7B, Time-R1-7B/3B, Qwen3-VL-8B, Qwen2.5-VL-7B다. 공통 쿼리 ID는 `shot`, `dark_shot`, `bench`, `closeup`, `football`이며 실제 영문 쿼리는 `settings.json`과 [queries.json](../experiments/basketball/queries.json)에 있다. `football`은 부정 쿼리 후보지만 GT 검수로 부재를 확정한 것은 아니다.

실험2 조건:

| 조건 | FPS | total_tokens (비주얼 예산) |
|---|---:|---:|
| A | 0.5 | 32768 |
| B | 1 | 32768 |
| C | 2 | 32768 |
| D | 1 | 16384 |

실험2는 all-json, max_frames=512, max_new_tokens=2048, 쿼리별 1회 실행이다. 모델별 시간 제한은 두지 않았다. 고정 순서이며 반복 측정/순서 통제 실험이 아니다. 완전한 조건은 각 `plan.json`과 `settings.json`을 우선한다.

## 원자료의 위치와 우선순위

- `experiments/basketball/artifacts/`: 서버 결과를 보존한 원본 복사본. 464개 원본 파일(약 4.06 MB)과 README·manifest·로그 추적 규칙. `manifest.json`에 원본 경로, 시간, 크기, SHA-256이 있다. 마지막 서버 점검에서 누락/변경 없음.
- `docs/analysis/basketball_v2/`: 실험1 분석, 원문 스냅샷, 그래프, 검증된 실행 코드 사본.
- `docs/analysis/basketball_parameters/`: 실험2 분석, 40회 원문 스냅샷, CSV·JSON, 그림 8종, 탐색기, 검증 코드.
- 실험2 `source/*/model-config.json`은 기록된 모델 리비전의 설정 사본이다. 분석용 patch/grid 정보 확인에 가중치나 Hugging Face 캐시는 필요 없다.
- `verified_source_snapshot/`은 실행 계획의 해시와 비교한 코드 사본이다. 미래에 수정될 현재 코드와 실행 당시 코드를 혼동하지 말 것.

원문은 `results.jsonl`의 `raw`이다. `spans`는 파서 결과이며 `legacy_spans`, `parse_status`, `hit_output_limit`, `incomplete_output` 등을 함께 확인한다. `results.normalized.jsonl`은 재해석 결과로 원본과 구분한다. aggregate 파일과 attempt 파일을 함께 합산하면 중복된다. 최종 attempt는 `state.json`으로 선택한다.

## GPU 없이 가능한 작업

- HTML/Markdown 열람, CSV·JSON·JSONL 재집계.
- 원문 응답과 파싱 구간 비교, 출력 잘림·엄격 JSON 형식·구간 범위 분석.
- 구간 합집합 길이·연결 성분·조건 간 예측 일치도·속성 일관성 진단.
- 전처리/생성/전체 시간, 토큰, 프로세스 RAM, GPU 메모리 피크 등 이미 측정된 비용 비교.
- 로컬 원본 영상이 있으면 수동 GT 라벨링과 예측 구간 검수.

기존 CPU 분석 테스트는 표준 라이브러리로 실행된다. 인수인계 시 3개 모두 통과했다.

```bash
python3 docs/analysis/basketball_parameters/test_analysis.py -v
```

그래프/보고서 재생성 코드에는 `numpy`, `matplotlib`, `markdown-it-py`가 사용된다. 브라우저 자동 검증은 선택적으로 `playwright`와 Chromium이 필요하다. 서버 전체 `pip-freeze`나 PyTorch를 로컬 분석 환경에 그대로 설치하지 말고 필요한 패키지만 설치한다. 한글 그림은 로컬 폰트 가용성도 확인한다.

## 첫 구현 과제: 재분석 경로 이식

**현재 재생성 스크립트는 그대로 실행하면 서버 경로 때문에 실패할 수 있다. `--run` 변경만으로는 충분하지 않다.**

- 실험1 `docs/analysis/basketball_v2/build_analysis.py`: `state[job_id]['attempt_dir']` 절대경로를 그대로 읽는다.
- 실험2 `docs/analysis/basketball_parameters/build_analysis.py`: 자기 실험과 `--experiment1` 양쪽의 `attempt_dir`에 같은 문제가 있다.
- 실험1 `critical_audit.py`: `derived_metrics.json`의 `source_run` 절대경로로 `plan.json`을 다시 읽는다.
- 실험2 설정 읽기는 모델 캐시가 없으면 이미 Git에 있는 `source/<job_id>/model-config.json`으로 대체한다. 이 사본을 지우지 말 것.
- 재생성 코드가 보고서와 분석 산출물을 덮어쓰므로 실행 전 `git status`, 실행 후 diff를 확인한다.

권장 수정: 원자료 JSON을 고치지 말고 코드에 경로 해석을 추가한다. 선택된 run 루트 `experiments/basketball/artifacts/<실험>/`를 기준으로 job ID와 기존 attempt 디렉터리 이름을 조합해 로컬 파일을 찾는다. 모든 재시도를 합치거나 항상 attempt_001로 고정하지 말 것. 보관본을 우선하도록 해 서버 절대경로가 우연히 존재해도 다른 자료를 읽지 않게 한다. 찾지 못하면 조용히 다른 실행으로 대체하지 말고 오류를 내도록 한다.

`critical_audit.py`는 명시적인 run 인자나 보관한 `source_plan.json`으로 원자료 위치를 해석하도록 개선할 수 있다. 원문/해시 manifest와 검증 소스 사본은 유지한다. 이식 후 48회/40회 행 수, 선택 attempt, 기존 CSV 주요 값과 원문 동일성, 모델 config fallback, 원본 해시 불변을 검사한다.

## 해석에서 지켜야 할 구분

- **다중 구간 생성 가능 / 모든 일치 장면을 찾아냄 / 각 구간이 정확함은 별개**다.
- TimeLens2의 native 프롬프트도 모든 구간을 찾으라는 지시를 포함한다. 다른 모델의 native는 단일 구간 중심인 경우가 있어 native 대 all-json을 모든 모델에 동일한 조작으로 취급하면 안 된다. 정확한 문구는 각 `prompt-template.txt`와 실행 코드에서 확인한다.
- 다른 모델도 all-json에서 여러 구간을 출력했다는 관찰만으로 원래 학습 목표나 exhaustive recall 능력을 증명하지 않는다. 'TimeLens2만 다중 출력 가능' 같은 절대적인 주장도 피한다.
- GT 구간 검수가 아직 없다. 반환 개수, 엄격 JSON, 실행 success, 예측 Jaccard, 속성 밖 비율은 의미 정확도/Recall이 아니다. 빈/빈 예측 Jaccard는 undefined로 취급한다.
- FPS 변경은 고정 비주얼 예산에서 시간·공간 배분을 함께 바꿀 수 있다. grid 환산값을 직접 기록한 샘플 시각/실제 디코딩 장수라고 부르지 않는다.
- 시간 표의 쿼리 5개는 동일 조건 5회 반복이 아니다. 모델 로딩, 첫 호출, 쿼리별 출력 길이, 입력 토큰 차이를 구분한다.
- 실험1 D 대응과 실험2 D의 10쌍 원문/구간/격자/입력 토큰 일치는 이 데이터의 관찰이며 모든 실행의 결정성을 보장하지 않는다.
- GPU는 A100X MIG 3g.40gb였다. 프로세스 CUDA allocated/reserved, 프로세스 RAM, 물리 GPU 센서 값을 구분한다. 물리 GPU 전력을 이 실험의 전용 에너지로 환산하지 않는다.
- 과거 이슈: 따옴표·한글 영상 경로 처리, Qwen2.5 입력 어댑터 분기, Time-R1 cache 비활성으로 인한 느린 생성, 다중 출력 파서/잘림 처리. 해당 수정 이력과 실패 attempt는 지우지 말고 재분석 근거로 활용한다.

## 영상과 남은 과제

영상 파일명: `2026_AsianGames_men's_basketball_final.mp4` (파일명이 실제 경기 정체를 검증하는 근거는 아님).

- 파일 크기: 87,446,365 bytes, 비디오 길이 약 222.556초, HEVC.
- SHA-256: `98ac45493e0d3a17bf307c62c4482168e5d92ce32edec4378e3634da7cd10a50`.
- 영상은 Git에 없다. 사용자 PC의 별도 사본을 사용한다. HEVC 브라우저 재생 지원은 로컬 환경에 따라 다르다.
- 영상 없이도 기록 기반 비용·출력 분석은 가능하나 의미 정확도 검수는 진행할 수 없다.

권장 작업 순서: 보고서/검토문 읽기 → 원자료 무결성 확인 → CPU 경로 이식 → 기존 결과 재현 → 필요한 비교 시각화 → 사용자가 영상 제공 시 GT 검수. GT는 `experiments/basketball/annotations.template.json`을 참고하되 템플릿을 정답으로 사용하지 않는다. `grounding/evaluate_basketball.py`의 영상 일치 검증과 일대일 IoU 매칭을 확인한 뒤 평가한다. 추가 모델 추론은 별도 GPU 작업으로 남긴다.

## 로컬 Codex에 바로 전달할 요청

> `docs/LOCAL_ANALYSIS_HANDOFF.md`와 연결된 두 실험 보고서·비판적 검토를 읽어줘. 지금은 GPU 없는 로컬에서 이미 저장된 농구 grounding 결과를 분석하는 단계다. 추론이나 모델 설치는 하지 말고 보관 원본과 manifest를 유지해줘. 먼저 자료 무결성과 CPU 테스트를 확인하고, 분석 코드가 옛 서버의 절대경로 없이 저장소의 artifacts를 읽도록 수정해줘. 특히 state.json의 attempt_dir, v2 audit의 source_run, 실험2 모델 config fallback을 확인해줘. 기존 48회/40회 결과를 재현한 뒤 정확도와 출력 행동을 구분해 후속 분석을 이어가자.
