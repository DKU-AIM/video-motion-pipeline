# 2차 실험: 5쿼리 모델 비교 + 기본 지시 대조

1차에서 드러난 입력 어댑터, 생성 캐시, 출력 형식 해석, 출력 잘림을 보완한다. 전체 영상·8개 모델은 그대로 사용하고 파라미터 대량 탐색은 하지 않는다.

## 실행 범위

- 본실험: 8개 모델 × 서로 다른 쿼리 5개 × 1회 = 40회. 모든 구간을 JSON으로 요청하는 all-json 지시.
- 대조: 8개 모델 × 같은 슛 쿼리 1개 × 1회 = 8회. 저장소의 기본(native) 지시.
- 합계: 16개 세션, 48회 추론. 고유 검색 문장은 5개뿐이다.
- 모델별·전체 시간 제한 없음. 앞선 빠른 비교 결과는 별도 폴더에 그대로 보존.

## 쿼리

| ID | 실제 문장 | 목적 |
|---|---|---|
| shot | A player attempts a shot at the basket. | 반복 행동의 다중 구간 회수 |
| dark_shot | A player wearing a dark jersey attempts a shot at the basket. | 같은 행동에 유니폼 속성을 붙였을 때 구별하는지 |
| bench | Players on the bench celebrate. | 벤치 반응 |
| closeup | The camera shows a close-up of a player's face. | 화면 구성 |
| football | A player kicks a soccer ball into a goal. | 부재 후보의 오탐; 부재 확정은 사람의 전체 검수 필요 |

속성 쿼리는 팀명·선수 이름의 외부 지식 대신 영상에서 보이는 유니폼 색을 사용한다. 기존 annotations.template.json의 동일 ID로 정답을 기록할 수 있다.

## 이전 이슈 보완

1. 로컬 영상 경로를 URI로 인코딩하지 않는다. 작은따옴표·한글·공백 파일명 회귀 테스트 포함.
2. TimeLens-7B 전용 전처리는 해당 모델 키에만 적용한다. 기본 Qwen2.5는 표준 전처리 경로 사용.
3. 모든 모델에 추론 KV cache를 명시적으로 켠다. Time-R1 설정의 기본값에 맡기지 않는다.
4. 원문, 전체 구간 해석, 기존 파서 결과를 함께 저장한다. 객체 배열, 바깥 배열 없는 연속 쌍, 분:초 표기 지원.
5. 출력 상한을 512에서 2048토큰으로 높인다. 무제한 출력은 아니며 여전히 잘릴 수 있다. 마지막 토큰이 EOS인지도 확인해 정상 종료를 길이 제한으로 오인하지 않는다.
6. 미완성 배열에서 완결된 구간을 복구하더라도 partial/incomplete로 표시한다. 빈 응답과 해석 실패도 구분.
7. 생성 토큰 진행, 전처리·생성 시간, 메모리와 자원 시계열을 기록한다. 오류 발생 시 전체 traceback도 보존한다.

FPS 1, 최대 프레임 256, 비주얼 토큰 예산 16384, SDPA, BF16, greedy, 반복 1회는 고정한다. 원문이 길어진 만큼 이전 512토큰 실험과 시간 수치를 그대로 동등 비교하지 않는다. GPU/NVIDIA 사용률의 MIG 측정 제한도 동일하다.

## 동료의 관찰을 검증하는 방법

"TimeLens2 외에는 첫 구간만 출력한다"를 보편적 모델 능력 주장과 특정 기본 설정의 관찰로 나눠 본다.

- 모델 원문에도 구간이 하나뿐인가?
- 원문에는 여러 구간이 있는데 기존 파서가 하나만 남겼는가?
- all-json 지시로 바꾸면 기본 지시보다 구간 수/분리가 달라지는가?
- 구간이 늘었더라도 실제로 서로 다른 올바른 사건인가?

`prompt_comparison.md`에서 동일한 슛 쿼리의 기본 지시 vs all-json, 새 해석 vs 기존 파서를 나란히 본다. 프롬프트별 결과에 차이가 있어도 단 한 영상·한 쿼리이므로 일반화하지 않는다. 정확도/누락은 GT 라벨링 전까지 결론내리지 않는다. 여기의 native는 저장소가 정의한 기본 템플릿이다. 특히 Qwen 대조군에 적용된 TimeLens 형태의 템플릿을 모델 저자의 공식 기본 실행이라고 주장하지 않는다. 동료의 실제 실행 코드와 지시문이 같았는지는 아직 확인되지 않았다.

## 실행/결과

```bash
cd /home/elicer/video-motion-pipeline
bash scripts/run_basketball_v2.sh
```

계획만 확인: `bash scripts/run_basketball_v2.sh --dry-run`.

이미 `basketball-v2` tmux 세션에서 실행 중이면 중복 실행하지 않는다. 결과는 `~/aim-workspace/results_grounding/basketball_v2/`에 저장한다.

- `progress.md`: 진행 상태.
- `comparison.md`: 모든 모델/조건/쿼리별 구간·시간·메모리.
- `prompt_comparison.md`: 슛 쿼리의 지시문·파서 대조표.
- `all_responses.jsonl`: 원문과 legacy_spans, spans, 해석/잘림 상태.
- `all_trials.csv`: 통합 수치와 legacy_discards_spans 표시.
- `plan.json`: 정확한 실행 목록과 입력·코드 해시.

첫 본실험 40회를 먼저 실행하고, 기본 지시 대조 8회를 뒤에 실행한다. 실패한 경우 다음 모델로 진행하며 성공한 모델은 재실행하지 않는다. 이번 세대의 코드/계획은 별도 폴더에서 관리해 1차 결과와 섞지 않는다.
