# 1차 모델 비교: 8개 모델 × 4개 쿼리

현재 실행 범위는 총 32회다. 기존 1,664회 전체 실험은 중단했고 결과는 보존했다. FPS·토큰 예산·프롬프트 변형·쿼리 paraphrase·반복 측정은 후속 실험으로 미룬다.

## 동일하게 주는 네 쿼리

| ID | 실제 입력 문장 | 관찰 목적 |
|---|---|---|
| shot | A player attempts a shot at the basket. | 반복되는 슛 구간 회수 |
| bench | Players on the bench celebrate. | 특정 반응 장면 검색 |
| closeup | The camera shows a close-up of a player's face. | 화면 구성 검색 |
| football | A player kicks a soccer ball into a goal. | 부재 후보에 대한 오탐 관찰; 실제 부재는 전체 검수 후 확정 |

모든 모델에 모든 구간을 JSON 배열로 요청하는 `all-json` 지시를 사용한다. 모델 계열별 입력 형식과 시스템 메시지는 기존 어댑터를 유지한다. native 지시와의 차이는 후속 비교 대상이다.

## 모델과 고정 조건

순서: TimeLens2-4B → TimeLens2-8B → TimeLens-8B → TimeLens-7B → Time-R1-7B → Time-R1-3B → Qwen3-VL-8B → Qwen2.5-VL-7B.

모델마다 위 네 문장을 한 번씩만 실행한다. FPS 1, 최대 프레임 256, 비주얼 토큰 예산 16384, 출력 최대 512토큰, SDPA, BF16, greedy decoding을 사용한다. 출력 토큰 상한 도달 여부는 결과에 기록한다. 이는 벽시계 시간 제한과 다르다.

**사용자 요청으로 모델별 시간 제한과 배치 전체 강제 종료 시간 모두 비활성화했다.** 각 모델은 완료 또는 실행 오류까지 기다린다. 2시간 이내 결과 확인이 목표지만 다운로드·모델별 추론 속도에 따라 초과할 수 있다. 완료한 모델의 결과는 즉시 공개한다.

## 실행과 결과

```bash
cd /home/elicer/video-motion-pipeline
bash scripts/run_basketball_quick.sh
```

tmux `basketball-quick`에서 이미 실행 중이면 중복 실행하지 않는다.

- 결과 폴더: `~/aim-workspace/results_grounding/basketball_quick/`
- `comparison.md`: 모델·쿼리별 반환 구간, 시간, GPU 최대 메모리 비교. 약 10초마다 갱신.
- `progress.md`: 모델별 실행 상태, 완료 횟수.
- `all_trials.csv`: 통합 수치 결과.
- `all_responses.jsonl`: 모델 원문·구간·오류를 포함한 통합 응답.
- 각 모델 attempt 폴더: 상세 자원 시계열, 입력/출력 토큰, 전처리/생성 시간, 실제 지시문.
- 배치 로그: `~/aim-workspace/results_grounding/basketball_quick.log`.

정답 라벨 작성 전에는 정확도 순위를 산출하지 않는다. 구간을 많이 반환하는 것만으로 좋은 모델로 판단하지 않는다. 기존 정답 양식과 평가기로 이후 정밀도·재현율을 평가할 수 있다.

## 후속 전체 실험

전체 조건 코드는 남아 있다. 현재 배치에서 자동으로 이어서 실행하지 않는다. 나중에 명시적으로 실행할 때 새 폴더를 사용한다(이전 전체 실행은 코드 해시가 달라져 그대로 재개할 수 없다).

```bash
bash scripts/run_basketball_all.sh --timeout-hours 0 \
  --out "$HOME/aim-workspace/results_grounding/basketball_full_later"
```

기존 결과 폴더 `basketball_all`은 보존용이며 현재 진행 상황은 `basketball_quick`에서 확인한다.

## 실행 중 수정 기록: Time-R1 생성 캐시

Time-R1-7B의 실제 GenerationConfig에 `use_cache=false`가 들어 있어 첫 쿼리 생성이 12분 이상 지연됐다. 오류 예외는 없었지만 정상적인 속도로 보기 어려웠다. 해당 attempt를 보존하고 생성 호출에 `use_cache=True`를 명시해 재시작했다. 완료된 앞선 네 모델은 실제 GenerationConfig에서 이미 캐시가 활성화된 것을 확인했으므로 결과를 유지했다. 모델 config.json과 generation_config.json/default의 우선순위를 구분해 확인했다.

수정 이후에는 `generation_progress.jsonl`에 첫 토큰·16토큰 간격 또는 약 5초 간격의 진행 이벤트·생성 종료를 기록한다. 입력 프롬프트 토큰은 생성 개수에서 제외한다. 추가 계측의 작은 오버헤드가 있을 수 있다. 수정 전후 계획과 설정 확인 근거는 결과 폴더의 `plan.before-cache-fix.json`, `cache-fix-audit.json`에 보관했다. 시간 제한에 의한 중단이 아니라 비효율적인 추론 설정의 수정이다.

## 마지막 모델 실패 및 결과 해석 보완

Qwen2.5-VL-7B는 TimeLens-7B와 프롬프트 종류를 공유한다는 이유로 TimeLens 전용 프로세서 경로를 탔다. 영상 텐서 대신 metadata가 포함된 묶음이 기본 Qwen 프로세서에 전달돼 입력 준비에서 IndexError가 발생했다. 분기를 모델 키로 제한해 기본 Qwen은 표준 입력 경로를 쓰도록 수정하고 마지막 모델의 네 쿼리만 재실행했다. 최초 실패 attempt와 수정 전 계획은 보존한다.

저장된 모델 원문에는 다음 형식도 있었다. 새 파서는 숫자 쌍 배열 외에 객체 배열(`start_seconds`/`end_seconds`), 바깥 배열 없는 연속 쌍, 분:초 표기를 보존적으로 읽는다. 따라서 기존 TimeLens-7B의 슛 1구간 집계는 원문 기준 13구간으로, Time-R1-7B의 해석 실패는 슛 11구간으로 정정됐다. 이는 모델을 다시 돌려 얻은 결과가 아니라 같은 원문의 재해석이다.

Qwen3-VL-8B의 슛·클로즈업 출력은 512토큰 상한에서 잘렸다. 완결된 쌍만 복구하고 `partial_json`, `incomplete_output=true`, `OUTPUT LIMIT`로 표시한다. 이를 완전한 검색 결과나 정답 구간으로 해석하지 않는다. 출력 상한을 늘린 추가 실험은 이번 1차 비교에서 자동 수행하지 않는다.

원본 `results.jsonl`은 변경하지 않는다. 재해석 파일은 `results.normalized.jsonl`이며 기존 파서 결과도 `original_spans`/`original_parse_status`로 보관한다. 최상위 `comparison.md`, `all_trials.csv`, `all_responses.jsonl`은 재해석 결과를 사용하고 평가기도 정규화 파일이 있으면 우선 사용한다. 개별 attempt의 기존 `summary.csv`는 최초 파서 기준이므로 최종 비교에는 최상위 통합 CSV를 사용한다.

배치가 멈춘 상태에서 재집계만 실행하려면:

```bash
python3 grounding/normalize_basketball_results.py \
  --run "$HOME/aim-workspace/results_grounding/basketball_quick"
```
