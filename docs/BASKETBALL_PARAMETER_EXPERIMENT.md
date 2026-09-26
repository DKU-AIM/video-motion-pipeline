# FPS·비주얼 토큰 예산 비교: TimeLens2-4B / TimeLens-8B

이전 2차 실험 48회 종료 후 시작한다. 대상은 같은 농구 영상이며, 확정된 두 모델만 사용한다.

| 조건 | FPS | 비주얼 토큰 예산 | 최대 프레임 | 출력 상한 |
|---|---:|---:|---:|---:|
| A | 0.5 | 32768 | 512 | 2048 |
| B | 1 | 32768 | 512 | 2048 |
| C | 2 | 32768 | 512 | 2048 |
| D | 1 | 16384 | 512 | 2048 |

- A/B/C: 같은 비주얼 예산에서 요청 FPS를 바꾼다.
- B/D: 같은 FPS에서 비주얼 예산을 바꾼다.
- 2모델 × 4조건 × 5쿼리 × 반복 1회 = 40회(8세션).
- 조건 A의 두 모델 → B의 두 모델 → C → D 순서로 실행한다. 동시 GPU 실행은 하지 않는다.
- 모델별/배치 전체 시간 제한은 없다. 메모리 부족 등 실패가 발생하면 원래 설정과 오류를 보존하고 다음 세션으로 진행한다. 자동 예산 변경은 하지 않는다.

## 고정된 쿼리와 추론 방식

1. A player attempts a shot at the basket.
2. A player wearing a dark jersey attempts a shot at the basket.
3. Players on the bench celebrate.
4. The camera shows a close-up of a player's face.
5. A player kicks a soccer ball into a goal.

각 문장 1회, all-json 다중 구간 지시, BF16, SDPA, greedy decoding, use_cache=True. 파서 보정과 영상 어댑터 수정, 출력 잘림 표시, 원문·기존 파서 비교 및 토큰 진행 기록을 유지한다. 기본 프롬프트 대조나 추가 모델 실험은 이번 범위에 없다.

## 해석 시 구분할 것

비주얼 토큰 예산은 실제 입력 토큰 수의 엄격한 상한이 아니다. FPS를 높이면서 총 픽셀 예산을 고정하면 프레임당 공간 해상도가 달라질 수 있다. 따라서 이번 결과는 고정된 비주얼 예산 아래에서 샘플링 빈도를 바꾼 결과이며, 공간 해상도까지 고정한 순수한 시간 샘플링 실험과 구분한다. 각 쿼리의 실제 input_tokens, video_grid_thw를 함께 본다.

정답 구간은 아직 사람의 검수가 필요하다. 반환 구간 수 증가만으로 정확도 향상을 주장하지 않는다. 가능한 경우 동일한 GT로 사건별 Precision/Recall/F1과 누락·오탐을 평가한다. 1회 측정이므로 시간 차이에도 변동이 있을 수 있으며 첫 쿼리의 워밍업 영향을 구분한다. 실패한 호출은 속도 평균에서 제외하되 실패 수는 함께 표시한다.

## 실행

```bash
cd /home/elicer/video-motion-pipeline
bash scripts/run_basketball_parameters.sh
```

실행 전 계획만 확인: `bash scripts/run_basketball_parameters.sh --dry-run`.

이미 `basketball-parameters` tmux 세션에서 실행 중이면 중복 실행하지 않는다. 중단 후 같은 명령으로 재개할 수 있다. 성공한 세션은 건너뛰고, 실패 재시도는 `--retry-failed`로 선택한다.

## 결과

`~/aim-workspace/results_grounding/basketball_parameters/`

- `plan.json`: 정확한 40회 실행 계획과 코드/입력 해시.
- `progress.md`: 세션별 진행 상태.
- `parameter_summary.md`: 조건별 평균 전처리·생성·전체 시간, GPU 최대 메모리, 실제 입력 토큰 수, 불완전 응답 수.
- `comparison.md`: 쿼리별 예측 구간과 처리 시간.
- `all_trials.csv`, `all_responses.jsonl`: 통합 수치 및 원문.
- 모델·조건별 attempt 폴더: 상세 자원 시계열, 생성 진행 로그, 입력 설정, 원문 출력과 오류.
- 배치 로그: `~/aim-workspace/results_grounding/basketball_parameters.log`.

이전 실험 결과는 `basketball_quick`, `basketball_v2`에 그대로 보존한다.
