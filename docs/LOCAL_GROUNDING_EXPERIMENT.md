# 로컬 영상으로 하는 그라운딩 위치·쿼리 안정성 실험

목적: 같은 춤 장면이 앞/뒤로 이동하거나 검색 문장의 표현이 달라져도 TimeLens2가 해당 구간을 찾는지 비교한다.

입력은 이미 설치된 `multi-hmr2/demo_data/sample_video.mp4`다. 프레임 확인 결과 여러 사람이 실내에서 춤추는 약 17.24초 영상이다. 추가 데이터셋을 다운로드하지 않는다. MotionGPT의 렌더링 영상은 도메인이 달라 이번 비교에 섞지 않는다.

- 영상: 검은 화면 4초 + 원본 + 검은 화면 12초 / 검은 화면 12초 + 원본 + 검은 화면 4초.
- 비교: TimeLens2-4B와 TimeLens2-8B, 각 3개 영어 표현. 총 12회 추론.
- 고정 조건: 비주얼 토큰 예산 8192, FPS 2, 최대 프레임 96, 출력 토큰 256, SDPA, greedy decoding.
- 기준: 전처리된 원본이 삽입된 시간 구간. 이 구간은 사람의 행동 경계를 수작업 라벨링한 정답이 아니다.
- 점수: 예측 구간 합집합과 기준 구간의 temporal IoU. 빈/잘못된 구간은 0점. 추론 실패도 계획된 전체 횟수의 분모에 포함한다.
- 질문: late의 예측이 early에서 약 8초 이동하는가? 세 표현에 따라 경계가 얼마나 달라지는가? 모델 크기에 따른 결과·시간·메모리 차이가 있는가?

검은 화면이 쉬운 단서이므로 이 실험 점수를 일반 영상 정확도나 공식 벤치마크 성능으로 해석하면 안 된다. 작은 진단 실험이며 통계적 우열을 입증하지 않는다.

## 실행

레포 루트에서:

```bash
bash scripts/run_local_grounding_experiment.sh
```

기본 workspace는 `~/aim-workspace`. 기존 `MOTION_WORKSPACE`, `GROUNDING_PYTHON` 설정을 지원한다. 첫 실행 시 Hugging Face 모델 가중치를 받는다. 두 모델은 순차 로드/해제한다. GPU/라이브러리 점검도 실행한다. 원본 파일은 수정하지 않는다.

4B만 먼저 실행하려면:

```bash
bash scripts/run_local_grounding_experiment.sh --models timelens2-4b
```

가중치 다운로드 없이 영상 준비만 확인하려면:

```bash
bash scripts/run_local_grounding_experiment.sh --prepare-only
```

OOM이 발생하면 새로운 실행에서 두 모델에 동일하게 `--total-tokens 4096`을 적용한다. 자동으로 예산을 바꾸지 않는다. 실패 시 종료 코드는 1이고 가능한 나머지 추론과 보고서 작성은 계속한다. 실패 기록을 확인한 뒤 재실행한다. 실행마다 새 결과 폴더를 만들며 기존 결과 폴더를 덮어쓰지 않는다. `--out`으로 새 경로를 지정할 수도 있다.

## 결과

터미널에 표시되는 `~/aim-workspace/results_grounding/local_dance_날짜_시간/`:

- `videos/early.mp4`, `videos/late.mp4`: 눈으로 검수할 합성 입력.
- `manifest.json`: 입력 SHA256, 정답 대신 사용하는 삽입 구간, 쿼리, 설정, 팀 코드 커밋.
- `results.jsonl`: 원문 출력, 구간, 점수, 오류, 시간, 메모리.
- `summary.csv`: 스프레드시트 비교용.
- `report.md`: 모델별 요약. 실행 중에도 갱신된다.
- `pip-freeze.txt`, `gpu.txt`: 실행 환경.

`wall_s`는 전처리와 추론을 포함하며 모델 다운로드/로딩은 제외한다. 첫 호출의 워밍업 효과가 있을 수 있어 엄밀한 속도 벤치마크는 아니다. GPU 메모리는 모델을 포함한 해당 쿼리의 PyTorch 최대 할당량이다. 원격 가중치 revision은 기존 Grounder의 기본값을 사용하므로 완전한 재현성 고정은 아니다.

실제 검증: 서버 원본을 이용한 영상 생성 및 길이/삽입 위치 확인, 점수 계산과 실패 보고 단위 테스트. 모델 전체 추론은 이 스크립트 작성 과정에서는 실행하지 않았다.
