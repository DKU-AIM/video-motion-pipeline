# 짧은 영상으로 전체 흐름 확인하기

이 데모는 **TimeLens → CoMotion → MotionGPT → HTML 검수 화면**을 차례로 실행합니다. 전체 영상 실험·벤치마크용 실행기는 아닙니다. 기본적으로 원본 첫 30초에서 한 쿼리를 추론한 뒤, 시간순 첫 유효 구간을 최대 6초만 사용합니다. 최소 2초의 연속 인물 트랙이 없으면 설명을 만들어내지 않고 중단합니다.

## 처음 한 번 준비

기존에 그라운딩 추론에 성공한 `~/vtg-env`를 그대로 사용할 수 있습니다. 새로운 서버에서는 [그라운딩 초기 설정](ELICE_BOOTSTRAP.md)을 먼저 실행합니다.

```bash
cd ~/video-motion-pipeline
bash scripts/setup_motion_demo.sh --prepare-only --install-system
```

CoMotion은 별도로 받은 **SMPL neutral v1.1.0**이 필요합니다. [모션 환경 안내](MOTION_SETUP.md)에 따라 정식으로 받은 파일을 서버에 올린 뒤 등록합니다. 이 파일이 없으면 공개 환경 준비는 가능하지만 메시 추론은 시작할 수 없습니다.

```bash
bash scripts/setup_motion_demo.sh --smpl /실제/업로드한/SMPL_NEUTRAL.pkl
source ~/motion-workspace/activate-motion.sh
export GROUNDING_PYTHON="$HOME/vtg-env/bin/python"
python3 scripts/run_motion_demo.py --check
```

새 그라운딩 설정 스크립트를 사용했다면 `export GROUNDING_PYTHON=...` 대신 `source ~/vtg-workspace/activate.sh`를 사용합니다. 모션과 그라운딩은 서로 다른 가상환경이며, 실행기는 해당 Python을 개별 프로세스로 실행해서 GPU 메모리를 단계 사이에 해제합니다.

## 샘플 한 번 실행

```bash
cd ~/video-motion-pipeline
source ~/motion-workspace/activate-motion.sh
export GROUNDING_PYTHON="$HOME/vtg-env/bin/python"

python3 scripts/run_motion_demo.py \
  --video ~/vtg-data/2Y8XQ.mp4 \
  --query "A man drinks water with a glass"
```

성공하면 터미널에 새 결과 경로가 표시됩니다.

```text
~/motion-workspace/runs/<실행시각-고유값>/
  index.html                 메시 비교 영상 + 인물별 설명
  review.zip                 내려받아 확인할 HTML/영상 묶음
  demo.json                  원본 시각, 쿼리, 실행 환경
  grounding.jsonl            모델 원문·예측 구간
  inputs/clip/full.mp4        실제 메시 입력 클립
  results/clip/full/comotion/ 메시 비교 MP4·원본 모션 파라미터
  motion_inputs/             22관절 모션과 트랙 정보
  results_motiongpt/         HumanML3D 특징과 설명 JSON
  logs/                      단계별 실행 로그
  status.json                진행 중/완료/실패 및 단계
```

웹 VS Code에서 `review.zip`을 다운로드하고 압축을 푼 다음 `index.html`을 브라우저에서 엽니다. HTML 파일만 내려받으면 영상 상대 경로가 끊기므로 ZIP 전체를 사용하세요. 비교 영상에는 입력과 트랙 ID가 붙은 메시가 좌우로 표시되고, 각 설명에는 원본 영상 기준 구간이 표시됩니다.

`review.zip`은 검수용입니다. 원본 모션 파라미터와 특징 배열, 모델 가중치는 포함하지 않습니다. 인스턴스를 삭제하기 전에 필요한 실험 데이터는 결과 폴더의 `results`, `motion_inputs`, `results_motiongpt`에서도 따로 보관하세요. 결과 폴더의 `ml-comotion`, `MotionGPT`, `assets`는 공용 모델 폴더를 가리키는 링크입니다.

## 기업 영상의 다른 부분 보기

먼저 [다운로드 안내](ELICE_BOOTSTRAP.md)로 영상을 받아 로컬 파일 경로를 확보합니다. 예를 들어 5분 지점부터 30초만 탐색하려면 다음처럼 실행합니다. 쿼리는 해당 영상에 맞게 바꿉니다.

```bash
python3 scripts/run_motion_demo.py \
  --video "$VIDEO_PATH" \
  --window-start 300 \
  --window-seconds 30 \
  --query "A person kicks a ball"
```

`--window-seconds` 상한은 60초, `--max-clip-seconds` 상한은 6초입니다. 원본 전체를 검색했다고 해석하면 안 됩니다. 예상 구간이 6초보다 길면 앞부분만 사용하며, 원래 예측과 실제 사용한 구간을 `demo.json`에 모두 기록합니다. 실행마다 새 폴더를 생성하고, 명시한 `--output` 폴더가 이미 존재하면 덮어쓰지 않습니다.

## 결과 해석과 오류

- 그라운딩은 **시각**을 찾습니다. 누가 요청한 행동을 하는지까지 확인하지 않습니다.
- 현재 캡셔닝 입력은 가장 긴 연속 트랙 최대 두 개입니다. 같은 인물이 여러 연속 구간으로 나뉘면 두 번 나올 수도 있습니다. 영상의 ID와 설명을 직접 검수해야 합니다.
- MotionGPT에는 3D 모션만 들어갑니다. 옷이나 컵·공 등 영상 객체를 관찰한 설명이 아닙니다.
- 원본 시각은 선택한 영상 구간과 프레임 인덱스로 환산한 값입니다. 재인코딩·프레임 반올림에 따른 프레임 수준 오차가 있을 수 있습니다.
- 메시 좌표는 카메라 기준이며 보정된 월드 좌표가 아닙니다. 원본 fps로 추론하고 비교 영상만 약 15fps로 샘플링할 수 있습니다.
- 모델 출력이 비었거나 2초 트랙이 없으면 실패로 표시합니다. 설명 JSON의 공식 HumanML3D 예제는 설치 확인용으로, 기업 영상 결과 화면에서 제외합니다.
- 실패 시 `status.json`, `logs/<단계>.log`를 확인합니다. 메시 상세 로그는 `results/clip/full/comotion/inference.log`와 `render.log`에 있습니다. 재실행은 새 결과 폴더를 생성하며 실패한 GPU 단계를 자동 재개하지 않습니다.

실제 SMPL 자산을 사용한 전체 GPU 실행은 별도 검증이 필요합니다. CPU 테스트/FFmpeg 테스트의 성공은 모델 추론이나 결과 품질 검증을 대신하지 않습니다.

## 2026-09-18 검증 기록

Elice Python 3.10.14 / A100 MIG 3g.40gb / NVIDIA driver 535.183.06에서 다음을 직접 확인했습니다.

- `setup_motion_demo.sh --prepare-only` 정상 종료 및 재실행. 그라운딩의 기존 `~/vtg-env`는 유지했습니다.
- pose/caption 두 환경의 `pip check`, PyTorch 2.5.1+cu121 CUDA 인식 통과.
- 기본 3D 도형을 실제 OSMesa로 64×64 래스터화하고 CUDA 행렬곱을 확인했습니다. 이것은 사람 메시 추론 결과가 아닙니다.
- 공식 HumanML3D `012314` 특징을 MotionGPT에 넣어 실제 GPU 캡셔닝 성공. 생성 단계 1.10초, 최대 할당 GPU 메모리 약 1.11GB(모델 다운로드·로딩 시간 제외). 생성된 설명의 정확도를 평가한 실험은 아닙니다.
- FFmpeg 4.4.2로 테스트 영상을 잘라 길이와 전체 디코딩을 검증했습니다.
- `setup_motion_demo.sh --check`의 남은 누락 항목은 SMPL neutral 파일입니다. SMPL 파일을 사용한 CoMotion 추론과 **전체 연결 실행·실제 검수 ZIP 생성은 아직 검증하지 못했습니다**.
