# 엘리스 새 인스턴스 실행 가이드 · AIM

> **대상:** 인스턴스를 생성 → 실험 → 결과 백업 → 삭제하고, 다음 실험 때 다시 생성하는 팀원.
> **검증 범위:** 2026-09-18 새 엘리스 Ubuntu 22.04 / A100 MIG 20GB에서 설치, TimeLens2-4B 추론, CoMotion·Multi-HMR 2 원본/크롭 메시, MotionGPT·MG-MotionLLM 설명 생성을 확인했습니다. [실제 검증 기록](ELICE_VALIDATION.md). 다른 GPU·모든 그라운딩 모델·긴 영상의 정확도를 검증한 것은 아닙니다.

## 0. 무엇을 실행할지 선택

| 단계 | 모델 | 설치 이름 | Python 환경 | 하는 일 |
| --- | --- | --- | --- | --- |
| 1. 그라운딩 | TimeLens / TimeLens2 등 기존 CLI 지원 모델 | `grounding` | `grounding_env` | RGB 영상 + 쿼리 → 시간 구간 |
| 2. 메시 복원 | CoMotion | `comotion` | `pose_env` | 클립 → 인물별 SMPL 동작 |
| 2. 메시 복원 | Multi-HMR 2 | `multihmr2` | `pose_env` | 클립 → Anny 메시 |
| 3. 모션 캡셔닝 | MotionGPT | `motiongpt` | `env` | 변환한 3D 모션 → 설명 |
| 3. 모션 캡셔닝 후보 | MG-MotionLLM | `mgllm` | `env` | 모션 특징 → 기본/상세 설명 |

`pose`는 CoMotion과 Multi-HMR 2를 함께 설치합니다. 현재 `elice.sh pose` 비교 실험은 두 모델을 모두 사용하므로 `pose`로 설치하세요. 모델별 설치 이름은 부분 재설치에도 사용할 수 있습니다. `all`은 다섯 모델 모두입니다.

모델은 외부 추론 API를 호출하는 방식이 아니라 서버의 GPU에서 실행합니다. GitHub에는 모델을 실행하는 팀 코드와 설치 설정만 있고, 가중치는 공식 배포처에서 따로 받습니다. VLM 설명 생성은 아직 모델/실행 코드가 확정되지 않았으므로 설치 목록에 없습니다.

## 1. 새 인스턴스 생성 및 SSH 접속

- Ubuntu 22.04, x86_64, Python 3.10과 NVIDIA 드라이버가 준비된 GPU 이미지를 선택합니다.
- 기존 실험 계열은 메시·캡셔닝에서 PyTorch 2.5.1 / CUDA 12.1을 사용합니다. 그라운딩 설치는 PyTorch 2.6.0 / CUDA 12.4입니다.
- 전체 설치 기준 NVIDIA 드라이버 550.54.14 이상을 권장합니다. `nvidia-smi`의 CUDA 표시는 드라이버 지원 범위이며, 별도 CUDA toolkit을 같은 버전으로 설치하라는 뜻이 아닙니다.
- Blackwell 등 새로운 GPU 아키텍처는 이 구형 메시/캡셔닝 조합의 지원을 별도로 확인해야 합니다. 단순히 torch만 최신으로 올리지 마세요. 설치 중 실제 CUDA 행렬 연산이 실패하면 중단합니다.

엘리스 화면에서 **현재 인스턴스의 SSH 명령**을 복사해 접속합니다. 인스턴스를 다시 만들면 주소·포트가 달라질 수 있으므로 예전 값을 그대로 쓰지 않습니다.

```bash
python3 --version
nvidia-smi
```

## 2. 팀 코드 받기

비공개 레포이므로 GitHub 계정에 레포 접근 권한과 SSH 키 또는 Git 인증 설정이 필요합니다. PEM은 엘리스 접속용이며 GitHub 인증용이 아닙니다.

```bash
git clone https://github.com/DKU-AIM/video-motion-pipeline.git
cd video-motion-pipeline
export MOTION_WORKSPACE="$HOME/aim-workspace"
```

영상·모델·결과는 레포 밖 `$MOTION_WORKSPACE`에 저장합니다. SSH 키나 접근 토큰을 코드에 넣지 않습니다.

## 3. 시스템 패키지 설치 — 인스턴스마다 한 번

```bash
bash scripts/setup_elice.sh system
```

FFmpeg, Python venv/개발 헤더, OSMesa 렌더링 라이브러리, Git, curl, tmux를 설치합니다. NVIDIA 드라이버는 이 스크립트가 설치하지 않으며 GPU 이미지에 준비되어 있어야 합니다.

긴 설치는 터미널 연결이 끊겨도 유지되도록 tmux에서 진행합니다.

```bash
tmux new -s aim
```

설치 전에 계획만 확인하려면 아래 명령을 사용합니다. 파일 생성·다운로드·GPU 실행은 하지 않습니다.

```bash
bash scripts/setup_elice.sh install all --dry-run
```

## 4. 모델별 의존성 설치 — 필요한 단계만

### 4-1. 그라운딩

```bash
bash scripts/setup_elice.sh install grounding
```

### 4-2. 메시 복원: CoMotion + Multi-HMR 2

```bash
bash scripts/setup_elice.sh install pose
```

두 모델은 호환되는 메시 전용 환경을 공유합니다. 그라운딩·캡셔닝 환경의 site-packages를 가져다 쓰지 않습니다.

### 4-3. MotionGPT

```bash
bash scripts/setup_elice.sh install motiongpt
```

### 4-4. MG-MotionLLM — 비교할 때만

```bash
bash scripts/setup_elice.sh install mgllm
```

전체 모델이 필요하면 위 네 명령 대신 한 번에 실행할 수 있습니다.

```bash
bash scripts/setup_elice.sh install all
```

**설치가 끝났다는 메시지는 추론 준비 완료라는 뜻이 아닙니다.** 다음 가중치·자산 준비와 점검이 필요합니다. 설치 스크립트는 고정한 공식 코드 커밋을 사용하며, 이미 있는 모델 소스가 수정되어 있거나 다른 커밋이면 덮어쓰지 않고 중단합니다.

## 5. 가중치와 필요한 파일 준비

설치한 모델에 대해 실행합니다. 다운로드에 실패하면 오류를 확인하고 같은 명령을 다시 실행하세요. 구글 드라이브 쿼터·접근 제한은 별도 해결이 필요합니다.

```bash
bash scripts/setup_elice.sh assets pose
bash scripts/setup_elice.sh assets motiongpt
# MG-MotionLLM도 사용할 때만
bash scripts/setup_elice.sh assets mgllm
```

그라운딩은 실제 `--model`로 선택한 모델의 가중치를 첫 추론 때 받습니다. MotionGPT의 FLAN-T5 캐시, Anny 등의 추가 자산 때문에 첫 실행 시 인터넷 연결이 필요할 수 있습니다.

### 별도로 가져올 파일

| 파일 | 위치 | 필요한 이유 |
| --- | --- | --- |
| SMPL_NEUTRAL.pkl | `$MOTION_WORKSPACE/assets/SMPL_NEUTRAL.pkl` | CoMotion의 SMPL 신체 모델. 공식 이용 조건에 맞게 준비 |
| reference_joints.npy | `$MOTION_WORKSPACE/assets/reference_joints.npy` | MotionGPT 변환에서 기준 골격의 길이 계산 |
| reference_features.npy | `$MOTION_WORKSPACE/assets/reference_features.npy` | 기존 코드의 기준 모션 캡셔닝 확인 사례 |

`reference_features.npy`는 기존 실험의 기준 자료를 복원해야 합니다. `reference_joints.npy`가 없으면 해당 특징으로부터 공식 변환 함수로 복원할 수 있습니다. 이름만 맞는 임의 파일이나 0 배열로 대체하면 안 됩니다. 원본 기준 특징은 GitHub에 업로드하지 않습니다. 기준 특징을 먼저 복원하고 `assets motiongpt`를 실행하면 관절 파일이 자동 생성됩니다. 나중에 복원했다면 다음 명령을 실행합니다.

```bash
"$MOTION_WORKSPACE/env/bin/python" scripts/prepare_motion_reference.py "$MOTION_WORKSPACE"
``` 준비하지 않으면 `check motiongpt`가 실패합니다.

```text
~/aim-workspace/
├── env.sh                      ← 설치기가 생성하는 경로 설정
├── grounding_env/              ← 그라운딩 Python
├── pose_env/                   ← 메시 Python
├── env/                        ← 캡셔닝 Python
├── ml-comotion/                ← 고정 커밋 공식 코드
├── multi-hmr2/
├── MotionGPT/
├── MG-MotionLLM/                ← 선택 설치
├── assets/                     ← 가중치·SMPL·기준 모션
├── inputs/                     ← 클리핑·크롭 입력
├── clip_manifest.json
└── results/                    ← 추론 후 생성
```

## 6. 설치 상태 확인

```bash
source "$MOTION_WORKSPACE/env.sh"
bash scripts/setup_elice.sh check grounding
bash scripts/setup_elice.sh check pose
bash scripts/setup_elice.sh check motiongpt
# 설치했다면
bash scripts/setup_elice.sh check mgllm
```

점검 내용: Python 실행 파일, 필수 파일 존재, CUDA 실제 연산, 주요 모델 import, 메시 환경의 OSMesa 렌더링, 기준 모션 배열 형태.

`MISSING`이나 오류가 나오면 해결한 뒤 다시 점검합니다. 체크포인트의 전체 로딩과 실제 영상 추론, 결과 품질 검수는 별도입니다. `check`는 입력 영상과 manifest가 없어도 환경만 확인할 수 있습니다.

## 7. 입력 영상 준비

그라운딩은 원본 영상과 검색 문장을 사용합니다. 이미 사람이 확인한 시간 구간이 있으면 그라운딩을 다시 실행하지 않고 클리핑한 영상부터 사용할 수 있습니다.

메시 비교에는 **동일 시간 구간의 원본 클립 `full.mp4`와 크롭 클립 `crop.mp4`**를 준비합니다. 자동 설치가 클리핑·대상 인물 선정을 수행하지는 않습니다.

```text
inputs/
└── sample01/
    ├── full.mp4
    └── crop.mp4
```

`clip_manifest.json` 예시 — 아래 값은 설명용이므로 실제 클립 정보로 바꿉니다. ROI는 원본 클립 픽셀 기준 `[x, y, width, height]`입니다.

```json
[
  {
    "id": "sample01",
    "source_start_seconds": 60.0,
    "category": "running",
    "roi_xywh": [100, 100, 300, 500]
  }
]
```

## 8. 순서대로 추론

레포 폴더에서 실행합니다. 재접속했다면 먼저 `source "$HOME/aim-workspace/env.sh"`를 다시 실행합니다.

### 8-1. 그라운딩 — 필요한 경우

```bash
bash scripts/elice.sh grounding --model timelens-8b \
  --video /path/to/video.mp4 \
  --query "A scene of Singing"
```

### 8-2. 원본·크롭 메시 비교

```bash
bash scripts/elice.sh pose
```

결과: `results/<id>/<full 또는 crop>/<comotion 또는 hmr2>/` 아래 메시 파라미터, 메타데이터, 비교 영상.

### 8-3. CoMotion 결과를 모션 캡셔닝 입력으로 변환

```bash
bash scripts/elice.sh export-motion
```

결과: `motion_inputs/manifest.json`, 인물별 관절 시퀀스.

**현재 공유 스크립트의 한계:** 원본·크롭 결과 양쪽에서 긴 연속 트랙을 자동 선택합니다. 사람이 지정한 한 명만 골랐다고 보장하지 않습니다. 캡셔닝 전에 manifest의 대상 ID와 실제 영상을 검수하고, 실험 대상 행만 남긴 manifest 사본을 사용해야 합니다. 크롭 영상에 한 사람만 충분히 크게 보이도록 준비하는 것도 필요합니다.

### 8-4. MotionGPT 캡셔닝

```bash
bash scripts/elice.sh motiongpt
```

입력은 RGB 영상이 아니라 변환된 3D 모션입니다. 현재 공유 코드의 지시문은 `Generate text: <motion tokens>`입니다. 결과는 `results_motiongpt/captions.json`과 사례별 `caption.json`, `features_263.npy`에 저장됩니다. 과거 PPT의 개별 상세 지시문 실험과 공유 실행본을 혼동하지 않습니다.

### 8-5. MG-MotionLLM 비교 — 선택

```bash
bash scripts/elice.sh mgllm
```

현재 공유 코드에서는 MotionGPT 단계에서 만든 `features_263.npy`를 읽으므로 8-4 이후 실행합니다. 결과는 `results_mgllm/captions.json`입니다. MotionGPT와 MG-MotionLLM을 모두 최종 채택했다는 의미는 아닙니다.

## 9. 오류를 만났을 때

| 증상 | 확인/조치 |
| --- | --- |
| Python 3.10 요구 오류 | Ubuntu 22.04/Python 3.10 이미지 사용. 필요하면 `ELICE_SETUP_PYTHON=python3.10`으로 지정 |
| CUDA unavailable / no kernel image | GPU 이미지, 드라이버와 GPU 아키텍처 확인. 이 버전 조합이 지원하지 않는 GPU일 수 있음 |
| MISSING SMPL/reference 파일 | 5번 파일 복원 후 `check` 재실행 |
| 다른 모델 커밋/수정 파일 감지 | 기존 폴더를 삭제하지 말고 새 workspace 지정 |
| 모델 다운로드 실패 | 다운로드 출처 접근/쿼터 확인 후 `assets <모델>` 재시도 |
| pip check 경고 | 아래의 headless 의존성 선택 설명 확인. 새 경고를 무조건 무시하지 않기 |
| 영상 manifest/입력 파일 오류 | 7번 폴더와 JSON 필드, 원본·크롭 파일 존재 확인 |

### 의존성 선택 및 재현성

- 직접 의존성은 `requirements/elice/`에 버전을 지정했습니다. 전이 의존성 전체의 완전한 lock 파일은 아니며, 설치 당시 `*-freeze.txt`를 보관합니다.
- CoMotion/Multi-HMR 2는 공식 코드 커밋을 고정하고, 추론에 쓰는 headless 의존성을 명시적으로 설치합니다. 공식 패키지는 `--no-deps`로 등록하여 GUI/학습용 패키지나 torch 버전이 다시 유입되지 않도록 했습니다.
- CoMotion의 coremltools/PyQt6/ruff/opencv-python 메타데이터와 실제 headless 설치(opencv-python-headless)는 다를 수 있습니다. PyRender 0.1.45의 PyOpenGL 3.1.0 고정 요구 대신 OSMesa용 3.1.7을 사용합니다. 따라서 `pip check`만으로 이 환경을 판정하지 않고 실제 import·렌더링을 검사합니다.
- 공식 모델 가중치는 배포처에서 받으며 모든 원격 모델 파일의 revision/hash를 고정한 완전 오프라인 재현 패키지는 아닙니다.
- 설치 중 만든 `setup-revisions.json`, `*-freeze.txt`를 실험 기록과 함께 보관하세요.

## 10. 인스턴스 삭제 전 백업

- [ ] `inputs/`, `clip_manifest.json`과 원본 영상 출처 기록
- [ ] `results/`, `motion_inputs/`, `results_motiongpt/`, `results_mgllm/` 중 생성된 결과
- [ ] `assets/reference_joints.npy`, `assets/reference_features.npy`와 이용 가능한 SMPL 자산
- [ ] `setup-revisions.json`, `*-freeze.txt`, 실행 로그와 사용한 팀 코드 커밋
- [ ] 로컬/NAS로 복사가 끝났고 파일이 열리는지 확인

가상환경 폴더는 다음 서버로 복사해 재사용하지 않고 이 가이드로 재설치합니다. 큰 가중치와 HF 캐시는 이용 조건에 맞는 영구 저장소에 보관하면 다음 다운로드 시간을 줄일 수 있습니다. 설치기 자체가 데이터를 외부로 백업해 주지는 않습니다.

## 다음 인스턴스에서 반복할 최소 흐름

```bash
# GPU Ubuntu 22.04/Python 3.10, Git 인증 준비 후
git clone https://github.com/DKU-AIM/video-motion-pipeline.git
cd video-motion-pipeline
export MOTION_WORKSPACE="$HOME/aim-workspace"
bash scripts/setup_elice.sh system
bash scripts/setup_elice.sh install pose
bash scripts/setup_elice.sh install motiongpt
bash scripts/setup_elice.sh assets pose
bash scripts/setup_elice.sh assets motiongpt

# 여기서 SMPL/reference 파일 및 입력 클립·manifest 복원
source "$MOTION_WORKSPACE/env.sh"
bash scripts/setup_elice.sh check pose && \
  bash scripts/setup_elice.sh check motiongpt && \
  bash scripts/elice.sh pose && \
  bash scripts/elice.sh export-motion && \
  bash scripts/elice.sh motiongpt
```

위 흐름은 환경 점검 실패 시 추론으로 넘어가지 않습니다. 실제 대표 클립을 먼저 확인한 뒤 전체 실험을 진행합니다.
