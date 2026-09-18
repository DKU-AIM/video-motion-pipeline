# 엘리스 인스턴스를 새로 만든 날

GitHub 인증과 저장소 clone은 직접 진행한 뒤, **Python 3.10 / Linux x86_64 / NVIDIA GPU** 서버에서 실행합니다. 맥 터미널에서 실행하는 스크립트가 아닙니다.

```bash
cd ~/video-motion-pipeline
bash scripts/bootstrap_elice.sh
source ~/vtg-workspace/activate.sh
```

처음에 기업 영상의 다운로드 주소 또는 Synology 단일 파일 공유 주소를 붙여넣습니다. 입력은 화면에 표시하지 않습니다. 공유 주소·비밀번호·영상은 GitHub에 올리지 않습니다. 새 인스턴스에서는 주소도 다시 입력합니다. GitHub 연결 이후에 URL 입력까지 없애려면 별도의 비밀 설정 주입 수단이 필요하며, 이 스크립트는 그러한 외부 서비스를 만들지 않습니다.

## 자동으로 준비하는 것

1. 운영체제, CPU 아키텍처, GPU 드라이버와 Python 버전 확인.
2. `~/vtg-workspace/env`에 전용 가상환경 생성. 기존의 `~/vtg-env`와는 별개입니다.
3. PyTorch 2.6.0 CUDA 11.8, torchvision 0.21.0, Transformers 4.57.1, accelerate 1.6.0, qwen-vl-utils 0.0.14 설치.
4. 영상 읽기는 torchvision 사용. 해당 가상환경에 decord가 남아 있으면 제거.
5. 패키지 의존성과 실제 GPU BF16 행렬 연산 확인. 실패하면 모델·영상 다운로드 전에 중단.
6. TimeLens-8B 가중치와 processor 파일을 Hugging Face에서 다운로드.
7. 영상 파일 다운로드 후 PyAV로 첫 프레임 디코딩 및 영상 정보 확인.
8. 다시 사용할 활성화 파일과 실제 설치 버전·GPU·코드 커밋·모델 snapshot 기록 저장.

이 단계는 **환경 준비**입니다. 기업의 긴 영상에 자동으로 추론을 시작하지 않습니다. 영상 첫 프레임 검사는 전체 영상의 모든 프레임이 정상이라는 보장은 아닙니다. 제공기관의 SHA-256이 있다면 `VIDEO_SHA256`으로 전달하여 파일 일치도 확인합니다.

핵심 패키지 버전은 고정되어 있지만 PyAV·matplotlib과 전이 의존성 전체를 잠근 lockfile은 아닙니다. 실제 해결된 버전은 `manifests/requirements-resolved.txt`에 기록됩니다. 모델은 기본적으로 `main`을 받으며 실제 commit과 snapshot 경로를 기록합니다. `MODEL_REVISION=기록한커밋`은 사전 다운로드 대상만 지정합니다. 기존 그라운딩 CLI는 모델 ID의 기본 revision을 로드하므로, 이 옵션만으로 추론 모델의 과거 revision이 고정되지는 않습니다. 과거 실험의 완전한 재현에는 추론 로더의 revision 지정과 그때의 전체 패키지 버전 적용도 필요합니다.

## 예제 영상으로 먼저 확인

기업 영상을 받기 전에 설치 흐름을 확인하려면 공식 공개 예제를 사용합니다.

```bash
bash scripts/bootstrap_elice.sh --sample
source ~/vtg-workspace/activate.sh

run_dir="$(mktemp -d "$VTG_ROOT/results/smoke.XXXXXXXX")"
bash scripts/elice.sh grounding \
  --model timelens-8b \
  --video "$VIDEO_PATH" \
  --query "A man drinks water with a glass" \
  --out "$run_dir/results.jsonl"
```

`--out`을 지정하지 않으면 기존 추론 코드가 현재 폴더의 `results.jsonl`을 덮어쓰므로, 실행마다 위처럼 결과 폴더를 새로 만듭니다. 예제 한 번의 실행은 실제 연구 영상에서의 정확도 검증이 아닙니다.

## 저장 위치

```text
~/vtg-workspace/
  env/                  Python 가상환경
  cache/huggingface/     모델 다운로드 캐시
  data/                 영상 및 다운로드 확인 정보
  results/              사용자가 지정하는 실험 결과
  manifests/            설치 버전, GPU, 코드와 모델 버전 기록
  activate.sh           새 터미널에서 source할 파일
```

스크립트 재실행 시 가상환경과 Hugging Face 캐시를 재사용합니다. 영상은 원본 URL의 해시와 로컬 파일 SHA-256이 확인되면 재사용합니다. 이는 저장된 복사본의 무결성 확인이며 원격 파일의 변경 여부를 새로 확인하는 것은 아닙니다. 다운로드가 끊기면 `.part`와 확인 정보를 남기며, 서버 식별자와 범위 응답을 검증할 수 있을 때 이어받습니다. 검증할 수 없으면 처음부터 다시 받습니다. 대용량 파일의 해시 계산에도 시간이 걸릴 수 있습니다. 같은 이름으로 다른 영상을 받을 때에는 `VIDEO_NAME`을 변경하세요.

```bash
# 이름만 지정하고 URL은 화면에 표시되지 않는 프롬프트에 입력
VIDEO_NAME=experiment-01.mov bash scripts/bootstrap_elice.sh

# 설치와 GPU 확인만 진행
bash scripts/bootstrap_elice.sh --skip-video --skip-model
```

자동 실행 환경에서는 `VIDEO_URL` 환경변수로 주소를 전달할 수 있습니다. URL을 명령줄에 직접 적으면 셸 기록에 남을 수 있으므로 일반 실습에서는 기본 프롬프트를 사용하세요. 스크립트는 공유 링크의 비밀번호나 NAS 계정 인증을 우회하지 않습니다.

## 비용과 데이터 보관

- 새 인스턴스의 빈 디스크에서는 매번 패키지·모델·영상을 다시 다운로드합니다. 스크립트는 수작업을 줄이지만 네트워크 전송 시간과 GPU 인스턴스 가동 중 비용을 없애지는 않습니다.
- 별도 영구 볼륨을 실제로 연결해 둔 경우 `VTG_ROOT=/연결한/볼륨/vtg-workspace`로 캐시·영상을 재사용할 수 있습니다. 이 스크립트는 볼륨 생성·과금·영구 보존 여부를 설정하거나 보장하지 않습니다.
- **인스턴스를 삭제하기 전에 `results/`와 `manifests/`를 외부 저장소로 복사하고, 복사본이 정상인지 확인하세요.** 보관 대상과 인증을 아직 정하지 않았으므로 자동 업로드와 인스턴스 삭제 기능은 포함하지 않습니다.
- 모델·원본 영상·기업 링크를 Git에 넣지 않습니다. 기본 작업 폴더는 저장소 바깥입니다.

## 현재 범위와 검증

자동 설치 대상은 이번에 실행한 **그라운딩 환경**입니다. 영상 설명 생성, MotionGPT, MG-MotionLLM 및 메시 복원은 별도 의존성과 자산을 확인한 뒤 추가해야 합니다. 기본 TimeLens-8B를 실행 확인에 사용하며, 연구용 최종 모델을 선정한 것은 아닙니다.

기존 수동 환경에서는 Python 3.10.14 / A100 MIG 40GB / 드라이버 535.183.06 / PyTorch 2.6.0+cu118로 TimeLens-8B 추론 성공을 확인했습니다. 새 bootstrap의 패키지 설치 및 실제 GPU 추론 전체 흐름은 새 엘리스 인스턴스에서 추가 확인해야 합니다. 로컬 오케스트레이션 테스트의 GPU·pip 호출은 가짜 실행이며 실제 GPU 검증을 대체하지 않습니다.

```bash
python3 -m unittest discover -s scripts -p 'test_*.py'
bash -n scripts/bootstrap_elice.sh
python3 grounding/test_vtg_run.py
```

참고: [TimeLens-8B 공식 모델 카드](https://huggingface.co/TencentARC/TimeLens-8B), [PyTorch 이전 버전 설치](https://pytorch.org/get-started/previous-versions/).

제공된 Synology 단일 파일 공유 방식은 세션 쿠키를 받은 뒤 실제 다운로드 경로에 32바이트 범위 요청을 보내 `206` 응답과 QuickTime 시그니처를 확인했습니다. 원본 영상 전체 다운로드와 엘리스 서버에서의 NAS 접근은 아직 검증하지 않았습니다.
