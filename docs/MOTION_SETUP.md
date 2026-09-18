# CoMotion·MotionGPT 데모 환경

이 안내는 Elice Linux GPU 인스턴스에서 **CoMotion → MotionGPT**만 독립 가상환경으로 준비합니다. 저장소 안에는 모델 가중치, 회사 영상, SMPL 파일을 저장하지 않습니다. 인스턴스와 별도로 유지되는 저장소가 있다면 `MOTION_ASSETS_ROOT`를 그 경로로 지정해 소스·가중치를 재사용할 수 있습니다. 가상환경은 같은 OS·Python·절대 경로가 유지되는 경우에만 재사용하고 `--check`로 확인하세요.

```bash
cd ~/video-motion-pipeline
bash scripts/setup_motion_demo.sh --assets-root ~/motion-workspace --prepare-only --install-system
source ~/motion-workspace/activate-motion.sh
```

스크립트는 다음 공개 자산을 고정 revision에서 준비합니다.

- [Apple CoMotion](https://github.com/apple/ml-comotion) `04b035af79a68267f6bb722b61bf5f13cb2b0068` 및 공식 체크포인트 아카이브
- [OpenMotionLab MotionGPT](https://github.com/OpenMotionLab/MotionGPT) `001aaca8d0ee218fc17f8265d11ac124044fe42f`
- [MotionGPT-base](https://huggingface.co/OpenMotionLab/MotionGPT-base) `a0a37a388137f15df8299c643a885a42b07772fe`의 `motiongpt_s3_h3d.tar`
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D) `9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23`의 `012314` 공개 sanity sample. `reference_joints.npy`는 `(170, 22, 3)`, `reference_features.npy`는 `(170, 263)`이며 SHA-256을 확인합니다.

`run_motiongpt.py`는 위 public sample을 **설치와 모델 로딩 확인용**으로 먼저 캡셔닝합니다. 이것은 기업 영상 실험 결과가 아닙니다.

## 서버로 직접 SMPL 다운로드

SMPL 사이트에서 본인 계정 가입·이메일 인증과 약관 확인을 완료했다면, **엘리스 서버 터미널**에서 다음을 실행할 수 있습니다.

```bash
cd ~/video-motion-pipeline
python3 scripts/download_smpl.py
```

서버 터미널에 계정 이메일과 비밀번호를 입력합니다. 비밀번호는 화면에 표시하거나 파일·명령 기록에 저장하지 않으며, 공식 HTTPS 다운로드 서비스에만 전송합니다. 브라우저 로그인 상태는 서버와 공유되지 않으므로 서버에서 한 번 인증해야 합니다. 인증정보를 채팅에 보내지 마세요.

ZIP은 서버의 `~/motion-workspace/assets/private/`에 저장되고, 필요한 neutral 모델만 CoMotion 경로에 등록합니다. 맥에서는 실행을 거부하므로 모델을 맥으로 내려받지 않습니다. 다른 `--assets-root`도 지정할 수 있습니다. 기존 모델은 덮어쓰지 않습니다. 인증이 실패하면 모델을 설치하지 않고 오류를 표시합니다.

2026-09-18 사용자 인증으로 공식 ZIP의 서버 직접 다운로드와 neutral 모델 등록을 확인했습니다. 이후 환경 검사와 전체 GPU 데모 실행도 통과했습니다. [실행 결과와 한계](VALIDATION_2026-09-18.md)를 참고하세요.

## 이미 확보한 SMPL 파일 등록

CoMotion은 neutral SMPL v1.1.0 파일이 필수입니다. 파일은 배포·GitHub 업로드 대상이 아니므로, 공식 SMPL 절차를 따라 권한 있는 사용자가 내려받아야 합니다. 준비한 뒤에만 아래처럼 private assets root로 복사합니다.

```bash
bash scripts/setup_motion_demo.sh \
  --assets-root ~/motion-workspace \
  --smpl /safe/path/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

복사 대상은 `ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl`입니다. [CoMotion 공식 README](https://github.com/apple/ml-comotion#download-smpl-body-model)의 요구 경로와 같습니다.

## 확인과 서버 요구사항

```bash
bash scripts/setup_motion_demo.sh --assets-root ~/motion-workspace --check
```

`--check`는 두 가상환경의 CUDA 인식, 공개 자산 해시, CoMotion 두 체크포인트, SMPL 파일, `ffmpeg`/`ffprobe`/`nvidia-smi`를 확인합니다. Elice 이미지에 `ffmpeg`와 OSMesa가 없다면 준비할 때 `--install-system`을 붙입니다. 이 옵션은 `sudo -n apt-get update`와 `sudo -n apt-get install -y ffmpeg libosmesa6 libgl1-mesa-dri`만 실행하며, 비밀번호 없는 sudo가 없으면 즉시 중단합니다.

`pose-env`는 CoMotion 공식 `torch==2.5.1` 요구사항에 맞춘 CUDA 12.1 wheel을 사용합니다. CoMotion의 `scenedetect`와 OpenCV가 무제한 의존성으로 최신 버전을 받아 API·NumPy 호환이 깨지는 것을 막기 위해 `scenedetect<0.7`, `opencv-python<4.12`, `numpy==1.26.4`도 고정합니다. 오프스크린 렌더러는 OSMesa를 위해 `PyOpenGL==3.1.7`을 사용하고, 이 버전을 허용하도록 수정한 [공식 pyrender 커밋](https://github.com/mmatl/pyrender/commit/7c613e8aed7142df9ff40767a8f10b7a19b6255c)을 설치합니다. `caption-env`도 분리해 MotionGPT 추론에 필요한 `torch`, `numpy`, `scipy`, `transformers`, `sentencepiece`, `tqdm`만 설치합니다. MotionGPT의 학습·WebUI·Blender·전체 데이터셋 의존성은 이 데모에 포함하지 않습니다.

환경을 준비했다고 해서 전체 파이프라인이 실행되는 것은 아닙니다. 다음 단계의 데모 러너가 `MOTION_ASSETS_ROOT`에서 `ml-comotion`, `MotionGPT`, `assets`를 읽고, 별도 `MOTION_WORKSPACE`에 입력 클립과 결과를 만듭니다.
