# 2026-09-18 Elice 데모 실행 검증

## 확인한 범위

기존 그라운딩 환경에 별도 CoMotion·MotionGPT 환경을 준비하고, 정식 SMPL neutral을 서버에 직접 내려받은 뒤 **TimeLens → CoMotion → 모션 변환 → MotionGPT → HTML/ZIP**을 순차 실행했다. 이번 결과는 공개 샘플의 연결 확인이며 기업 제공 영상 실험이나 모델 정확도 평가가 아니다.

- 환경: Elice Linux, Python 3.10.14, NVIDIA driver 535.183.06, A100 MIG 3g.40gb.
- 그라운딩: 기존 `~/vtg-env`, PyTorch 2.6.0+cu118. 새 bootstrap으로 처음부터 만든 환경은 이번 실행에 사용하지 않았다.
- 모션: 별도 pose/caption 가상환경, PyTorch 2.5.1+cu121, FFmpeg 4.4.2, OSMesa.
- 실행 ID: `20260918T144017Z-e6c474`.
- 추론 코드: `b54154aed76d7c8c8766ae87f383252143a5bf8a`.
- 서버 직접 SMPL 다운로드 도구의 저장소 커밋: `337ec6c`.

## 실행과 관측 결과

```bash
cd ~/video-motion-pipeline
python3 scripts/download_smpl.py
bash scripts/setup_motion_demo.sh --check
python3 scripts/run_motion_demo.py \
  --video ~/vtg-data/2Y8XQ.mp4 \
  --query "A man drinks water with a glass"
```

SMPL 로그인은 사용자가 서버 터미널에서 입력했다. ZIP과 모델은 저장소 밖 서버 경로에 저장했고, 맥으로 복사하지 않았다. `--check`에서 두 환경의 의존성·CUDA·OSMesa 검사가 통과했으며, 데모는 최종 `state: complete`로 종료했다.

| 단계 | 관측한 결과 | 기록된 시간 | 시간 해석 |
| --- | --- | --- | --- |
| TimeLens-8B | `[[17.0, 20.0]]`, 피크 GPU 메모리 19.3GB | 5.74초 | 그라운딩 추론 지연; 다운로드·전체 설치 시간 아님 |
| CoMotion | Track 1, 19프레임 모두 인물 1명, 비교 영상 디코딩 통과 | 17.12초 | 하위 프로세스 실행·결과 저장; 비교 영상 렌더링 제외 |
| 모션 변환 | 22관절, 20FPS 보간 59프레임, 58×263 특징, 모션 토큰 14개 | 별도 측정 없음 | 보간으로 원본 관측 정보가 추가되지는 않음 |
| MotionGPT | 아래 캡션 생성, 최대 할당 GPU 메모리 약 1.107GB | 0.223초 | 토큰화·생성; 모델 로딩·특징 전처리 제외 |

단계별 측정 범위가 달라 시간 합계를 전체 처리시간으로 해석하지 않는다.

그라운딩 원문:

> The event happens in 17.0 - 20.0 seconds.

MotionGPT의 실제 Track 1 출력:

> a person kneels down while holding onto a stool

요청한 음수 행동과 다른 설명이다. MotionGPT에는 RGB 영상이나 그라운딩 질의가 들어가지 않고 3D 모션만 입력된다. 복원 오차·물체 정보 부재·입력 FPS·학습 데이터 차이 등 가능한 원인은 아직 분리 검증하지 않았다. 자동 캡션은 사람이 수정·승인할 초안으로 취급한다.

캡션 JSON의 `official_reference_012314` 테니스 설명은 별도 설치 확인용 샘플이다. 위 영상 결과와 혼합하지 않는다.

## 시간·좌표 검증의 한계

- 요청 창은 0~30초이고 저장된 그라운딩 입력은 30.08초다.
- 요청한 메시 구간은 17~20초지만 실제 6.117FPS·19프레임 영상은 3.106098초다. 재인코딩과 프레임 단위 경계를 고려해야 한다.
- 보간 모션은 마지막 관측 시점을 기준으로 59프레임이 생성되어 59/20 = 2.95초로 기록된다.
- 특징에서 관절로 복원하는 내부 변환의 최대 절대 오차는 약 `2.98e-8`m다. 실제 사람의 3D 정답에 대한 오차가 아니다.
- 카메라 기준 좌표와 가정한 렌더링 카메라 파라미터를 사용했다. 월드 좌표·절대 이동이나 가려진 하체의 정확도는 검증하지 않았다.
- 긴 연속 트랙을 선택하므로 질의 속 인물과 같은지 별도 검수해야 한다.

## 검수 결과물과 보관

서버 실행 폴더에는 `index.html`, `review.zip`, 비교 영상, 원시 모션, JSON, 단계별 로그를 생성했다. ZIP 무결성을 확인했고, 비교 영상 전체 디코딩과 19프레임 검사를 통과했다. 브라우저에서 입력과 녹색 메시의 나란한 표시를 확인했다.

이후 별도의 로컬 설명 패키지 `motion-demo-20260918-explained`를 작성했다. 단계별 입력·출력, 실패한 캡션의 해석, 측정 범위, 시간 차이 설명을 추가하고 영상 3개·이미지 3장·JSON을 함께 복사했다. 15개 파일, 로컬 링크 20개와 ZIP 무결성을 검사했다. Chrome에서 `file://` HTML 표시를 확인했다.

이 설명 패키지는 이번 실행을 위해 작성한 결과물이며 `run_motion_demo.py`의 기본 HTML 생성 기능에 합쳐진 변경은 아니다. 저장소의 배포 방침에 따라 결과 영상·이미지·로컬 경로·모델·인증정보는 커밋하지 않는다. 설명 패키지는 HTML만 떼어내지 말고 폴더 전체 또는 ZIP으로 보관한다. 원시 모션 배열은 검수 ZIP에 포함되지 않는다.

## 테스트 증거와 남은 검증

- 전체 연결 실행 전 서버에서 기존 스크립트 테스트 37개 통과(FFmpeg 통합 검사 포함).
- SMPL 다운로드 도구에 파일 선택·중복 거부·덮어쓰기 방지 테스트 3개 추가.
- 로컬 스크립트 테스트는 총 40개 중 39개 통과, FFmpeg가 없는 환경의 통합 검사 1개 건너뜀. 실제 GPU 실행 증거는 위 서버 기록과 별개로 구분한다.
- SMPL의 실제 로그인·다운로드·등록은 사용자 인증 후 성공했다. 비밀번호를 자동 테스트나 저장소에 보관하지 않는다.
- 완전히 새 인스턴스에서 bootstrap부터 전체 과정을 한 번에 재현하는 검증, 기업 영상 다운로드 전체 전송·추론, 그라운딩/메시/캡션 품질 평가는 남아 있다.

코드 확인 명령:

```bash
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 grounding/test_vtg_run.py
bash -n scripts/bootstrap_elice.sh scripts/setup_motion_demo.sh scripts/elice.sh
git diff --check
```
