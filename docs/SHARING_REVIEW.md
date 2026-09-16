# 공유 전 정리 내역

## 선택한 방식

한 개의 저장소에서 단계별 폴더를 구분합니다. 모델마다 저장소를 나누면 공통 좌표 변환과 입출력 포맷 변경을 여러 곳에서 관리해야 합니다.
지금은 단계별 폴더 + 모델별 실행 파일이 적절합니다. mesh_recovery와 smpl_eval은 각각 최근 실험 실행과 기존 평가 패키지로 역할이 다릅니다.

## 제외 대상

- data/: 원본·결과·NAS 다운로드 및 서버 감시 스크립트. 필요한 추론·변환 코드만 선별 추출.
- docs의 PPT 제작 디렉터리: 발표 결과물과 빌드 도구.
- Captured-Motion-Dataset, smpl, videos, vis, outputs: 데이터·라이선스 자산·산출물.
- 개인 인수인계 메모, 메일 링크, 구형 배포 번들, 기존 가상환경.
- 영상별 run_motiongpt_* / run_mgllm_* 복제본: 공통 실행본만 유지.

기존 파일의 일괄 영구 삭제는 하지 않았습니다. 실험 재현 자료와 발표 파일은 보존합니다.
기존 Git 이력을 복사하지 않아 과거 데이터나 비밀 정보가 새 저장소 이력에 따라가지 않도록 했습니다.

## 배포 전 남은 작업

- Organization 및 저장소 이름 확인, 비공개 저장소 생성과 업로드.
- 새 서버에서 모델별 의존성 및 체크포인트 준비 과정을 재검증하고 버전 고정.
- captioning의 자동 최장 트랙 선택 대신 명시적인 track_id manifest 지원은 후속 개선 대상.
- 영상별 반복 실험 스크립트에만 있는 수동 선택 정보는 원래 실험 폴더에 보존. 본 공유본은 모든 과거 실험의 재현 패키지가 아님.
- 재사용성이 확인된 후 smpl_eval의 패키지명 변경 여부 판단. 지금은 import와 테스트 호환 유지.

## 입출력 계약

mesh_recovery/run_pose_batch.py는 MOTION_WORKSPACE/clip_manifest.json의 각 id에 대해 inputs/<id>/full.mp4와 crop.mp4를 읽습니다.
모델 저장소는 workspace/ml-comotion 및 multi-hmr2에, 체크포인트는 각 공식 경로에 준비해야 합니다.
results/<id>/<full|crop>/<comotion|hmr2>에 native 파라미터와 metadata 및 comparison.mp4를 씁니다.

annotation/motion_captioning/export_motion_inputs.py는 workspace/results/*/*/comotion/motion_tracks.pt와 metadata.json을 읽습니다.
연속된 인물 트랙을 선정하여 motion_inputs/*.npy와 manifest.json을 만듭니다. 최소 2초, 최대 6초입니다.
MotionGPT는 workspace/MotionGPT와 assets, MG-MotionLLM은 workspace/MG-MotionLLM과 assets의 기존 체크포인트 구성을 요구합니다.
MG-MotionLLM은 MotionGPT 단계에서 생성한 features_263.npy를 사용하므로 현재 구현의 실행 순서를 지켜야 합니다.
