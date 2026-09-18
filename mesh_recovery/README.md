# 메시 복원 · CoMotion / Multi-HMR 2

[새 인스턴스 전체 실행 순서](../docs/ELICE_GUIDE.md)

| 모델 | 공식 코드 폴더 | 출력 표현 |
| --- | --- | --- |
| CoMotion | `$MOTION_WORKSPACE/ml-comotion` | SMPL 인물 트랙, 모션 캡셔닝 변환 입력 |
| Multi-HMR 2 | `$MOTION_WORKSPACE/multi-hmr2` | Anny 메시/파라미터, 현재 캡셔닝 변환 입력 아님 |

두 모델의 비교 실행은 공통 `pose_env`를 사용합니다. 설치 버전은 `requirements/elice/pose.txt`에 있습니다.

```bash
export MOTION_WORKSPACE="$HOME/aim-workspace"
bash scripts/setup_elice.sh install pose
bash scripts/setup_elice.sh assets pose
# 이용 가능한 SMPL_NEUTRAL.pkl을 workspace/assets/에 준비
source "$MOTION_WORKSPACE/env.sh"
bash scripts/setup_elice.sh check pose
# clip_manifest.json과 inputs/<id>/full.mp4, crop.mp4 준비 후
bash scripts/elice.sh pose
```

`install comotion`, `install multihmr2`로 소스를 따로 설치할 수 있지만, 현재 `pose` 비교 실행은 두 모델을 모두 필요로 합니다. 파일 규약은 전체 가이드의 7번을 참고하세요. 오류가 발생하면 실행기가 비정상 종료하여 후속 캡셔닝으로 넘어가지 않습니다.
