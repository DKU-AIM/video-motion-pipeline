# 후보 1: 모션 캡셔닝

CoMotion SMPL → 연속된 인물별 22관절 시퀀스 → HumanML3D 263차원 표현 → MotionGPT / MG-MotionLLM 설명.

기존 실험 코드입니다. 최종 설명 생성 방식으로 선정된 것은 아닙니다. 실행 순서와 환경은 루트 README를 참고하세요.


## 새 인스턴스 설치와 실행

[전체 실행 가이드](../../docs/ELICE_GUIDE.md)를 먼저 확인하세요. 레포 루트에서 실행합니다.

```bash
export MOTION_WORKSPACE="$HOME/aim-workspace"
bash scripts/setup_elice.sh install motiongpt
# 기존 기준 모션 reference_features.npy를 assets/에 복원
bash scripts/setup_elice.sh assets motiongpt
source "$MOTION_WORKSPACE/env.sh"
bash scripts/setup_elice.sh check motiongpt
# CoMotion 추론 결과가 준비된 뒤
bash scripts/elice.sh export-motion
bash scripts/elice.sh motiongpt
```

비교 후보 MG-MotionLLM을 사용할 때만 추가합니다.

```bash
bash scripts/setup_elice.sh install mgllm
bash scripts/setup_elice.sh assets mgllm
bash scripts/setup_elice.sh check mgllm
bash scripts/elice.sh mgllm
```

두 캡셔닝 모델은 `env`를 공유하며 의존성은 `requirements/elice/caption.txt`에 있습니다. 현재 MG-MotionLLM은 MotionGPT 단계에서 저장한 특징을 읽습니다. RGB 영상이나 카테고리 정답을 모션 캡셔닝 입력으로 주는 구조가 아닙니다.
