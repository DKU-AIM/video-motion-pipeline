# 그라운딩 · 영상과 쿼리 → 시간 구간

[새 인스턴스 전체 실행 순서](../docs/ELICE_GUIDE.md)

레포 루트에서 시스템 패키지 설치 후 실행합니다.

```bash
export MOTION_WORKSPACE="$HOME/aim-workspace"
bash scripts/setup_elice.sh install grounding
source "$MOTION_WORKSPACE/env.sh"
bash scripts/setup_elice.sh check grounding
bash scripts/elice.sh grounding --list
bash scripts/elice.sh grounding --model timelens2-4b \
  --video /path/to/video.mp4 --query "A person shoots a basketball." \
  --out "$MOTION_WORKSPACE/grounding.jsonl"
```

선택한 모델의 가중치는 첫 추론 때 Hugging Face에서 받습니다. 설치 환경: `requirements/elice/grounding.txt`, 실행 Python: `grounding_env`.

실제 새 서버의 대표 모델 검증은 TimeLens2-4B입니다. 지원 목록의 모든 모델/긴 영상에서 메모리와 정확도를 검증한 것은 아닙니다.
