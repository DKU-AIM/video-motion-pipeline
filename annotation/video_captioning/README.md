# 영상 기반 행동 설명과 분류

검수한 선수의 RGB 클립에 농구 맥락을 포함한 질문을 넣어 Qwen 설명을 생성한 뒤, `classify.py`로 행동 라벨을 붙입니다. 한 클립에 복수 라벨을 허용하며 근거가 되는 표현을 보존합니다. `examples/captions.jsonl`에는 3대3 실험의 캡션 6건이 들어 있습니다.

저장소 루트에서 실행합니다.

```bash
python3 annotation/video_captioning/classify.py examples/captions.jsonl outputs/caption-labels
```

JSONL의 각 줄은 `clip_id`와 `raw_output`을 갖습니다. ID는 고유해야 합니다. 출력은 `matches.json`과 `counts.csv`입니다. `--review-id ID`를 반복해 수동 검수 대상을 표시할 수 있습니다. 공개 예시는 저장된 캡션으로 분류를 재현합니다.
