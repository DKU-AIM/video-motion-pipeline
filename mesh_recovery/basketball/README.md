# 농구 선수 추적과 복원

캡스톤 농구 실험에서 사용한 추적·크롭·HMR 2.0 실행 코드를 공개용 경로 인자로 옮겼습니다. 추적은 공식 4D-Humans의 PHALP 연동을 호출하고, 현재 프레임에서 관측된 박스만 내보냅니다. 복원은 사용자가 지정한 ID와 프레임 범위에 10% 여백을 적용한 크롭에서 HMR 2.0을 다시 실행합니다.

## 준비와 실행

- CUDA 환경, PyTorch, OpenCV, FFmpeg와 FFprobe, PyRender용 오프스크린 렌더링 환경이 필요합니다. 영상 파일은 오디오 스트림을 포함해야 전체 내보내기의 검사를 통과합니다.
- [4D-Humans](https://github.com/shubham-goel/4D-Humans)와 [PHALP](https://github.com/brjathu/PHALP)의 설치 안내에 따라 체크포인트와 SMPL을 준비합니다. 실험에서 사용한 커밋은 각각 `efe18deff163b29dff87ddbd575fa29b716a356c`, `96f7e6c09fb858ec3f597d59246c151ab4394bc3`입니다.
- `AIM_4DHUMANS_DIR`에 4D-Humans 체크아웃 경로를 지정합니다. 추적 결과와 원본 영상을 아래처럼 같은 케이스 폴더에 둡니다.

```bash
python3 mesh_recovery/basketball/run_tracking.py \
  --id 02 --video inputs/02/입력_원본클립.mp4 --output-dir inputs/02

python3 mesh_recovery/basketball/run_mesh.py \
  --input-dir inputs --output-dir outputs/basketball --segment 2
```

`--segment 2`는 예시 계획의 세 번째 항목으로, 케이스 02의 ID 9 드리블 구간입니다. 다른 영상에는 직접 검수한 ID와 시작 포함·끝 제외 프레임을 `--plan`에 지정합니다. 복원 결과에는 영상, 프레임별 좌표 매핑과 SMPL 배열이 포함됩니다. 공개용 경로 인자를 적용한 버전은 정적 검사와 크롭 기하 테스트를 수행했으며, GPU 실험 결과는 원래 실행 기록을 사용했습니다.
