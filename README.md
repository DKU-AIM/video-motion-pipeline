# AIM · 농구 영상에서 선수별 3D 모션 데이터까지

단국대학교 캡스톤디자인 2 · 박지우 · 박주희 · 안균승

농구 경기 영상에서 같은 선수를 추적하고, **선수 영상·3D 모션·행동 설명·카테고리**를 원본 시각과 추적 ID로 연결합니다. 복원된 인체 동작을 MuJoCo에서 확인하고, GMR로 Unitree G1 관절에 옮겨 재생과 물리 추종을 실험했습니다.

[파이프라인과 영상](#파이프라인) · [데이터 활용 목표](#데이터-활용-목표) · [실행 방법](#실행-방법)

## 제출 자료

| 제출 항목 | 자료 |
| --- | --- |
| 소스 코드 · 실행 스크립트 · 개발 문서 | [실행 방법](#실행-방법) · [농구 추적·복원 실행](mesh_recovery/basketball/README.md) · [모델 환경 설정](docs/ELICE_GUIDE.md) |
| 발표자료 또는 포스터 PDF | 최종본 준비 중 |
| 프로젝트 결과보고서 PDF | [AIM 프로젝트 결과보고서](docs/AIM-result-report.pdf) |
| 외부에서 접속 가능한 최종 시연 영상 | 녹화 및 공개 링크 준비 중 |

## 파이프라인

![실제 농구 프레임과 모델 그림을 넣은 AIM 파이프라인](docs/images/basketball-pipeline.png)

**대표 예시** · 남색 21번 선수(PHALP ID 9)의 드리블 6초, 60 FPS·360프레임. [FIBA 원본 경기 15:35–15:41](https://www.youtube.com/watch?v=kXuDZJ2TT3I&t=935s)

같은 선수의 크롭에서 **3D 모션 복원**과 **행동 설명·분류**를 각각 진행하고, 두 결과를 추적 ID와 원본 시각으로 연결합니다. 각 단계를 펼치면 설명과 실제 영상을 볼 수 있습니다.

<details>
<summary><b>1. 원본 영상</b> · 전체 코트가 담긴 경기 클립</summary>

아래 영상은 모델을 적용하기 전의 입력입니다. 이후 단계에서는 같은 구간에서 남색 21번 선수를 따라갑니다.

https://github.com/user-attachments/assets/9bf0d29a-7fa7-4894-b55c-cea6bdbbc55a

</details>

<details>
<summary><b>2. 선수 검출·추적</b> · 4D-Humans로 선수의 ID와 박스 연결</summary>

- **ViTDet-H**가 각 프레임에서 사람의 박스·마스크·검출 점수를 얻습니다.
- **HMR 2.0**이 검출된 사람의 자세·체형·카메라를 추정해 추적에 필요한 인체 정보를 제공합니다.
- **PHALP**가 이전 트랙에서 예측한 자세·위치와 외형 정보를 현재 검출과 비교해 ID를 연결합니다.

영상에서 **ID 9** 박스를 따라가면 코트를 이동하는 같은 선수를 확인할 수 있습니다.

https://github.com/user-attachments/assets/1a101d66-de72-47bc-9d09-3f1aa0de1c1d

[추적 실행 코드](mesh_recovery/basketball/run_tracking.py) · [4D-Humans 논문](https://openaccess.thecvf.com/content/ICCV2023/html/Goel_Humans_in_4D_Reconstructing_and_Tracking_Humans_with_Transformers_ICCV_2023_paper.html) · [PHALP 코드](https://github.com/brjathu/PHALP)

</details>

<details>
<summary><b>3. 선수 크롭</b> · ID 9의 박스에 사방 10% 여백을 더해 자르기</summary>

선택한 ID의 관측 박스를 따라 원본 RGB를 프레임마다 자릅니다. 좌우에는 각각 박스 너비의 10%, 위아래에는 각각 높이의 10%를 더해 팔다리가 잘리는 것을 줄입니다. 원본 프레임 번호와 크롭 좌표도 기록합니다.

https://github.com/user-attachments/assets/1849b2d7-4701-44ea-a950-70e9b4043e94

[크롭 좌표 계산 코드](mesh_recovery/basketball/crop_geometry.py)

</details>

<details>
<summary><b>4. 3D 모션 복원</b> · HMR 2.0과 SMPL로 메시·관절 생성</summary>

선수 크롭에 HMR 2.0을 다시 실행해 자세·체형·카메라를 추정하고, SMPL 메시와 관절을 시간순으로 저장합니다. 영상은 **왼쪽이 선수 크롭, 오른쪽이 같은 시각의 메시 합성**입니다.

https://github.com/user-attachments/assets/2e10f2d4-3c9b-4aae-af1d-bf6bbcd7f37e

[복원 실행 코드](mesh_recovery/basketball/run_mesh.py)

</details>

<details>
<summary><b>5. 행동 설명·분류</b> · Qwen 설명에서 드리블·이동 라벨 추출</summary>

Qwen에 선수 RGB 크롭과 농구 맥락을 담은 질문을 줍니다. 실제 응답인 “농구공을 드리블하며 코트를 가로지르고 있습니다”에서 규칙으로 **‘드리블하며’ → 드리블**, **‘가로지르고’ → 이동**을 매칭합니다. 한 클립에 여러 라벨과 근거 문구를 함께 기록할 수 있습니다.

https://github.com/user-attachments/assets/ceebea53-25dc-49b6-8d82-6648387d5271

[분류 규칙과 실행 방법](annotation/video_captioning/README.md) · [분류 코드](annotation/video_captioning/classify.py)

</details>

**선수별 데이터 한 세트** · 3D 모션 + 대응 선수 영상 + 행동 설명 문장 + 행동 카테고리

<details>
<summary>초기 설계와 실험 순서 보기</summary>

![초기 농구 데이터 구축 설계](docs/images/basketball-design.png)

- **9월** · 자연어 쿼리로 시간 구간을 먼저 찾는 설계에서 출발했습니다. [초기 설계 기록](docs/early-research-plan.md)
- **10월 초** · 농구 FPS·질의 변경과 축구 특화 모델을 시험했습니다. 짧은 농구 동작의 구간을 안정적으로 잡기 어려워, 선택한 짧은 영상에서 선수 추적과 크롭을 먼저 수행하는 방식으로 바꿨습니다.
- **10월 중순** · CoMotion과 Multi-HMR 2의 가림 후 ID 끊김과 자세 흔들림 사례를 검토했습니다. 선택한 경기에는 4D-Humans를 적용했고, 선수 크롭에서 HMR 2.0을 다시 실행했습니다.
- **11월** · 선수별 Qwen 설명과 규칙 분류를 연결하고, SMPL 및 G1의 MuJoCo 재생과 물리 추종을 시험했습니다.

</details>

<details>
<summary><b>실험에서 확인한 범위</b> · 농구 표본, 분류 집계, G1 물리 추종</summary>

- **농구 영상** · 검수한 5명의 6구간에서 선수 크롭과 SMPL을 연결했습니다. 대표 드리블 구간은 60 FPS, 360프레임입니다. 추적 평가는 선택 구간의 화면과 로그를 확인하는 방식이었습니다.
- **행동 분류** · 별도 3대3 경기의 선수 영상 6개에서 경합 5, 공 소지 4, 이동 2, 드리블 1을 기록했습니다. 한 영상에 여러 라벨을 허용한 출력 집계입니다.
- **MuJoCo** · 저장한 SMPL 정점을 고정 루트로 재생했습니다. GMR로 변환한 G1 29관절은 기구학적 목표 재생으로 확인했습니다.
- **G1 물리 제어** · 대표 전신 드리블의 평가 10회에서 결합 기준 통과는 0회였습니다. 하체와 허리에 기본 서기 목표를 준 팔 중심 3초 과제는 동일 참조 모션의 초기 교란 10회에서 통과했고 팔 RMSE는 평균 0.162 rad였습니다. 실험 범위는 시뮬레이션의 해당 참조 모션입니다.

</details>

## 실행 방법

실험별 코드가 들어 있습니다. 공개 샘플로 캡션에서 라벨을 만드는 단계를 모델 설치 없이 확인할 수 있습니다.

```bash
git clone https://github.com/DKU-AIM/video-motion-pipeline.git
cd video-motion-pipeline
python3 annotation/video_captioning/classify.py examples/captions.jsonl outputs/caption-labels
```

결과는 `matches.json`과 `counts.csv`에 저장됩니다.

| 경로 | 용도 |
| --- | --- |
| [grounding/](grounding/) | 초기 시간 구간 탐색 모델 실험 |
| [mesh_recovery/](mesh_recovery/) · [smpl_eval/](smpl_eval/) | 복원 비교와 검수 코드 |
| [annotation/](annotation/) | 영상 설명 분류와 모션 캡셔닝 비교 |
| [docs/](docs/) | 결과보고서, 구현 이미지, 영상과 개발 문서 |

GPU 추론에는 단계별 체크포인트와 SMPL 자산이 필요합니다. [설치 및 검증 가이드](docs/ELICE_GUIDE.md)

## 데이터 활용 목표

**이미 촬영된 경기 영상에서 휴머노이드가 모방할 전신 동작의 참조 데이터를 구축하는 것이 목표입니다.** 새 동작을 추가할 때마다 센서 수트를 입고 다시 촬영해야 하는 부담을 줄이고, 급정지·방향 전환처럼 균형과 전신 협응이 필요한 동작을 수집하려는 접근입니다.

행동 라벨로 원하는 동작을 골라내고, 대응 영상으로 확인한 뒤 3D 모션을 로봇의 관절 목표로 변환합니다. 이번 프로젝트에서는 **SMPL 인체 동작 재생과 GMR을 통한 Unitree G1 관절 목표 재생**까지 MuJoCo에서 확인했습니다. 모방 학습에 활용하려면 해당 목표를 로봇이 균형을 유지하며 따라갈 수 있는지 추가 검증이 필요합니다.

<details>
<summary><b>활용 시연 1</b> · 같은 선수의 영상·SMPL 메시·MuJoCo 인체 재생 비교</summary>

같은 6초 구간을 나란히 비교합니다. MuJoCo에서는 저장한 SMPL 정점을 고정 루트로 재생했습니다.

https://github.com/user-attachments/assets/696af566-1b09-47fb-a163-98d9ff9a65a3

</details>

<details>
<summary><b>활용 시연 2</b> · Unitree G1 관절 목표 재생</summary>

GMR로 변환한 G1의 29개 관절 목표를 MuJoCo에서 기구학적으로 재생합니다. [전신 물리 추종 실험 영상](https://github.com/DKU-AIM/video-motion-pipeline/blob/main/docs/media/g1-fullbody.mp4)

https://github.com/user-attachments/assets/659e9350-3e5b-4a06-800b-066f3644c5b1

</details>
