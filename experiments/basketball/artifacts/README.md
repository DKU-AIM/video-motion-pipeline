# 농구 비디오 그라운딩 실험 보관본

이 디렉터리는 `/home/elicer/aim-workspace/results_grounding/`의 분석용 텍스트 기록을 복사한 스냅샷이다. 원본 실행 디렉터리를 이동하거나 수정하지 않았다. 이후 실행 결과는 자동 반영되지 않는다.

- 보관 시점(UTC): `2026-09-26T04:37:20.344660+00:00`
- 복사 파일 수: 464개, 총 4063716 bytes
- 파일별 원본 경로·수정 시각·크기·SHA-256: [manifest.json](manifest.json)
- 보관 대상: 원문 응답, 파싱 결과, 쿼리/설정, 실제 프롬프트 템플릿, 시간·자원 기록, 환경 정보, 실패·수정 이력.
- 영상, 가중치, 가상환경, 인증 설정은 포함하지 않았다.

## 실험별 시작점

| 디렉터리 | 보관 당시 상태 | 먼저 볼 파일 |
|---|---|---|
| `basketball_quick/` | 최종 8개 작업 성공. 과거 실패·재시도도 포함 | [요약](basketball_quick/final_summary.md), [비교](basketball_quick/comparison.md), [상태](basketball_quick/state.json) |
| `basketball_v2/` | 16개 작업 성공, 48회 호출 기록 | [프롬프트 비교](basketball_v2/prompt_comparison.md), [비교](basketball_v2/comparison.md), [진행 기록](basketball_v2/progress.md) |
| `basketball_parameters/` | 8개 작업 성공, 40회 호출 기록 | [파라미터 요약](basketball_parameters/parameter_summary.md), [비교](basketball_parameters/comparison.md), [진행 기록](basketball_parameters/progress.md) |
| `basketball_all/` | 초기 전체 실험 중단: 1개 작업 interrupted | [진행 기록](basketball_all/progress.md) |
| `basketball_all_before_pathfix_20260926T013935/` | 경로 수정 전: 2개 failed, 1개 interrupted | [진행 기록](basketball_all_before_pathfix_20260926T013935/progress.md) |
| `basketball_timing_check/` | 시간 측정 진단 실행 | [설정](basketball_timing_check/settings.json), [응답](basketball_timing_check/results.jsonl) |

`success`는 실행 완료 상태다. 응답의 정확성이나 출력 완전성을 보장하지 않는다. 정답 구간 검수가 없어 구간 개수를 정확도처럼 해석하면 안 된다. 초기 중단·실패 실행은 정상 비교군과 섞지 않는다.

## 파일 활용법

각 모델·조건의 `attempt_*`를 보존했다. 재시도 중 어떤 실행이 최종 채택되었는지는 해당 실험의 `state.json`을 확인한다. 전체 attempt를 무조건 합산하면 중복 집계될 수 있다.

| 파일 | 용도 |
|---|---|
| `results.jsonl` | 모델 원문 `raw`, 추출 구간, 오류, 처리·생성 시간, 토큰 수, 메모리 피크 |
| `results.normalized.jsonl` (존재할 때) | 원문을 보존한 채 파서 수정으로 다시 해석한 결과; 원본과 비교 |
| `settings.json`, `prompt-template.txt` | 실제 입력 조건·쿼리 스냅샷·프롬프트 확인 |
| `resources.jsonl` | 시간에 따른 프로세스 CPU/RAM과 GPU 관측; `gpu_scope`에 유의하며 물리 GPU 수치를 MIG/단일 프로세스 전용 사용량으로 단정하지 말 것 |
| `generation_progress.jsonl`, `load.json` | 생성 진행과 모델 로딩 시간 |
| `pip-freeze.txt`, `gpu.txt`, `code-commit.txt` | 실행 당시 패키지·GPU·코드 정보 |
| `plan.json`, `state.json`, `*audit.json` | 계획, 실행 상태, 코드/파서 수정 이력 |
| `all_trials.csv`, `all_responses.jsonl`, `summary.csv` | 편리한 집계본; 원문/attempt와 중복이므로 별도 실험처럼 합산하지 말 것 |
| `*.log` | 실행 로그와 실패 원인; 이 보관 디렉터리에서만 Git 추적 허용 |

## 경로와 재현 범위

문서 링크는 이 보관 디렉터리 기준 상대경로다. JSON과 과거 로그 안의 `/home/elicer/...`는 **당시 실행 이력**이므로 바꾸지 않았다. 기록에 남은 절대경로가 내려받은 컴퓨터에 존재할 필요는 없다. 데이터는 각 파일에서 직접 읽을 수 있다.

기존 분석 스크립트의 경로 처리는 이번 보관 작업에서 수정하지 않았다. 다른 컴퓨터에서 분석 프로그램을 실행하려면 해당 입력 경로를 이 보관본으로 지정하거나 수정해야 한다. 모델 재실행 및 장면 확인에는 영상이 별도로 필요하며, `settings.json`의 영상 해시로 입력 동일성을 확인할 수 있다.

`code-commit.txt`만으로 실행 당시 미커밋 변경까지 재현할 수는 없다. `plan.json`의 코드 해시, 수정 이력 및 [v2 검증 소스 스냅샷](../../../docs/analysis/basketball_v2/verified_source_snapshot/README.md)을 함께 확인한다. 보관 시점 Git 커밋은 manifest에 별도로 기록했으며 실행 당시 커밋과 다를 수 있다.

## 환경 보충 자료

[environment/](environment/)의 `grounding_env-freeze.txt`, `grounding_env-torch.txt`는 워크스페이스 환경 기록이다. 실행별 `attempt_*/pip-freeze.txt`가 있으면 그것을 우선 참고한다. `setup-revisions.json`은 포즈·모션 등 외부 저장소 설치 리비전이며, 그라운딩 모델 가중치 리비전 목록은 아니다.

## 보관 무결성 확인

이 디렉터리에서 다음 명령을 실행한다. Python 표준 라이브러리만 필요하다.

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('manifest.json').read_text())
for item in manifest['files']:
    data = Path(item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
print(f"Verified {len(manifest['files'])} archived files")
PY
```

manifest는 복사한 원본 자료만 대상으로 하며 이 README, `.gitignore`, manifest 자체는 포함하지 않는다.
