# v2 기록 해시와 일치하는 소스만 보관

`basketball_probe.py`, `vtg_run.py`는 v2 plan.json의 SHA256과 현재 파일이 일치함을 확인해 복사했다.

`basketball_batch.py`는 후속 실험 추가로 해시가 달라져 v2 원본이라고 복사하지 않았다. 원본 배치의 복원이 완료됐다고 주장하지 않는다. plan.json에는 당시 실행 목록·설정·해시가 남아 있다. git commit만으로 미커밋 소스를 재현할 수 없으므로 다음 실행부터 전체 소스·설정 사본과 의존성 lock을 함께 보관한다.
