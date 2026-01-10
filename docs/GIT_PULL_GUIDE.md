# Git Pull 가이드

엘리스 클라우드 인스턴스에서 최신 코드를 받는 방법입니다.

## 현재 상황

```bash
git status
# Your branch is behind 'origin/Jaewon' by 1 commit
```

이는 원격 저장소에 새로운 커밋(Load Balancing 구현)이 있지만, 로컬에는 아직 받지 않은 상태입니다.

## 해결 방법

### 1. 간단한 방법 (Fast-forward)

```bash
git pull origin Jaewon
```

또는

```bash
git pull
```

### 2. 안전한 방법 (변경사항 확인 후)

```bash
# 현재 변경사항 확인
git status

# 변경사항이 없으면 pull
git pull origin Jaewon

# 변경사항이 있으면 stash 후 pull
git stash
git pull origin Jaewon
git stash pop  # stash한 변경사항 다시 적용
```

### 3. 완전히 동기화

```bash
# 원격 브랜치 정보 업데이트
git fetch origin

# 현재 브랜치 확인
git branch

# Jaewon 브랜치가 아닌 경우 체크아웃
git checkout Jaewon

# 최신 코드 pull
git pull origin Jaewon
```

## Pull 후 확인

```bash
# 최신 커밋 확인
git log --oneline -5

# 상태 확인 (깨끗해야 함)
git status

# 새로 추가된 파일 확인
ls -la src/load_balancing.py
ls -la src/throughput_measurement.py
ls -la src/dht_utils.py
```

## 예상 결과

Pull 후에는 다음 파일들이 추가되어야 합니다:
- `src/load_balancing.py`
- `src/throughput_measurement.py`
- `src/dht_utils.py`
- `docs/LOAD_BALANCING_USAGE.md`
- `docs/ELICE_CLOUD_LOAD_BALANCING_TEST.md`
- `scripts/elice_test_load_balancing.sh`

그리고 다음 파일들이 수정되어야 합니다:
- `src/main.py`
- `requirements.txt`


