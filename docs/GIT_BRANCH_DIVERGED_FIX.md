# Git Branch Diverged 해결 가이드

## 상황

```
Your branch and 'origin/Jaewon' have diverged,
and have 6 and 32 different commits each, respectively.
```

로컬 브랜치와 원격 브랜치가 서로 다른 히스토리를 가지고 있는 상태입니다.

## 해결 방법

### 방법 1: 원격 버전으로 완전히 맞추기 (권장 - Load Balancing 테스트용)

로컬 변경사항을 버리고 원격의 최신 버전을 사용:

```bash
# 원격 정보 업데이트
git fetch origin

# 원격 버전으로 완전히 맞추기 (로컬 커밋 삭제)
git reset --hard origin/Jaewon

# 상태 확인
git status
```

**주의**: 이 방법은 로컬의 6개 커밋을 삭제합니다. 중요한 변경사항이 있다면 먼저 백업하세요.

### 방법 2: Merge 사용 (로컬 커밋 유지)

로컬 커밋을 유지하면서 원격 변경사항을 병합:

```bash
git pull origin Jaewon
```

Merge commit이 생성되며, 두 히스토리가 합쳐집니다.

### 방법 3: Rebase 사용 (선형 히스토리)

로컬 커밋을 원격 커밋 위에 올림:

```bash
git pull --rebase origin Jaewon
```

로컬 커밋이 원격 커밋 위에 재배치됩니다 (선형 히스토리).

## Load Balancing 테스트 권장 방법

Load Balancing을 테스트하려면 **방법 1 (reset --hard)**을 권장합니다:

1. 원격의 최신 Load Balancing 코드를 사용
2. 로컬의 오래된 커밋과 충돌 방지
3. 깨끗한 상태에서 테스트

## 단계별 실행

```bash
# 1. 현재 상태 확인
git log --oneline -10
git log origin/Jaewon --oneline -10

# 2. 원격 정보 가져오기
git fetch origin

# 3. 원격 버전으로 맞추기
git reset --hard origin/Jaewon

# 4. 확인
git status
git log --oneline -5

# 5. 새 파일 확인
ls -la src/load_balancing.py
ls -la src/throughput_measurement.py
ls -la src/dht_utils.py
```

## 문제 발생 시

만약 중요한 로컬 변경사항이 있다면:

```bash
# 로컬 커밋들을 백업 브랜치로 저장
git branch backup-local-commits

# 그 다음 원격으로 맞추기
git reset --hard origin/Jaewon
```

나중에 필요하면:
```bash
git checkout backup-local-commits
# 필요한 파일만 cherry-pick 또는 수동으로 복사
```


