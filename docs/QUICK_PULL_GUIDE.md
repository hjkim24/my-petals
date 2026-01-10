# Elice Cloud 인스턴스에서 Git Pull 빠른 가이드

## 🚀 빠른 실행 (추천)

Elice Cloud 인스턴스에 SSH 접속 후 다음 명령어들을 순서대로 실행하세요:

### Step 1: 프로젝트 디렉토리로 이동

```bash
cd ~/my-petals
```

### Step 2: 현재 상태 확인

```bash
git status
```

**예상 결과**:
```
On branch Jaewon
Your branch is behind 'origin/Jaewon' by 1 commit, and can be fast-forwarded.
```

### Step 3: 최신 코드 받기

```bash
git pull origin Jaewon
```

**또는 간단하게**:
```bash
git pull
```

### Step 4: 확인

```bash
# 상태 확인 (깨끗해야 함)
git status

# 최신 커밋 확인
git log --oneline -3

# 새로 추가된 파일 확인
ls -la src/load_balancing.py
ls -la src/throughput_measurement.py
ls -la src/dht_utils.py
```

## ⚠️ 문제가 생긴 경우

### 경우 1: 로컬 변경사항이 있는 경우

```bash
# 현재 변경사항 확인
git status

# 변경사항 임시 저장
git stash

# 최신 코드 받기
git pull origin Jaewon

# 저장한 변경사항 다시 적용
git stash pop
```

### 경우 2: 브랜치가 다른 경우

```bash
# 현재 브랜치 확인
git branch

# Jaewon 브랜치로 전환
git checkout Jaewon

# 최신 코드 받기
git pull origin Jaewon
```

### 경우 3: 충돌이 발생한 경우

```bash
# 충돌 파일 확인
git status

# 수동으로 충돌 해결 후
git add <해결한_파일>
git commit -m "Resolve merge conflicts"
```

### 경우 4: 완전히 초기화하고 싶은 경우 (⚠️ 주의: 로컬 변경사항 삭제)

```bash
# 원격 상태로 완전히 리셋 (로컬 변경사항 모두 삭제)
git fetch origin
git reset --hard origin/Jaewon
```

## 📝 한 줄로 실행하기

```bash
cd ~/my-petals && git pull origin Jaewon
```

## ✅ Pull 성공 확인

다음 명령어로 새로 추가된 파일들을 확인하세요:

```bash
# Load Balancing 핵심 파일들
ls -lh src/load_balancing.py src/throughput_measurement.py src/dht_utils.py

# 새로 추가된 문서들
ls -lh docs/*LOAD_BALANCING*.md docs/*ELICE*.md

# requirements.txt에 numpy 추가되었는지 확인
grep numpy requirements.txt
```

**예상 결과**: `numpy`가 표시되어야 합니다.

## 🔄 서버 재시작

코드를 업데이트한 후에는 서버를 재시작해야 합니다:

```bash
# 기존 프로세스 종료 (Ctrl+C 또는)
pkill -f "python -m src.main"

# 새로 시작
python -m src.main --model meta-llama/Llama-3.1-8B ...
```

