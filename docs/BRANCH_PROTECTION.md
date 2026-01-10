# Main Branch 보호 - 안전한 병합 가이드

⚠️ **중요**: Main 브랜치는 팀원들이 공유하는 브랜치이므로 **절대 직접 수정하지 않습니다!**

## ✅ 안전한 방법: Jaewon 브랜치에서만 작업

### Main 브랜치는 읽기 전용으로만 사용

아래 명령어들은 **Main 브랜치를 변경하지 않고**, Main의 코드를 Jaewon으로 가져오는 것입니다.

## 📋 안전한 병합 절차

### 1단계: Jaewon 브랜치로 이동 (Main 브랜치는 만지지 않음)

```bash
# ✅ 안전: Jaewon 브랜치로 이동 (Main은 건드리지 않음)
git checkout Jaewon

# 현재 브랜치 확인 (Jaewon인지 확인)
git branch
# 출력: * Jaewon  <- 현재 Jaewon 브랜치에 있음
#       main    <- Main 브랜치는 그대로 둠
```

### 2단계: Main 브랜치의 최신 정보만 가져오기 (읽기 전용)

```bash
# ✅ 안전: 원격 저장소 정보만 가져오기 (Main 브랜치 변경 없음)
git fetch origin

# ✅ 안전: Main 브랜치의 최신 변경사항 확인 (읽기만 함)
git log origin/main --oneline -10
```

### 3단계: Main의 변경사항을 Jaewon으로 병합 (Main은 그대로)

```bash
# ✅ 안전: Main의 변경사항을 Jaewon으로 가져오기
# Main 브랜치 자체는 전혀 변경되지 않음!
git merge origin/main

# 또는
git rebase origin/main
```

**중요**: 이 명령어는:
- ✅ Jaewon 브랜치에 Main의 변경사항을 가져옴
- ✅ Main 브랜치는 전혀 변경하지 않음
- ✅ Main 브랜치는 원격/로컬 모두 그대로 유지됨

### 4단계: Jaewon 브랜치만 푸시 (Main은 건드리지 않음)

```bash
# ✅ 안전: Jaewon 브랜치만 원격에 푸시
git push origin Jaewon

# ❌ 절대 하지 말 것: Main 브랜치에 푸시
# git push origin main  <- 이건 절대 하지 마세요!
```

## ⚠️ 절대 하지 말아야 할 것들

### ❌ Main 브랜치를 직접 수정하는 명령어들

```bash
# ❌ 절대 하지 마세요!
git checkout main              # Main 브랜치로 이동 (필요시만, 수정 안 함)
git push origin main           # Main 브랜치에 푸시 (절대 금지!)
git reset --hard origin/main   # Main 브랜치 리셋 (절대 금지!)
git merge Jaewon main          # Main 브랜치에 Jaewon 병합 (절대 금지!)
```

### ❌ 실수로 Main 브랜치를 수정한 경우 대처법

```bash
# 만약 실수로 Main 브랜치에서 작업을 시작했다면:
git status  # 현재 상태 확인

# Main 브랜치에 커밋하지 않았다면:
git stash   # 변경사항 임시 저장
git checkout Jaewon  # Jaewon으로 이동
git stash pop  # 변경사항 복원 (Jaewon에서)

# Main 브랜치에 커밋했다면:
git checkout main
git reset --hard origin/main  # 원격 Main으로 되돌림 (로컬만)
git checkout Jaewon
# 변경사항은 이미 Jaewon에 있다면 괜찮음
```

## ✅ 안전한 작업 흐름 (완전한 예시)

```bash
# 1. 저장소 클론 (처음만)
git clone https://github.com/hjkim24/my-petals.git
cd my-petals

# 2. Jaewon 브랜치로 이동 (Main은 건드리지 않음)
git checkout Jaewon

# 3. 현재 브랜치 확인 (Jaewon인지 확인)
git branch
# * Jaewon  <- 확인!

# 4. 원격 저장소 정보만 가져오기 (읽기 전용)
git fetch origin

# 5. Main 브랜치와 Jaewon 브랜치 비교 (읽기만)
git log Jaewon..origin/main --oneline  # Main에만 있는 커밋 확인

# 6. Main의 변경사항을 Jaewon으로 병합 (Main은 그대로)
git merge origin/main

# 7. 충돌 발생 시 (Jaewon 브랜치에서만 해결)
# ... 충돌 파일 수동 편집 ...
git add .
git commit -m "Merge main into Jaewon"

# 8. Jaewon 브랜치만 푸시 (Main은 건드리지 않음)
git push origin Jaewon
```

## 🔍 Main 브랜치 보호 확인

### 작업 전 확인 사항

```bash
# 1. 현재 브랜치 확인 (Main이 아닌지 확인)
git branch
# ✅ * Jaewon  <- Jaewon이면 안전
# ❌ * main    <- Main이면 Jaewon으로 이동

# 2. Main 브랜치가 변경되지 않았는지 확인
git checkout main  # 확인만 (수정 안 함)
git status
git diff origin/main  # 로컬과 원격 Main 비교 (차이가 없어야 함)
git checkout Jaewon  # 다시 Jaewon으로 돌아옴
```

### 작업 후 확인 사항

```bash
# Main 브랜치가 여전히 그대로인지 확인
git checkout main
git status  # "Your branch is up to date with 'origin/main'" 확인
git diff origin/main  # 차이가 없어야 함
git checkout Jaewon  # Jaewon으로 돌아옴
```

## 💡 추가 보안 팁

### 1. Git 설정으로 Main 브랜치 보호 (로컬에서)

```bash
# Main 브랜치에 직접 푸시 방지 (pre-push hook 설정)
# .git/hooks/pre-push 파일 생성 (선택사항)

cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
protected_branch='main'
current_branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

if [ $protected_branch = $current_branch ]; then
    echo "❌ Protected branch '$protected_branch' cannot be pushed to!"
    echo "Please create a new branch for your changes."
    exit 1
fi

exit 0
EOF

chmod +x .git/hooks/pre-push
```

### 2. GitHub에서 Main 브랜치 보호 규칙 설정 (팀장/관리자 권한 필요)

GitHub 저장소 설정에서:
- Settings → Branches → Branch protection rules
- Main 브랜치에 protection rule 추가:
  - ✅ Require pull request reviews
  - ✅ Require status checks to pass
  - ✅ Include administrators (모든 사람 적용)
  - ✅ Restrict who can push to matching branches

## 📊 작업 요약

| 작업 | Main 브랜치 | Jaewon 브랜치 | 안전 여부 |
|------|------------|--------------|----------|
| `git checkout Jaewon` | 변경 없음 | 이동만 | ✅ 안전 |
| `git fetch origin` | 읽기만 | 읽기만 | ✅ 안전 |
| `git merge origin/main` | 변경 없음 | 변경됨 | ✅ 안전 |
| `git push origin Jaewon` | 변경 없음 | 푸시됨 | ✅ 안전 |
| `git push origin main` | 푸시됨 | 변경 없음 | ❌ **금지!** |

## 🎯 핵심 정리

**질문**: Main 브랜치는 안 건드는 거지?

**답변**: 
✅ **네, Main 브랜치는 전혀 건드리지 않습니다!**

- Main 브랜치는 **읽기 전용**으로만 사용 (정보만 가져옴)
- 모든 수정 작업은 **Jaewon 브랜치에서만** 진행
- Main 브랜치에 **절대 push하지 않음**
- `git merge origin/main`은 Main의 코드를 Jaewon으로 가져올 뿐, Main 자체는 변경하지 않음

**안전한 명령어 순서**:
```bash
git checkout Jaewon              # ✅ Jaewon으로 이동
git fetch origin                 # ✅ 정보만 가져오기
git merge origin/main            # ✅ Main을 Jaewon으로 병합 (Main은 그대로)
git push origin Jaewon           # ✅ Jaewon만 푸시
```

Main 브랜치는 팀원들이 공유하는 브랜치이므로 안전하게 보호됩니다!
