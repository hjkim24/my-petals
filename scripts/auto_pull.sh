#!/bin/bash
# auto_pull.sh - 주기적으로 git pull을 체크하고 자동으로 업데이트하는 스크립트

set -e

REPO_DIR=${1:-"."}
BRANCH=${2:-"main"}
CHECK_INTERVAL=${3:-60}  # 기본 60초마다 체크
LOG_FILE="${REPO_DIR}/auto_pull.log"

cd "$REPO_DIR" || exit 1

echo "=========================================="
echo "자동 Git Pull 시작"
echo "=========================================="
echo "레포지토리: $(pwd)"
echo "브랜치: $BRANCH"
echo "체크 간격: ${CHECK_INTERVAL}초"
echo "로그 파일: $LOG_FILE"
echo "=========================================="

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Git pull 실행 함수
do_pull() {
    log "Git pull 시작..."
    
    # 현재 커밋 해시 저장
    OLD_COMMIT=$(git rev-parse HEAD)
    
    # 원격 변경사항 가져오기
    git fetch origin "$BRANCH" 2>&1 | tee -a "$LOG_FILE"
    
    # 로컬과 원격 비교
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse "origin/$BRANCH")
    
    if [ "$LOCAL" != "$REMOTE" ]; then
        log "새로운 변경사항 발견! Pull 시작..."
        
        # 변경사항이 있으면 pull
        if git pull origin "$BRANCH" 2>&1 | tee -a "$LOG_FILE"; then
            NEW_COMMIT=$(git rev-parse HEAD)
            log "업데이트 완료: $OLD_COMMIT -> $NEW_COMMIT"
            
            # Stage가 실행 중인 경우 재시작 옵션
            if [ -n "$4" ]; then
                log "Stage 재시작 중..."
                $4
            fi
            
            return 0
        else
            log "ERROR: Git pull 실패!"
            return 1
        fi
    else
        log "변경사항 없음"
        return 0
    fi
}

# 무한 루프로 주기적으로 체크
while true; do
    do_pull "$@"
    sleep "$CHECK_INTERVAL"
done

