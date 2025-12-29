#!/bin/bash
# simple_auto_pull.sh - 간단한 자동 pull 스크립트 (cron 사용)

set -e

REPO_DIR=${1:-"."}
BRANCH=${2:-"main"}

cd "$REPO_DIR" || exit 1

# 로그 파일
LOG_FILE="auto_pull.log"

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "자동 pull 시작..."

# 원격 변경사항 가져오기
git fetch origin "$BRANCH" 2>&1 >> "$LOG_FILE"

# 로컬과 원격 비교
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" != "$REMOTE" ]; then
    log "새로운 변경사항 발견! Pull 시작..."
    
    OLD_COMMIT="$LOCAL"
    
    if git pull origin "$BRANCH" 2>&1 >> "$LOG_FILE"; then
        NEW_COMMIT=$(git rev-parse HEAD)
        log "업데이트 완료: $OLD_COMMIT -> $NEW_COMMIT"
        
        # Stage 재시작이 필요한 경우 (선택사항)
        # pkill -f "src.main" && ./deploy_direct.sh ...
        
        exit 0
    else
        log "ERROR: Git pull 실패!"
        exit 1
    fi
else
    log "변경사항 없음"
    exit 0
fi

