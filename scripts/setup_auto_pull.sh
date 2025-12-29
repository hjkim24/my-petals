#!/bin/bash
# setup_auto_pull.sh - systemd 서비스로 자동 pull 설정

set -e

REPO_DIR=${1:-"$(pwd)"}
BRANCH=${2:-"main"}
CHECK_INTERVAL=${3:-60}
USER=${4:-"$USER"}

echo "=========================================="
echo "자동 Git Pull Systemd 서비스 설정"
echo "=========================================="
echo "레포지토리: $REPO_DIR"
echo "브랜치: $BRANCH"
echo "체크 간격: ${CHECK_INTERVAL}초"
echo "사용자: $USER"
echo "=========================================="

# 절대 경로로 변환
REPO_DIR=$(cd "$REPO_DIR" && pwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# systemd 서비스 파일 생성
SERVICE_FILE="/etc/systemd/system/my-petals-auto-pull.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Auto Git Pull for My Petals
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=/bin/bash $SCRIPT_DIR/auto_pull.sh $REPO_DIR $BRANCH $CHECK_INTERVAL
Restart=always
RestartSec=10
StandardOutput=append:$REPO_DIR/auto_pull.log
StandardError=append:$REPO_DIR/auto_pull.log

[Install]
WantedBy=multi-user.target
EOF

echo "Systemd 서비스 파일 생성 완료: $SERVICE_FILE"

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable my-petals-auto-pull.service
sudo systemctl start my-petals-auto-pull.service

echo "=========================================="
echo "자동 Git Pull 서비스 시작됨"
echo "=========================================="
echo "서비스 상태 확인: sudo systemctl status my-petals-auto-pull"
echo "로그 확인: tail -f $REPO_DIR/auto_pull.log"
echo "서비스 중지: sudo systemctl stop my-petals-auto-pull"
echo "서비스 시작: sudo systemctl start my-petals-auto-pull"
echo "=========================================="

