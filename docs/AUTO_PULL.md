# 자동 Git Pull 설정 가이드

main 브랜치에 push할 때마다 각 GPU 인스턴스에서 자동으로 pull을 받도록 설정하는 방법입니다.

## 방법 1: Systemd 서비스 (권장)

백그라운드에서 지속적으로 실행되며, 서버 재부팅 후에도 자동으로 시작됩니다.

### 설정 방법

각 GPU 인스턴스에서 실행:

```bash
# 1. 레포지토리 클론 (처음 한 번만)
git clone <your-repo>
cd my-petals

# 2. 자동 pull 서비스 설정
chmod +x setup_auto_pull.sh
sudo ./setup_auto_pull.sh /path/to/my-petals main 60

# 파라미터 설명:
# - /path/to/my-petals: 레포지토리 경로
# - main: 브랜치 이름 (기본값: main)
# - 60: 체크 간격(초) (기본값: 60초)
```

### 서비스 관리

```bash
# 서비스 상태 확인
sudo systemctl status my-petals-auto-pull

# 서비스 시작
sudo systemctl start my-petals-auto-pull

# 서비스 중지
sudo systemctl stop my-petals-auto-pull

# 서비스 재시작
sudo systemctl restart my-petals-auto-pull

# 서비스 비활성화 (재부팅 시 자동 시작 안 함)
sudo systemctl disable my-petals-auto-pull

# 로그 확인
tail -f /path/to/my-petals/auto_pull.log
```

---

## 방법 2: Cron Job (간단한 방법)

주기적으로 체크하는 간단한 방법입니다.

### 설정 방법

각 GPU 인스턴스에서 실행:

```bash
# 1. 레포지토리 클론
git clone <your-repo>
cd my-petals

# 2. Cron job 추가 (매 1분마다 체크)
chmod +x simple_auto_pull.sh
crontab -e

# 다음 줄 추가:
*/1 * * * * /path/to/my-petals/simple_auto_pull.sh /path/to/my-petals main >> /path/to/my-petals/auto_pull.log 2>&1
```

### Cron 표현식 예시

```bash
# 매 1분마다
*/1 * * * * /path/to/my-petals/simple_auto_pull.sh /path/to/my-petals main

# 매 5분마다
*/5 * * * * /path/to/my-petals/simple_auto_pull.sh /path/to/my-petals main

# 매 시간마다
0 * * * * /path/to/my-petals/simple_auto_pull.sh /path/to/my-petals main
```

### Cron 관리

```bash
# 현재 cron job 확인
crontab -l

# cron job 편집
crontab -e

# cron job 삭제
crontab -r
```

---

## 방법 3: 수동 실행 (테스트용)

백그라운드에서 직접 실행하는 방법입니다.

### 실행 방법

```bash
# 백그라운드에서 실행
nohup ./auto_pull.sh /path/to/my-petals main 60 > auto_pull.log 2>&1 &

# 프로세스 확인
ps aux | grep auto_pull

# 프로세스 종료
pkill -f auto_pull.sh
```

---

## 방법 4: GitHub Webhook (고급)

GitHub에서 push 이벤트를 받아 즉시 pull하는 방법입니다.

### 설정 방법

1. **각 GPU 인스턴스에 webhook 서버 설정:**

```bash
# webhook_server.py 생성 필요
# GitHub에서 push 이벤트를 받으면 git pull 실행
```

2. **GitHub에서 Webhook 설정:**
   - Repository Settings → Webhooks → Add webhook
   - Payload URL: `http://<GPU_IP>:8080/webhook`
   - Content type: `application/json`
   - Events: `Just the push event`

---

## 추천 설정

### 체크 간격 권장값

- **개발 중**: 30-60초 (빠른 반영)
- **프로덕션**: 5-10분 (부하 감소)

### Stage 자동 재시작 (선택사항)

코드 변경 후 Stage를 자동으로 재시작하려면 `auto_pull.sh`를 수정:

```bash
# auto_pull.sh 마지막 부분 수정
if [ -n "$4" ]; then
    log "Stage 재시작 중..."
    # Stage 종료
    pkill -f "src.main --stage 1" || true
    pkill -f "src.main --stage 2" || true
    pkill -f "src.main --stage 3" || true
    pkill -f "src.main --stage 0" || true
    
    # 잠시 대기
    sleep 2
    
    # Stage 재시작 (deploy_direct.sh 사용)
    ./deploy_direct.sh 1 gpt2 "10,20,30" "" <PUBLIC_IP> 8002 8003
    # ... 다른 stage들도 재시작
fi
```

---

## 문제 해결

### Git pull 실패 시

```bash
# 로그 확인
tail -f auto_pull.log

# 수동으로 pull 시도
git pull origin main

# 충돌 해결
git stash
git pull origin main
git stash pop
```

### 서비스가 실행되지 않는 경우

```bash
# systemd 로그 확인
sudo journalctl -u my-petals-auto-pull -f

# 서비스 파일 확인
cat /etc/systemd/system/my-petals-auto-pull.service

# 권한 확인
ls -la /path/to/my-petals/auto_pull.sh
```

### 충돌 발생 시

자동 pull은 충돌을 자동으로 해결하지 않습니다. 수동으로 해결해야 합니다:

```bash
# 충돌 확인
git status

# 충돌 해결 후
git add .
git commit -m "Resolve merge conflict"
```

---

## 보안 고려사항

1. **SSH 키 사용**: HTTPS 대신 SSH를 사용하는 것을 권장합니다.
2. **읽기 전용 권한**: 각 GPU 인스턴스에는 읽기 전용 권한만 부여하는 것을 권장합니다.
3. **Private Repository**: 프로덕션 환경에서는 private repository를 사용하세요.

---

## 빠른 시작

가장 간단한 방법 (Cron 사용):

```bash
# 각 GPU 인스턴스에서 실행
cd /path/to/my-petals
chmod +x simple_auto_pull.sh
crontab -e

# 다음 줄 추가 (매 1분마다 체크):
*/1 * * * * /path/to/my-petals/simple_auto_pull.sh /path/to/my-petals main
```

이제 main 브랜치에 push할 때마다 최대 1분 내에 자동으로 pull됩니다!

