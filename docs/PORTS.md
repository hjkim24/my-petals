# 포트 설정 가이드

분산 GPU 인스턴스 배포 시 각 인스턴스에서 열어야 하는 포트 목록입니다.

## 포트 할당

각 Stage는 고유한 포트 쌍을 사용합니다:

| Stage | DHT 포트 | RPC 포트 | 용도 |
|-------|----------|----------|------|
| Stage1 | 8002 | 8003 | DHT 부트스트랩, Stage1 서버 |
| Stage2 | 8004 | 8005 | Stage2 서버 |
| Stage3 | 8006 | 8007 | Stage3 서버 (최종) |
| Stage0 | 8008 | 8009 | 클라이언트 |

## 인스턴스별 포트 설정

### 인스턴스 1 (Stage1)
**열어야 할 포트:**
- **8002/TCP** (DHT) - 필수
- **8003/TCP** (RPC) - 필수

**설명:**
- DHT 포트(8002): 다른 Stage들이 DHT 네트워크에 연결하기 위해 필요
- RPC 포트(8003): Stage0에서 Stage1로 RPC 요청을 보내기 위해 필요

### 인스턴스 2 (Stage2)
**열어야 할 포트:**
- **8004/TCP** (DHT) - 필수
- **8005/TCP** (RPC) - 필수

**설명:**
- DHT 포트(8004): DHT 네트워크 참여용
- RPC 포트(8005): Stage1에서 Stage2로 RPC 요청을 보내기 위해 필요

### 인스턴스 3 (Stage3)
**열어야 할 포트:**
- **8006/TCP** (DHT) - 필수
- **8007/TCP** (RPC) - 필수

**설명:**
- DHT 포트(8006): DHT 네트워크 참여용
- RPC 포트(8007): Stage2에서 Stage3로 RPC 요청을 보내기 위해 필요

### 인스턴스 4 (Stage0)
**열어야 할 포트:**
- **8008/TCP** (DHT) - 필수
- **8009/TCP** (RPC) - 필수

**설명:**
- DHT 포트(8008): DHT 네트워크 참여용
- RPC 포트(8009): Stage3에서 Stage0로 토큰을 반환하기 위해 필요

## 방화벽 설정 방법

### AWS EC2
1. **Security Group 설정:**
   - EC2 Console → Security Groups → 해당 인스턴스의 Security Group 선택
   - Inbound Rules → Edit inbound rules
   - 각 포트에 대해 규칙 추가:
     - Type: Custom TCP
     - Port: 8000 (또는 해당 Stage의 포트)
     - Source: 0.0.0.0/0 (또는 다른 인스턴스 IP만 허용)
     - Description: "DHT Port for Stage1"

### Google Cloud Platform (GCP)
```bash
# 각 인스턴스에서 실행
gcloud compute firewall-rules create allow-stage1-dht \
    --allow tcp:8000 \
    --source-ranges 0.0.0.0/0 \
    --description "DHT port for Stage1"

gcloud compute firewall-rules create allow-stage1-rpc \
    --allow tcp:8001 \
    --source-ranges 0.0.0.0/0 \
    --description "RPC port for Stage1"
```

### Azure
```bash
# Azure Portal에서:
# Network Security Group → Inbound security rules → Add
# 각 포트에 대해 규칙 추가
```

### Ubuntu/Debian (UFW)
```bash
# Stage1 인스턴스에서
sudo ufw allow 8002/tcp comment "DHT Port"
sudo ufw allow 8003/tcp comment "RPC Port"

# Stage2 인스턴스에서
sudo ufw allow 8004/tcp comment "DHT Port"
sudo ufw allow 8005/tcp comment "RPC Port"

# Stage3 인스턴스에서
sudo ufw allow 8006/tcp comment "DHT Port"
sudo ufw allow 8007/tcp comment "RPC Port"

# Stage0 인스턴스에서
sudo ufw allow 8008/tcp comment "DHT Port"
sudo ufw allow 8009/tcp comment "RPC Port"

# 방화벽 활성화
sudo ufw enable
sudo ufw status
```

### CentOS/RHEL (firewalld)
```bash
# Stage1 인스턴스에서
sudo firewall-cmd --permanent --add-port=8002/tcp
sudo firewall-cmd --permanent --add-port=8003/tcp
sudo firewall-cmd --reload

# Stage2 인스턴스에서
sudo firewall-cmd --permanent --add-port=8004/tcp
sudo firewall-cmd --permanent --add-port=8005/tcp
sudo firewall-cmd --reload

# Stage3 인스턴스에서
sudo firewall-cmd --permanent --add-port=8006/tcp
sudo firewall-cmd --permanent --add-port=8007/tcp
sudo firewall-cmd --reload

# Stage0 인스턴스에서
sudo firewall-cmd --permanent --add-port=8008/tcp
sudo firewall-cmd --permanent --add-port=8009/tcp
sudo firewall-cmd --reload
```

## 포트 확인 방법

### 포트가 열려있는지 확인
```bash
# 로컬에서 포트 리스닝 확인
sudo netstat -tlnp | grep <PORT>
# 또는
sudo ss -tlnp | grep <PORT>

# 원격에서 포트 접근 가능 여부 확인
telnet <PUBLIC_IP> <PORT>
# 또는
nc -zv <PUBLIC_IP> <PORT>
```

### Python으로 포트 확인
```bash
python3 -c "import socket; s = socket.socket(); s.connect(('PUBLIC_IP', PORT)); print('Port open')"
```

## 보안 권장사항

### 1. Source IP 제한 (권장)
모든 인스턴스의 IP를 알고 있다면, Source를 특정 IP로 제한:

```bash
# AWS Security Group 예시
# Source: 1.2.3.4/32 (Stage1 IP만 허용)
# Source: 5.6.7.8/32 (Stage2 IP만 허용)
# Source: 9.10.11.12/32 (Stage3 IP만 허용)
# Source: 13.14.15.16/32 (Stage0 IP만 허용)
```

### 2. VPC 내부 통신 (가장 안전)
모든 인스턴스가 같은 VPC/네트워크에 있다면:
- Private IP 사용
- Security Group에서 VPC CIDR만 허용
- Public IP는 필요 없음

### 3. SSH 포트 (22번)는 별도 관리
SSH 접근은 별도의 Security Group 규칙으로 관리하세요.

## 포트 충돌 해결

다른 서비스와 포트가 충돌하는 경우, `deploy_direct.sh`나 `deploy.sh`에서 포트를 변경할 수 있습니다:

```bash
# 예: Stage1을 다른 포트로 실행
./deploy_direct.sh 1 gpt2 "10,20,30" "" <PUBLIC_IP> 9000 9001
```

단, 모든 인스턴스에서 동일하게 변경해야 합니다.

## 빠른 참조

### 모든 포트 한 번에 열기 (테스트용)
```bash
# UFW 사용 시
for port in 8002 8003 8004 8005 8006 8007 8008 8009; do
    sudo ufw allow $port/tcp
done
sudo ufw enable
```

### 포트 상태 확인 스크립트
```bash
#!/bin/bash
# check_ports.sh
PUBLIC_IP=$1
PORTS=(8002 8003 8004 8005 8006 8007 8008 8009)

for port in "${PORTS[@]}"; do
    if nc -zv -w 2 $PUBLIC_IP $port 2>&1 | grep -q "succeeded"; then
        echo "✓ Port $port is open"
    else
        echo "✗ Port $port is closed"
    fi
done
```

사용법: `./check_ports.sh <PUBLIC_IP>`

