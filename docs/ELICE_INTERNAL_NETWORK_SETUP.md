# Elice Cloud 내부 네트워크 설정 가이드

## 확인된 정보

- **Stage1 hostname**: `4df1013acaba`
- **Stage1 내부 IP**: `10.0.2.100`
- **Stage1 터널 IP**: `119.59.0.14` (central-02)

## Elice Cloud 네트워크 구조

Elice Cloud는 여러 계층의 IP를 사용합니다:
1. **내부 IP** (10.0.2.x): 인스턴스 간 직접 통신용
2. **터널 IP** (119.59.0.x): 터널 게이트웨이 IP
3. **공인 IP**: 외부 인터넷에서 접근하는 IP

## 해결 방법

### 방법 1: 내부 IP 사용 (같은 네트워크인 경우 - 권장)

Elice Cloud 인스턴스들이 같은 VPC/네트워크에 있다면, 내부 IP를 사용하는 것이 가장 안정적입니다.

#### Stage1 설정 (10.0.2.100)

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip 10.0.2.100 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

**또는 내부 IP를 자동 감지**:
```bash
INTERNAL_IP=$(getent hosts $(hostname -f) | awk '{print $1}')
echo "Using internal IP: $INTERNAL_IP"

python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip $INTERNAL_IP \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

#### Stage2 설정

먼저 Stage2의 내부 IP 확인:
```bash
# Stage2에서 실행
hostname -f
getent hosts $(hostname -f)
```

예를 들어 Stage2가 `10.0.2.101`이라면:
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip 10.0.2.101 \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

### 방법 2: 공인 IP 사용 (외부 접근 필요 시)

각 인스턴스에서 실제 공인 IP 확인:
```bash
curl ifconfig.me
```

그 IP를 `--public_ip`로 사용

### 방법 3: 0.0.0.0 바인딩 (모든 인터페이스)

내부 IP와 외부 IP 모두 지원:
```bash
# 코드가 이미 0.0.0.0을 지원하므로, public_ip만 설정하면 됨
--public_ip $(getent hosts $(hostname -f) | awk '{print $1}')
```

## 네트워크 연결 테스트

### Stage1에서 Stage2로 연결 테스트
```bash
# Stage2의 IP를 알고 있다면
ping -c 3 10.0.2.101  # Stage2 내부 IP

# 포트 열림 확인
nc -zv 10.0.2.101 8004  # Stage2 DHT 포트
```

### Stage2에서 Stage1로 연결 테스트
```bash
ping -c 3 10.0.2.100  # Stage1 내부 IP
nc -zv 10.0.2.100 8002  # Stage1 DHT 포트
```

## 권장 설정 (자동 감지 스크립트)

각 인스턴스에서 실행:

```bash
#!/bin/bash
# detect_ip.sh

# 1. 내부 IP 감지
INTERNAL_IP=$(getent hosts $(hostname -f 2>/dev/null || hostname) | awk '{print $1}' | head -1)

# 2. 공인 IP 확인 (백업)
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "")

# 3. 최종 IP 선택 (내부 IP 우선)
if [ -n "$INTERNAL_IP" ] && [[ $INTERNAL_IP =~ ^10\. ]]; then
    SELECTED_IP=$INTERNAL_IP
    echo "Using internal IP: $SELECTED_IP"
elif [ -n "$PUBLIC_IP" ]; then
    SELECTED_IP=$PUBLIC_IP
    echo "Using public IP: $SELECTED_IP"
else
    echo "Error: Could not determine IP"
    exit 1
fi

# 4. 실행
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip $SELECTED_IP \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

## Stage별 설정 요약

### Stage1 (10.0.2.100)
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip 10.0.2.100 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

### Stage2 (내부 IP 확인 후)
```bash
# Stage2 IP 확인
STAGE2_IP=$(getent hosts $(hostname -f) | awk '{print $1}')
echo "Stage2 IP: $STAGE2_IP"

python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip $STAGE2_IP \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

## 주의사항

1. **방화벽**: Elice Cloud의 보안 그룹에서 DHT 포트(8002, 8004, 8006)가 열려있는지 확인
2. **네트워크**: 같은 VPC/서브넷에 있는지 확인 (내부 IP 통신 가능 여부)
3. **포트**: 각 인스턴스가 다른 포트를 사용해야 함


