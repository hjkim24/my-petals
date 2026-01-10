# Elice Cloud IP 설정 가이드

## 확인된 정보

- **central-02**: `119.59.0.14` (Stage1으로 추정)
- **central-01**: `119.59.0.13` (다른 인스턴스)

## 각 인스턴스의 IP 확인

### Stage1 (central-02, 119.59.0.14)
```bash
# 터널 IP 확인
getent hosts central-02.tcp.tunnel.elice.io
# 결과: 119.59.0.14

# 실제 공인 IP 확인
curl ifconfig.me
```

### Stage2 인스턴스
```bash
# 터널 호스트 이름 확인
hostname
# 또는
hostname -f

# 터널 IP 확인
getent hosts $(hostname -f)
# 또는 직접 확인
getent hosts central-XX.tcp.tunnel.elice.io

# 실제 공인 IP 확인
curl ifconfig.me
```

### Stage3 인스턴스
```bash
# 동일한 방법으로 확인
curl ifconfig.me
getent hosts $(hostname -f)
```

## Elice Cloud 터널 설정

Elice Cloud는 TCP 터널을 사용하므로:
1. 각 인스턴스는 터널 IP(예: 119.59.0.14)를 가짐
2. 실제 공인 IP는 다를 수 있음
3. 같은 네트워크 내에서는 터널 IP로 통신 가능할 수 있음

## 해결 방법

### 방법 1: 터널 IP 사용 (같은 네트워크인 경우)

#### Stage1 설정 (central-02)
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip 119.59.0.14 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

#### Stage2 설정 (터널 IP 확인 후)
```bash
# Stage2의 터널 IP 확인
STAGE2_TUNNEL_IP=$(getent hosts $(hostname -f) | awk '{print $1}')
echo "Stage2 tunnel IP: $STAGE2_TUNNEL_IP"

python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip $STAGE2_TUNNEL_IP \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

### 방법 2: 실제 공인 IP 사용 (외부 접근 필요 시)

각 인스턴스에서 실제 공인 IP 확인:
```bash
curl ifconfig.me
```

그 IP를 `--public_ip`로 사용

### 방법 3: 자동 IP 감지 (권장)

각 인스턴스에서:
```bash
# 터널 IP 자동 감지
TUNNEL_IP=$(getent hosts $(hostname -f 2>/dev/null || hostname) | awk '{print $1}' | head -1)
if [ -z "$TUNNEL_IP" ]; then
    TUNNEL_IP=$(curl -s ifconfig.me)
fi
echo "Using IP: $TUNNEL_IP"

python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip $TUNNEL_IP \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

## Stage별 설정 예시

### Stage1 (119.59.0.14)
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip 119.59.0.14 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --mean_balance_check_period 120
```

### Stage2 (터널 IP 확인 필요)
먼저 Stage2의 터널 IP 확인:
```bash
hostname -f
getent hosts $(hostname -f)
```

예를 들어 `119.59.0.13`이라면:
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip 119.59.0.13 \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

### Stage3 (터널 IP 확인 필요)
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip STAGE3_TUNNEL_IP \
    --dht_port 8006 \
    --mean_balance_check_period 120
```

## 확인 사항

1. **Elice Cloud 방화벽/보안 그룹**: DHT 포트(8002, 8004, 8006)가 열려있는지 확인
2. **터널 설정**: 각 인스턴스가 같은 터널 네트워크에 있는지 확인
3. **포트 충돌**: 각 인스턴스가 다른 포트를 사용하는지 확인

## 빠른 확인 명령어

```bash
# 1. 현재 인스턴스 정보
echo "Hostname: $(hostname)"
echo "Tunnel IP: $(getent hosts $(hostname -f 2>/dev/null || hostname) | awk '{print $1}')"
echo "Public IP: $(curl -s ifconfig.me)"

# 2. 네트워크 연결 테스트
ping -c 3 119.59.0.14  # Stage1로 ping 테스트

# 3. 포트 열림 확인
nc -zv 119.59.0.14 8002  # Stage1 DHT 포트 확인
```


