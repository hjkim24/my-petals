# DHT 연결 문제 해결 가이드

## 문제
```
P2PDaemonError: Daemon failed to start: failed to connect to bootstrap peers
```

## 원인
Stage1이 `--public_ip` 없이 실행되어 내부 IP로만 리스닝하고 있어서, Stage2가 외부에서 접근할 수 없습니다.

## 해결 방법

### 방법 1: Stage1을 공인 IP로 재시작 (권장)

#### 1단계: Stage1의 공인 IP 확인
Stage1 인스턴스에서:
```bash
curl ifconfig.me
# 결과 예: 119.59.0.14
```

#### 2단계: Stage1 재시작 (Ctrl+C로 중지 후)
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

**변경 사항**: `--public_ip 119.59.0.14` 추가

#### 3단계: Stage1 로그에서 DHT multiaddr 확인
Stage1 실행 후 다음 메시지 확인:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [...]
```

이 multiaddr을 Stage2와 Stage3에서 사용하세요.

#### 4단계: Stage2 재시작
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch" \
    --public_ip STAGE2_공인IP \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

**변경 사항**: `--public_ip STAGE2_공인IP` 추가 (Stage2의 실제 공인 IP)

### 방법 2: Stage2를 독립적으로 실행 (임시 해결책)

Stage1과 연결하지 않고 Stage2를 독립적으로 실행:

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --public_ip STAGE2_공인IP \
    --dht_port 8004 \
    --mean_balance_check_period 120
```

**변경 사항**: `--dht_initial_peers` 제거, `--public_ip` 추가

그러면 Stage2가 첫 번째 서버로 동작하고, Stage3가 Stage2를 찾을 수 있습니다.

### 방법 3: Elice Cloud 내부 네트워크 사용 (같은 네트워크인 경우)

만약 Stage1과 Stage2가 같은 Elice Cloud 네트워크에 있다면, 내부 IP를 사용할 수 있습니다:

```bash
# Stage1: 내부 IP로 실행
python -m src.main ... --dht_port 8002

# Stage2: Stage1의 내부 IP를 initial_peers로 사용
python -m src.main ... \
    --dht_initial_peers "/ip4/STAGE1_내부IP/tcp/8002/p2p/..." \
    --dht_port 8004
```

## 각 인스턴스의 공인 IP 확인

### Stage1
```bash
curl ifconfig.me
```

### Stage2
```bash
curl ifconfig.me
```

### Stage3
```bash
curl ifconfig.me
```

## 권장 순서

1. ✅ Stage1 재시작: `--public_ip` 추가
2. ✅ Stage2 재시작: `--public_ip` 추가, `--dht_initial_peers`에 Stage1 정보 사용
3. ✅ Stage3 실행: `--public_ip` 추가, `--dht_initial_peers`에 Stage1 또는 Stage2 정보 사용

## 주의사항

- Elice Cloud의 보안 그룹/방화벽에서 DHT 포트(8002, 8004, 8006 등)가 열려있는지 확인
- 각 인스턴스의 실제 공인 IP를 정확히 입력해야 함


