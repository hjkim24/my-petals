# Stage1-Stage2 연결 문제 해결 가이드

## 🔴 발견된 문제들

### 문제 1: Stage1에 `--public_ip` 없음
- Stage1이 DHT에 올바른 주소로 announce하지 못함
- Stage2가 Stage1을 찾을 수 없음

### 문제 2: Stage2의 `dht_initial_peers` IP 잘림
- `119.59.0.14`가 `9.0.14`로 잘려서 입력됨
- 잘못된 IP로 연결 시도하여 실패

### 문제 3: DHT 조회 불안정
- 0개 → 8개 → 0개로 변하는 현상
- DHT 연결이 불안정하거나 전파 시간 문제

## ✅ 해결 방법

### Step 1: Stage1 재시작 (public_ip 추가)

**Stage1 인스턴스에서**:

먼저 Stage1의 IP 확인:
```bash
getent hosts $(hostname -f)
# 결과 예시: 10.0.2.100 또는 터널 IP 119.59.0.14
```

#### 옵션 A: 내부 IP 사용 (같은 네트워크, 권장)

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') \
    --mean_balance_check_period 120
```

#### 옵션 B: 터널 IP 사용 (외부 접근)

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_port 8002 \
    --rpc_port 8003 \
    --public_ip 119.59.0.14 \
    --public_dht_port 22452 \
    --public_rpc_port 50192 \
    --mean_balance_check_period 120
```

**중요**: Stage1 실행 후 다음 로그를 확인하세요:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): ...
INFO:__main__:P2P initialized for Load Balancing server, PeerID: 12D3KooW...
```

### Step 2: Stage1의 PeerID와 Multiaddr 확인

Stage1 로그에서 다음 정보를 복사:
1. **PeerID**: `12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU`
2. **Multiaddr** (또는 직접 구성):
   - 내부 IP 사용 시: `/ip4/10.0.2.100/tcp/8002/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU`
   - 터널 IP 사용 시: `/ip4/119.59.0.14/tcp/22452/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU`

### Step 3: Stage2 재시작 (올바른 dht_initial_peers 사용)

**Stage2 인스턴스에서**:

먼저 Stage2의 IP 확인:
```bash
getent hosts $(hostname -f)
```

#### 옵션 A: 내부 IP 사용 (Stage1과 같은 네트워크)

```bash
# Stage1의 내부 IP가 10.0.2.100이라고 가정
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" \
    --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') \
    --dht_port 8004 \
    --rpc_port 8005 \
    --mean_balance_check_period 120
```

#### 옵션 B: 터널 IP 사용

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" \
    --public_ip 119.59.0.14 \
    --public_dht_port 29354 \
    --public_rpc_port 15930 \
    --dht_port 8004 \
    --rpc_port 8005 \
    --mean_balance_check_period 120
```

**⚠️ 주의**: 
- `dht_initial_peers`의 IP 주소가 **완전한 IP 주소**여야 함 (`119.59.0.14`, 절대 `9.0.14` 아님!)
- PeerID도 Stage1의 실제 PeerID와 일치해야 함

### Step 4: 연결 확인

Stage2 실행 후 다음 로그가 나타나야 합니다:

```
INFO:src.dht_utils:Retrieved 8 module infos from DHT (total_blocks=32)
INFO:src.dht_utils:Block coverage: 0 to 7 (8 blocks found)
INFO:__main__:Found 1 unique server(s) in DHT: ['12D3KooWSfbNJ...']
INFO:__main__:Selected blocks: [8, 9, 10, 11, 12, 13, 14, 15] (start=8, end=16)
```

**성공 신호**:
- ✅ Stage2가 Stage1의 블록 [0-7]을 찾음
- ✅ Stage2가 다른 블록 [8-15]를 선택
- ✅ `Retrieved 8 module infos` 또는 더 많은 모듈 정보 조회

**실패 신호**:
- ❌ `Retrieved 0 module infos` 계속 반복
- ❌ `No existing servers found`
- ❌ Stage2도 [0-7] 블록 선택

## 🔧 추가 문제 해결

### DHT 조회가 여전히 불안정한 경우

DHT 전파 시간을 늘리기 위해 코드를 수정할 수 있습니다:

```python
# src/main.py에서 retry_delay 증가
retry_delay = 5.0  # 2.0 → 5.0으로 증가
```

또는 환경 변수로 설정:
```bash
export DHT_RETRY_DELAY=5.0
```

### 방화벽 확인

각 인스턴스에서 포트가 열려있는지 확인:

```bash
# Stage2에서 Stage1으로 연결 테스트
nc -zv 10.0.2.100 8002  # 내부 IP 사용 시
# 또는
nc -zv 119.59.0.14 22452  # 터널 IP 사용 시
```

## 📋 체크리스트

- [ ] Stage1에 `--public_ip` 추가
- [ ] Stage1 실행 후 PeerID와 Multiaddr 확인
- [ ] Stage2의 `--dht_initial_peers`에 **완전한 IP 주소** 입력
- [ ] Stage2에 `--public_ip` 추가
- [ ] Stage2가 Stage1의 블록을 찾는지 확인
- [ ] Stage2가 다른 블록 범위를 선택하는지 확인

