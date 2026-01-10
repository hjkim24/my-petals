# 엘리스 클라우드 Load Balancing 테스트 가이드

## 준비 사항

1. **환경 설정**
   ```bash
   ./scripts/initial_install.sh
   source venv/bin/activate  # 또는 . venv/bin/activate (Linux/Mac)
   # Windows: venv\Scripts\activate
   hf auth login
   ```

2. **모델 정보 확인**
   - 모델: `meta-llama/Llama-3.1-8B`
   - 전체 레이어 수: **32개** (total_blocks=32)
   - 각 서버가 담당할 블록 수: **4-8개** 권장

## Load Balancing 테스트 시나리오

### ⚠️ 중요: Load Balancing vs 고정 Splits

Load Balancing 모드는 **고정된 splits를 사용하지 않습니다**. 각 서버가 동적으로 최적의 블록을 선택합니다.

### 방법 1: 각 Stage를 별도 인스턴스에서 실행 (권장)

각 stage는 별도 인스턴스에서 실행하되, Load Balancing을 사용하여 동일 stage 내에서 블록이 자동 분배됩니다.

#### Step 1: 첫 번째 서버 시작 (Stage 1)

```bash
# stage-1 인스턴스에서 실행
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
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

**출력 확인**: 로그에서 다음 정보 확인
```
DHT visible multiaddrs (use for --dht_initial_peers): [...]
Selected blocks: [0, 1, 2, 3, 4, 5, 6, 7]
P2P initialized for Load Balancing server, PeerID: ...
Load Balancing server ready, blocks=[0, 1, 2, 3, 4, 5, 6, 7], throughput=XX.XX rps
```

**DHT Peer 주소 복사**: 출력된 multiaddr을 복사합니다.
예: `/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW...`

#### Step 2: 두 번째 서버 시작 (같은 Stage 1)

```bash
# stage-2 인스턴스에서 실행 (같은 Stage 1로!)
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW..." \
    --dht_port 8004 \
    --rpc_port 8005 \
    --public_ip 119.59.0.14 \
    --public_dht_port 29354 \
    --public_rpc_port 15930 \
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

**출력 확인**: 다른 블록이 선택되었는지 확인
```
Selected blocks: [8, 9, 10, 11, 12, 13, 14, 15]  # 첫 번째와 다른 블록
```

#### Step 3: 세 번째 서버 시작 (같은 Stage 1)

```bash
# stage-3 인스턴스에서 실행 (같은 Stage 1로!)
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW..." \
    --dht_port 8006 \
    --rpc_port 8007 \
    --public_ip 119.59.0.14 \
    --public_dht_port 59491 \
    --public_rpc_port 38548 \
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

#### Step 4: 네 번째 서버 시작 (Stage 1, 마지막 블록 담당)

```bash
# stage-0 인스턴스에서 실행 (같은 Stage 1로!)
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW..." \
    --dht_port 8008 \
    --rpc_port 8009 \
    --public_ip 119.59.0.14 \
    --public_dht_port 41826 \
    --public_rpc_port 23619 \
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

**예상 결과**: 4개 서버가 32개 블록을 분할
- 서버 1: 블록 0-7 (또는 다른 최적 구간)
- 서버 2: 블록 8-15 (또는 다른 최적 구간)
- 서버 3: 블록 16-23 (또는 다른 최적 구간)
- 서버 4: 블록 24-31 (또는 다른 최적 구간)

### 방법 2: Stage별로 분리 실행 (기존 방식 + Load Balancing)

각 stage를 다른 블록 범위로 분리하되, 각 stage 내에서도 Load Balancing 사용:

#### Stage 1 서버 (블록 0-7 범위 담당)

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
    --public_rpc_port 50192
```

#### Stage 2 서버 (블록 8-15 범위 담당)

```bash
# 첫 번째 서버의 DHT peer 정보 필요
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/첫서버PeerID" \
    --dht_port 8004 \
    --rpc_port 8005 \
    --public_ip 119.59.0.14 \
    --public_dht_port 29354 \
    --public_rpc_port 15930
```

## ⚠️ 주의사항

### 1. Load Balancing 모드에서는 splits가 무시됨

`--splits "8,16,24"` 옵션은 Load Balancing 모드에서 **무시됩니다**.
블록 선택은 자동으로 처리량 기반으로 결정됩니다.

### 2. 모든 서버는 같은 stage로 실행

Load Balancing을 테스트하려면 **모든 서버를 같은 stage(예: stage 1)로 실행**해야 합니다.
각 서버가 다른 블록 범위를 자동으로 선택합니다.

### 3. DHT 초기 피어 설정

- **첫 번째 서버**: `--dht_initial_peers` 옵션 없음
- **나머지 서버들**: 첫 번째 서버의 DHT multiaddr을 `--dht_initial_peers`로 전달

### 4. Client (Stage 0)는 별도 실행

```bash
# Client는 별도로 실행 (Load Balancing과 무관)
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 0 \
    --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/서버1PeerID,/ip4/119.59.0.14/tcp/29354/p2p/서버2PeerID" \
    --dht_port 8008 \
    --rpc_port 8009 \
    --public_ip 119.59.0.14 \
    --public_dht_port 41826 \
    --public_rpc_port 23619 \
    --prompt "Hello, how are you?"
```

## 모니터링

### 1. 블록 선택 확인

각 서버의 로그에서 다음 확인:
```
Selected blocks: [0, 1, 2, 3, 4, 5, 6, 7]
Load Balancing server ready, blocks=[0, 1, 2, 3, 4, 5, 6, 7], throughput=15.23 rps
```

### 2. 재조정 확인

주기적으로 (평균 120초마다) 다음 로그 확인:
```
Swarm balance quality: 85.2% (initial=12.45, new=14.62)
Load balancing detected imbalance, will rebalance blocks
```

### 3. 처리량 측정

각 서버의 처리량이 로그에 표시됩니다:
- `throughput=XX.XX rps`: 서버 처리량

## 문제 해결

### 문제 1: "Failed to import Load Balancing modules"

```bash
# numpy 설치 확인
pip install numpy>=1.20.0

# 모듈 경로 확인
python -c "from src.load_balancing import choose_best_blocks; print('OK')"
```

### 문제 2: 서버들이 같은 블록 선택

- DHT 연결 확인: `--dht_initial_peers`가 올바른지 확인
- 여러 서버가 동시에 시작되면 race condition 가능 (약간의 지연 추가)

### 문제 3: 재조정이 너무 자주 발생

```bash
# balance_quality 값을 높임 (더 보수적)
--balance_quality 0.90  # 10% 이상 개선 시에만 재조정

# 재조정 주기 늘림
--mean_balance_check_period 300  # 5분
```

### 문제 4: 파이프라인 연결 실패

- 모든 서버가 같은 DHT 네트워크에 연결되어 있는지 확인
- 방화벽 설정 확인 (포트 개방)
- `--public_ip`와 `--public_dht_port`, `--public_rpc_port` 확인

## 빠른 시작 스크립트

각 인스턴스에서 실행할 스크립트를 미리 만들어두면 편리합니다.

### stage1_server1.sh
```bash
#!/bin/bash
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
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

### stage1_server2.sh (DHT peer 정보 필요)
```bash
#!/bin/bash
# 첫 번째 서버의 DHT peer ID를 여기에 입력
DHT_PEER="/ip4/119.59.0.14/tcp/22452/p2p/첫서버PeerID"

python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "$DHT_PEER" \
    --dht_port 8004 \
    --rpc_port 8005 \
    --public_ip 119.59.0.14 \
    --public_dht_port 29354 \
    --public_rpc_port 15930 \
    --balance_quality 0.75 \
    --mean_balance_check_period 120
```

## 예상 결과

Load Balancing이 정상 작동하면:
1. 각 서버가 서로 다른 블록 구간 선택
2. 전체 파이프라인 연결 유지 (0~31 블록 모두 커버)
3. 주기적으로 처리량 체크 및 불균형 시 재조정
4. 서버 추가/제거 시 자동으로 블록 재분배

논문에서 언급한 **85-90% 처리량 유지** 효과를 확인할 수 있습니다.
