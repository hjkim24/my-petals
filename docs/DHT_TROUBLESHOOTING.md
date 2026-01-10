# DHT 연결 문제 해결 가이드

## 현재 문제
Stage2가 Stage1을 찾지 못하고 있습니다.

## 확인 사항

### 1. Stage1이 정상 등록되었는지 확인

Stage1 로그에서 다음 메시지 확인:
```
✅ INFO:src.dht_utils:Registered server ... on DHT: blocks [0:8], throughput=X.XX rps
✅ INFO:src.dht_utils:Registered 8 blocks for server ...
```

### 2. Stage1과 Stage2의 Peer ID 확인

**Stage1 로그**:
- DHT 등록 Peer ID: `12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch` (초기 로그에서 확인)
- 등록된 블록: [0:8]

**Stage2 설정**:
- `--dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch"`
- ✅ Peer ID는 일치함

### 3. IP 주소와 포트 확인

**Stage1**:
- IP: `119.59.0.14` (추정)
- DHT 포트: `22452` (추정, 로그에서 확인 필요)

**Stage2**:
- `--dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/..."`
- DHT 포트: `8004`

⚠️ **중요**: Stage1의 DHT 포트가 `22452`가 맞는지 확인 필요!
Stage1이 `--dht_port 8002`로 실행되었다면, `--dht_initial_peers`의 포트도 `8002`여야 합니다!

### 4. 모델 이름 일치 확인

두 서버 모두 같은 모델 이름 사용:
```bash
--model meta-llama/Llama-3.1-8B
```
✅ 일치함

## 해결 방법

### 방법 1: Stage1의 실제 DHT 포트 확인

Stage1 로그에서 다음 확인:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [...]
```

또는 Stage1이 등록한 DHT 포트 확인 (예: `--dht_port 8002`)

### 방법 2: Stage2의 initial_peers 수정

Stage1이 `--dht_port 8002`로 실행되었다면:
```bash
--dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch"
```

### 방법 3: Stage3 실행 시도

현재 Stage2가 정상 작동 중이므로, Stage3를 실행하여:
- Stage2의 블록을 찾을 수 있는지 확인
- Stage3가 다른 블록을 선택하는지 확인

## 임시 해결책: Stage3 실행

Stage2가 정상 작동 중이므로, Stage3를 실행하여 Load Balancing이 작동하는지 확인:

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/STAGE2_IP/tcp/8004/p2p/12D3KooWHG6QEACTQvZjTwWhs2xnyhFpf2mxPc27CRCyxhYAzb9V" \
    --dht_port 8006 \
    --mean_balance_check_period 120
```

**주의**: Stage2의 실제 IP 주소와 Peer ID를 사용해야 합니다!


