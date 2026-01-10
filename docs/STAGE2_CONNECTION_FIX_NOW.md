# Stage2 연결 문제 즉시 해결

## 🔴 현재 문제

1. **포트 불일치**: Stage2가 `tcp/8002`로 연결 시도하지만, Stage1은 `--public_dht_port 29354` 사용
2. **PeerID 확인**: Stage1의 DHT PeerID는 `12D3KooWEjiUjNY6a9rPftfm1gicLcJSQFC9KnSsvu9NfW3syazS` (P2P PeerID 아님)

## ✅ 즉시 해결 방법

### 방법 1: 포트 수정 (빠른 수정)

Stage2에서 포트를 `29354`로 변경:

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/119.59.0.14/tcp/29354/p2p/12D3KooWEjiUjNY6a9rPftfm1gicLcJSQFC9KnSsvu9NfW3syazS" --public_ip 119.59.0.14 --public_dht_port 29354 --public_rpc_port 15930 --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

**변경 사항**: `tcp/8002` → `tcp/29354`

### 방법 2: Stage1 재시작 후 Multiaddr 확인 (권장)

Stage1을 재시작하면 이제 multiaddr 로그가 나타납니다:

**Stage1 재시작**:
```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip 119.59.0.14 --public_dht_port 29354 --public_rpc_port 50192 --mean_balance_check_period 120
```

**로그에서 확인**:
```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): ['/ip4/119.59.0.14/tcp/29354/p2p/...']
```

이 multiaddr을 그대로 Stage2의 `--dht_initial_peers`에 사용하세요.

### 방법 3: 내부 IP 사용 (같은 네트워크인 경우)

Elice Cloud에서 같은 네트워크에 있다면 내부 IP 사용:

#### Stage1 (내부 IP)

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --mean_balance_check_period 120
```

#### Stage2 (내부 IP)

```bash
# Stage1의 내부 IP 확인 필요 (예: 10.0.2.100)
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/<Stage1_DHT_PeerID>" --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

## 🔍 문제 진단

### 네트워크 연결 테스트

Stage2에서 Stage1으로 연결 테스트:

```bash
# 터널 IP로 테스트
nc -zv 119.59.0.14 29354

# 결과 확인
# Connection to 119.59.0.14 port 29354 [tcp/*] succeeded!  ← 성공
# Connection refused  ← 실패 (방화벽 또는 포트 미오픈)
```

### PeerID 확인

**중요**: DHT PeerID와 P2P PeerID는 다릅니다!

- **DHT PeerID**: `dht.peer_id` - DHT 연결용
- **P2P PeerID**: `p2p.peer_id` - RPC 통신용

`--dht_initial_peers`에는 **DHT PeerID**를 사용해야 합니다.

Stage1 로그를 보면:
- `Registered server 12D3KooWHNDaWrqW...` - 서버 등록용 PeerID
- `P2P initialized for Load Balancing server, PeerID: 12D3KooWEjiUjNY6a...` - P2P PeerID

**DHT PeerID는 로그에 표시되지 않았습니다** (코드 수정 후 나타날 것).

## 💡 추천 순서

1. **즉시**: 방법 1 (포트 29354로 변경) 시도
2. **안 되면**: Stage1 재시작 → multiaddr 로그 확인 → Stage2 실행
3. **여전히 안 되면**: 내부 IP 사용 시도
4. **그래도 안 되면**: 네트워크 연결 테스트 (nc 명령어)

## 📋 체크리스트

- [ ] Stage2의 포트가 Stage1의 `public_dht_port`와 일치하는지 확인
- [ ] PeerID가 DHT PeerID인지 확인 (P2P PeerID 아님)
- [ ] 네트워크 연결 테스트 (nc 명령어)
- [ ] 방화벽 설정 확인 (Elice Cloud 보안 그룹)

