# Bootstrap Peer 연결 실패 해결 가이드

## 🔴 에러 메시지

```
hivemind.p2p.p2p_daemon_bindings.utils.P2PDaemonError: Daemon failed to start: 
failed to connect to bootstrap peers
```

## 🔍 문제 원인

### Stage1 설정
- `--public_ip 119.59.0.14`
- `--public_dht_port 29354` ← **실제 announce 포트**
- 내부 포트: `8002`

### Stage2 설정 (잘못됨)
- `dht_initial_peers`: `/ip4/119.59.0.14/tcp/8002/...` ← **잘못된 포트!**
- Stage1은 포트 `29354`에서 announce하는데, Stage2는 `8002`로 연결 시도

## ✅ 해결 방법

### 방법 1: 올바른 포트 사용 (터널 IP)

Stage2의 `dht_initial_peers`에서 포트를 `29354`로 변경:

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/119.59.0.14/tcp/29354/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" --public_ip 119.59.0.14 --public_dht_port 29354 --public_rpc_port 15930 --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

**변경 사항**: `tcp/8002` → `tcp/29354` (Stage1의 `public_dht_port`)

### 방법 2: 내부 IP 사용 (권장, 같은 네트워크인 경우)

Elice Cloud 인스턴스들이 같은 내부 네트워크에 있다면 내부 IP를 사용하는 것이 더 안정적입니다.

#### Step 1: Stage1 재시작 (내부 IP 사용)

Stage1 인스턴스에서:
```bash
# Stage1 내부 IP 확인
getent hosts $(hostname -f)
# 결과 예시: 10.0.2.100

# Stage1 재시작 (내부 IP 사용)
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --mean_balance_check_period 120
```

Stage1 로그에서 확인:
```
INFO:__main__:P2P initialized for Load Balancing server, PeerID: 12D3KooW...
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): ...
```

#### Step 2: Stage2 시작 (내부 IP 사용)

Stage2 인스턴스에서:
```bash
# Stage1의 내부 IP가 10.0.2.100이라고 가정
# Stage1의 PeerID는 로그에서 확인한 실제 값 사용

python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

**중요**: 
- IP: `10.0.2.100` (Stage1의 내부 IP)
- 포트: `8002` (Stage1의 내부 DHT 포트)
- PeerID: Stage1 로그에서 확인한 실제 값

### 방법 3: Stage1의 실제 Multiaddr 확인

Stage1 실행 후 로그에서 다음을 확인:

```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): ['/ip4/119.59.0.14/tcp/29354/p2p/12D3KooW...']
```

이 multiaddr을 그대로 Stage2의 `--dht_initial_peers`에 사용하세요.

## 🔧 추가 문제 해결

### 방화벽 확인

Stage2에서 Stage1으로 연결 테스트:

```bash
# 터널 IP 사용 시
nc -zv 119.59.0.14 29354

# 내부 IP 사용 시
nc -zv 10.0.2.100 8002
```

연결이 안 되면 방화벽 설정 확인 필요.

### PeerID 확인

Stage1의 실제 PeerID는 로그에서 확인해야 합니다. 이전 실행의 PeerID와 다를 수 있습니다.

### 네트워크 연결 확인

Elice Cloud에서 인스턴스 간 직접 통신이 가능한지 확인:

```bash
# Stage2에서 Stage1으로 ping
ping -c 3 10.0.2.100  # 내부 IP
# 또는
ping -c 3 119.59.0.14  # 터널 IP
```

## 📋 체크리스트

- [ ] Stage1의 `public_dht_port` 확인 (로그 또는 실행 옵션)
- [ ] Stage2의 `dht_initial_peers` 포트가 Stage1의 `public_dht_port`와 일치하는지 확인
- [ ] PeerID가 Stage1 로그의 실제 값과 일치하는지 확인
- [ ] IP 주소가 올바른지 확인 (내부 IP 또는 터널 IP)
- [ ] 네트워크 연결 테스트 (nc, ping)
- [ ] 방화벽 설정 확인

## 💡 권장 설정

**같은 네트워크**: 내부 IP 사용 (더 빠르고 안정적)
**다른 네트워크**: 터널 IP 사용, 포트는 `public_dht_port` 사용

