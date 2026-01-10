# Stage1 실행 명령어 (Elice Cloud)

## 옵션 1: 터널 IP 사용 (외부 접근 가능, 현재 설정)

터널을 통해 외부에서 접근해야 하는 경우:

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

**특징**:
- 내부에서 `0.0.0.0:8002`, `0.0.0.0:8003`로 리스닝
- DHT에 `119.59.0.14:22452`로 announce
- Stage2는 `--dht_initial_peers`에 `"/ip4/119.59.0.14/tcp/22452/p2p/<PeerID>"` 사용

## 옵션 2: 내부 IP 사용 (같은 네트워크 내, 권장)

Stage2도 같은 내부 네트워크(10.0.2.x)에 있다면:

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
    --public_ip 10.0.2.100 \
    --mean_balance_check_period 120
```

**특징**:
- 내부에서 `0.0.0.0:8002`, `0.0.0.0:8003`로 리스닝
- DHT에 `10.0.2.100:8002`로 announce (public_dht_port 미지정 시 dht_port 사용)
- Stage2는 `--dht_initial_peers`에 `"/ip4/10.0.2.100/tcp/8002/p2p/<PeerID>"` 사용
- 더 빠르고 안정적인 연결 (내부 네트워크)

## 실행 후 확인 사항

Stage1 실행 후 로그에서 다음을 확인:

```
INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): ...
```

이 주소를 Stage2의 `--dht_initial_peers`에 사용하세요.

## Stage2 연결 설정

Stage1이 실행되면 PeerID를 로그에서 확인하고, Stage2 실행 시:

### 터널 IP 사용 시:
```bash
--dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/<Stage1_PeerID>"
```

### 내부 IP 사용 시:
```bash
--dht_initial_peers "/ip4/10.0.2.100/tcp/8002/p2p/<Stage1_PeerID>"
```

