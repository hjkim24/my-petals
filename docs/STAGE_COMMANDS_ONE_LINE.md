# Stage 실행 명령어 (한 줄 버전)

## Stage1 - 내부 IP 자동 감지 버전 (권장)

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --mean_balance_check_period 120
```

## Stage1 - 터널 IP 사용 버전

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip 119.59.0.14 --public_dht_port 22452 --public_rpc_port 50192 --mean_balance_check_period 120
```

## Stage2 - 내부 IP 사용 버전

**⚠️ 중요**: Stage1 실행 후 로그에서 PeerID 확인 필요!

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/119.59.0.14/tcp/8002/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

## Stage2 - 터널 IP 사용 버전

**⚠️ 중요**: PeerID는 Stage1 로그에서 확인한 실제 값으로 변경!

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_initial_peers "/ip4/119.59.0.14/tcp/22452/p2p/12D3KooWSfbNJ2PTtDZdHfVzjT3ZyXKZ8GopG5zWc4SmFLxiFCyU" --public_ip 119.59.0.14 --public_dht_port 29354 --public_rpc_port 15930 --dht_port 8004 --rpc_port 8005 --mean_balance_check_period 120
```

## 사용 방법

1. 위 명령어를 그대로 복사
2. Elice Cloud 인스턴스 터미널에 붙여넣기
3. Enter 키로 실행

**주의사항**:
- Stage2의 PeerID는 Stage1 실행 후 로그에서 확인한 실제 값으로 변경해야 합니다
- IP 주소도 각 인스턴스의 실제 IP로 변경해야 합니다

