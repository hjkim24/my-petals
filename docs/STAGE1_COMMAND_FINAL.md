# Stage1 최종 명령어 (한 줄)

## Stage1 - 터널 IP 사용

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip 119.59.0.14 --public_dht_port 22452 --public_rpc_port 50192 --balance_quality 0.75 --mean_balance_check_period 120
```

## Stage1 - 내부 IP 자동 감지

```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --stage 1 --use_load_balancing --num_blocks 8 --total_blocks 32 --dht_port 8002 --rpc_port 8003 --public_ip $(getent hosts $(hostname -f) | awk '{print $1}') --balance_quality 0.75 --mean_balance_check_period 120
```

