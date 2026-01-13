#!/bin/bash
# RunPod에서 Stage 1 실행 명령어 (최종 버전)
# 공인 IP: 213.173.105.6
# 내부 포트: 8002 (DHT), 8003 (RPC)
# 외부 포트: 30247 (DHT), 30248 (RPC)

# 방법 1: 직접 실행
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --dht_port 8002 \
    --rpc_port 8003 \
    --public_ip 213.173.105.6 \
    --public_dht_port 30247 \
    --public_rpc_port 30248 \
    --stage 1

# 방법 2: deploy_direct.sh 스크립트 사용 (권장)
# ./scripts/deploy_direct.sh 1 meta-llama/Llama-3.1-8B "8,16,24" "" 213.173.105.6 8002 8003 30247 30248

