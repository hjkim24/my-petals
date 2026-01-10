#!/bin/bash
# 엘리스 클라우드 Load Balancing 테스트 스크립트

# 공통 설정
MODEL="meta-llama/Llama-3.1-8B"
PUBLIC_IP="119.59.0.14"
TOTAL_BLOCKS=32
NUM_BLOCKS=8

# 첫 번째 서버 (stage-1 인스턴스)
# 실행 전에 DHT peer ID를 로그에서 확인해야 함!
if [ "$1" == "server1" ]; then
    python -m src.main \
        --model "$MODEL" \
        --splits "8,16,24" \
        --stage 1 \
        --use_load_balancing \
        --num_blocks "$NUM_BLOCKS" \
        --total_blocks "$TOTAL_BLOCKS" \
        --dht_port 8002 \
        --rpc_port 8003 \
        --public_ip "$PUBLIC_IP" \
        --public_dht_port 22452 \
        --public_rpc_port 50192 \
        --balance_quality 0.75 \
        --mean_balance_check_period 120

# 두 번째 서버 (stage-2 인스턴스)
# 사용법: ./elice_test_load_balancing.sh server2 "/ip4/119.59.0.14/tcp/22452/p2p/첫서버PeerID"
elif [ "$1" == "server2" ]; then
    if [ -z "$2" ]; then
        echo "Error: DHT peer ID required"
        echo "Usage: $0 server2 <DHT_PEER_ID>"
        echo "Example: $0 server2 \"/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW...\""
        exit 1
    fi
    
    python -m src.main \
        --model "$MODEL" \
        --splits "8,16,24" \
        --stage 1 \
        --use_load_balancing \
        --num_blocks "$NUM_BLOCKS" \
        --total_blocks "$TOTAL_BLOCKS" \
        --dht_initial_peers "$2" \
        --dht_port 8004 \
        --rpc_port 8005 \
        --public_ip "$PUBLIC_IP" \
        --public_dht_port 29354 \
        --public_rpc_port 15930 \
        --balance_quality 0.75 \
        --mean_balance_check_period 120

# 세 번째 서버 (stage-3 인스턴스)
elif [ "$1" == "server3" ]; then
    if [ -z "$2" ]; then
        echo "Error: DHT peer ID required"
        exit 1
    fi
    
    python -m src.main \
        --model "$MODEL" \
        --splits "8,16,24" \
        --stage 1 \
        --use_load_balancing \
        --num_blocks "$NUM_BLOCKS" \
        --total_blocks "$TOTAL_BLOCKS" \
        --dht_initial_peers "$2" \
        --dht_port 8006 \
        --rpc_port 8007 \
        --public_ip "$PUBLIC_IP" \
        --public_dht_port 59491 \
        --public_rpc_port 38548 \
        --balance_quality 0.75 \
        --mean_balance_check_period 120

# 네 번째 서버 (stage-0 인스턴스)
elif [ "$1" == "server4" ]; then
    if [ -z "$2" ]; then
        echo "Error: DHT peer ID required"
        exit 1
    fi
    
    python -m src.main \
        --model "$MODEL" \
        --splits "8,16,24" \
        --stage 1 \
        --use_load_balancing \
        --num_blocks "$NUM_BLOCKS" \
        --total_blocks "$TOTAL_BLOCKS" \
        --dht_initial_peers "$2" \
        --dht_port 8008 \
        --rpc_port 8009 \
        --public_ip "$PUBLIC_IP" \
        --public_dht_port 41826 \
        --public_rpc_port 23619 \
        --balance_quality 0.75 \
        --mean_balance_check_period 120

# Client (테스트용)
elif [ "$1" == "client" ]; then
    if [ -z "$2" ]; then
        echo "Error: DHT peer IDs required (comma-separated)"
        exit 1
    fi
    
    python -m src.main \
        --model "$MODEL" \
        --splits "8,16,24" \
        --stage 0 \
        --dht_initial_peers "$2" \
        --dht_port 8010 \
        --rpc_port 8011 \
        --public_ip "$PUBLIC_IP" \
        --prompt "Hello, how are you? Tell me about distributed systems."

else
    echo "Usage: $0 {server1|server2|server3|server4|client} [DHT_PEER_ID]"
    echo ""
    echo "Examples:"
    echo "  # First server (no DHT peer needed)"
    echo "  $0 server1"
    echo ""
    echo "  # Other servers (need first server's DHT peer ID)"
    echo "  $0 server2 \"/ip4/119.59.0.14/tcp/22452/p2p/12D3KooW...\""
    echo ""
    echo "  # Client (need all server DHT peer IDs)"
    echo "  $0 client \"/ip4/119.59.0.14/tcp/22452/p2p/ID1,/ip4/119.59.0.14/tcp/29354/p2p/ID2\""
    exit 1
fi


