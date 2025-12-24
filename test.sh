#!/bin/bash

# 테스트 스크립트: 4-스테이지 파이프라인 테스트
# 사용법: ./test.sh [모델명] [splits] [max_tokens]
# splits 예: 10,20,30  (Stage0:0~10, Stage1:10~20, Stage2:20~30, Stage3:30~end)

MODEL=${1:-"gpt2"}
SPLITS=${2:-"10,20,30"}
MAX_TOKENS=${3:-32}

# 로컬 IP 주소 가져오기
LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo "127.0.0.1")

echo "=========================================="
echo "분산 추론 테스트"
echo "=========================================="
echo "모델: $MODEL"
echo "Splits: $SPLITS"
echo "로컬 IP: $LOCAL_IP"
echo "=========================================="
echo ""
echo "다음 단계를 따라하세요:"
echo ""
echo "1. 터미널 1에서 Stage1 (서버, DHT 부트스트랩) 실행:"
echo "   python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --splits $SPLITS \\"
echo "     --stage 1 \\"
echo "     --dht_port 8000 \\"
echo "     --rpc_port 8001 \\"
echo "     --dht_initial_peers \"\""
echo ""
echo "2. Stage1 로그에 나온 DHT 멀티어드레스(/ip4/.../tcp/8000/p2p/...)를 복사"
echo ""
echo "3. 터미널 2에서 Stage2 (서버) 실행:"
echo "   python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --splits $SPLITS \\"
echo "     --stage 2 \\"
echo "     --dht_port 8002 \\"
echo "     --rpc_port 8003 \\"
echo "     --dht_initial_peers /ip4/$LOCAL_IP/tcp/8000/p2p/<DHT_FROM_STAGE1>"
echo ""
echo "4. 터미널 3에서 Stage3 (마지막 서버) 실행:"
echo "   python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --splits $SPLITS \\"
echo "     --stage 3 \\"
echo "     --dht_port 8004 \\"
echo "     --rpc_port 8005 \\"
echo "     --dht_initial_peers /ip4/$LOCAL_IP/tcp/8000/p2p/<DHT_FROM_STAGE1>"
echo ""
echo "5. 터미널 4에서 Stage0 (클라이언트) 실행:"
echo "   LOCAL_RANK=0 python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --splits $SPLITS \\"
echo "     --stage 0 \\"
echo "     --dht_port 8006 \\"
echo "     --rpc_port 8007 \\"
echo "     --dht_initial_peers /ip4/$LOCAL_IP/tcp/8000/p2p/<DHT_FROM_STAGE1> \\"
echo "     --prompt \"Hello, how are you?\" \\"
echo "     --max_new_tokens $MAX_TOKENS"
echo ""
echo "=========================================="
echo "참고:"
echo "- Stage1을 먼저 실행해 DHT 부트스트랩 멀티어드레스를 확보하세요"
echo "- 각 스테이지는 서로 다른 포트를 사용하세요 (예: DHT 8000/8002/8004/8006, RPC 8001/8003/8005/8007)"
echo "- GPU 배치는 각 터미널에서 CUDA_VISIBLE_DEVICES/LOCAL_RANK로 조정하세요"
echo "=========================================="
