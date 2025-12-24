#!/bin/bash

# 테스트 스크립트: 분산 추론 시스템 테스트
# 사용법: ./test.sh [모델명] [split_layer]

MODEL=${1:-"gpt2"}  # 기본값: gpt2
SPLIT_LAYER=${2:-6}  # 기본값: 6 (gpt2는 12개 레이어)
MAX_TOKENS=${3:-32}

# 로컬 IP 주소 가져오기
LOCAL_IP=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo "127.0.0.1")

echo "=========================================="
echo "분산 추론 테스트"
echo "=========================================="
echo "모델: $MODEL"
echo "Split Layer: $SPLIT_LAYER"
echo "로컬 IP: $LOCAL_IP"
echo "=========================================="
echo ""
echo "다음 단계를 따라하세요:"
echo ""
echo "1. 터미널 1에서 Stage1 (서버) 실행:"
echo "   python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --split_layer $SPLIT_LAYER \\"
echo "     --stage 1 \\"
echo "     --dht_port 8000 \\"
echo "     --rpc_port 8001 \\"
echo "     --dht_initial_peers \"\""
echo ""
echo "2. 터미널 2에서 Stage0 (클라이언트) 실행:"
echo "   LOCAL_RANK=0 python mini_petals_stage1.py \\"
echo "     --model $MODEL \\"
echo "     --split_layer $SPLIT_LAYER \\"
echo "     --stage 0 \\"
echo "     --dht_port 8002 \\"
echo "     --rpc_port 8003 \\"
echo "     --dht_initial_peers \"$LOCAL_IP:8000\" \\"
echo "     --prompt \"Hello, how are you?\" \\"
echo "     --max_new_tokens $MAX_TOKENS"
echo ""
echo "=========================================="
echo "참고:"
echo "- Stage1을 먼저 실행해야 합니다"
echo "- 같은 머신에서 테스트할 경우 다른 포트를 사용하세요"
echo "- GPU가 2개 이상이면 LOCAL_RANK로 지정 가능합니다"
echo ""
echo "GPU 1개 환경:"
echo "- 옵션 1: 같은 GPU 공유 (작은 모델만 가능)"
echo "  두 터미널 모두 LOCAL_RANK=0 사용"
echo "- 옵션 2: 하나는 GPU, 하나는 CPU"
echo "  Stage1: CUDA_VISIBLE_DEVICES=\"\" 사용"
echo "  Stage0: LOCAL_RANK=0 사용"
echo "- 옵션 3: 둘 다 CPU (느리지만 테스트 가능)"
echo "  두 터미널 모두 CUDA_VISIBLE_DEVICES=\"\" 사용"
echo "=========================================="

