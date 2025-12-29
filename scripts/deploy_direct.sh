#!/bin/bash
# deploy_direct.sh - Docker 없이 직접 실행 (이미 PyTorch 설치된 경우)

set -e

STAGE=${1:-1}
MODEL=${2:-"gpt2"}
SPLITS=${3:-"10,20,30"}
DHT_INITIAL_PEERS=${4:-""}
PUBLIC_IP=${5:-""}
DHT_PORT=${6:-$((8000 + ($STAGE - 1) * 2))}
RPC_PORT=${7:-$((8001 + ($STAGE - 1) * 2))}
PROMPT=${8:-"Hello, how are you?"}
MAX_TOKENS=${9:-32}

echo "=========================================="
echo "Stage $STAGE 직접 실행 (Docker 없이)"
echo "=========================================="
echo "모델: $MODEL"
echo "Splits: $SPLITS"
echo "DHT Port: $DHT_PORT"
echo "RPC Port: $RPC_PORT"
echo "Public IP: $PUBLIC_IP"
echo "=========================================="

# Python 가상환경 확인 및 생성 (선택사항)
if [ ! -d "venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
echo "의존성 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt
# hivemind는 소스에서 직접 설치 (플랫폼별 바이너리 문제 방지)
echo "hivemind 설치 중 (소스에서 빌드)..."
pip install --force-reinstall --no-binary=hivemind hivemind

# 기존 프로세스 종료 (같은 stage가 실행 중인 경우)
pkill -f "src.main --stage $STAGE" 2>/dev/null || true
sleep 1

# Stage 실행
echo "Stage $STAGE 실행 중..."

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.." || exit 1

CMD="python -m src.main \
    --model $MODEL \
    --splits $SPLITS \
    --stage $STAGE \
    --dht_port $DHT_PORT \
    --rpc_port $RPC_PORT \
    --dht_initial_peers \"$DHT_INITIAL_PEERS\""

if [ -n "$PUBLIC_IP" ]; then
    CMD="$CMD --public_ip $PUBLIC_IP"
fi

if [ $STAGE -eq 0 ]; then
    CMD="$CMD --prompt \"$PROMPT\" --max_new_tokens $MAX_TOKENS"
fi

# 백그라운드 실행 및 로그 저장
nohup $CMD > stage${STAGE}.log 2>&1 &
PID=$!

echo "Stage $STAGE 실행됨 (PID: $PID)"
echo "로그 확인: tail -f stage${STAGE}.log"
echo "프로세스 종료: kill $PID"

