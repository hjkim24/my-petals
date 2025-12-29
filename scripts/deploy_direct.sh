#!/bin/bash
# deploy_direct.sh - Docker 없이 직접 실행 (이미 PyTorch 설치된 경우)

set -e

STAGE=${1:-1}
MODEL=${2:-"gpt2"}
SPLITS=${3:-"10,20,30"}
DHT_INITIAL_PEERS=${4:-""}
PUBLIC_IP=${5:-""}
DHT_PORT=${6:-$((8002 + ($STAGE - 1) * 2))}
RPC_PORT=${7:-$((8003 + ($STAGE - 1) * 2))}
PUBLIC_DHT_PORT=${8:-""}  # 외부 DHT 포트 (포트 포워딩용, 예: RunPod)
PUBLIC_RPC_PORT=${9:-""}   # 외부 RPC 포트 (포트 포워딩용, 예: RunPod)
PROMPT=${10:-"Hello, how are you?"}
MAX_TOKENS=${11:-32}

echo "=========================================="
echo "Stage $STAGE 직접 실행 (Docker 없이)"
echo "=========================================="
echo "모델: $MODEL"
echo "Splits: $SPLITS"
echo "DHT Port: $DHT_PORT"
echo "RPC Port: $RPC_PORT"
echo "Public IP: $PUBLIC_IP"
if [ -n "$PUBLIC_DHT_PORT" ]; then
    echo "Public DHT Port: $PUBLIC_DHT_PORT"
fi
if [ -n "$PUBLIC_RPC_PORT" ]; then
    echo "Public RPC Port: $PUBLIC_RPC_PORT"
fi
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
if pip show hivemind > /dev/null 2>&1; then
    echo "hivemind가 이미 설치되어 있습니다. 설치 단계를 건너뜁니다."
else
    echo "hivemind 설치 중 (소스에서 빌드)..."
    pip install --force-reinstall --no-binary=hivemind hivemind
fi

# 기존 프로세스 종료 (같은 stage가 실행 중인 경우)
pkill -f "src.main --stage $STAGE" 2>/dev/null || true
sleep 1

# Stage 실행
echo "Stage $STAGE 실행 중..."

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.." || exit 1

# 배열을 사용하여 명령어 구성 (따옴표/공백 문제 해결)
CMD_ARGS=(
    "python" "-m" "src.main"
    "--model" "$MODEL"
    "--splits" "$SPLITS"
    "--stage" "$STAGE"
    "--dht_port" "$DHT_PORT"
    "--rpc_port" "$RPC_PORT"
    "--dht_initial_peers" "$DHT_INITIAL_PEERS"
)

if [ -n "$PUBLIC_IP" ]; then
    CMD_ARGS+=("--public_ip" "$PUBLIC_IP")
fi

if [ -n "$PUBLIC_DHT_PORT" ]; then
    CMD_ARGS+=("--public_dht_port" "$PUBLIC_DHT_PORT")
fi

if [ -n "$PUBLIC_RPC_PORT" ]; then
    CMD_ARGS+=("--public_rpc_port" "$PUBLIC_RPC_PORT")
fi

if [ $STAGE -eq 0 ]; then
    CMD_ARGS+=("--prompt" "$PROMPT" "--max_new_tokens" "$MAX_TOKENS")
fi

# 백그라운드 실행 및 로그 저장
nohup "${CMD_ARGS[@]}" > stage${STAGE}.log 2>&1 &
PID=$!

echo "Stage $STAGE 실행됨 (PID: $PID)"
echo "로그 확인: tail -f stage${STAGE}.log"
echo "프로세스 종료: kill $PID"

