#!/bin/bash
# bitsandbytes 소스에서 컴파일하여 설치 (CUDA 12.8용)
# RunPod 환경에서 실행

set -e

echo "=========================================="
echo "bitsandbytes 소스에서 컴파일 (CUDA 12.8)"
echo "=========================================="

# 1. CUDA 버전 확인
echo "1. CUDA 버전 확인 중..."
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "128")
echo "   PyTorch CUDA 버전: $CUDA_VERSION"

# CUDA 버전을 숫자로 변환 (예: 12.8 -> 128)
if [[ "$CUDA_VERSION" == "12."* ]]; then
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    CUDA_VERSION_INT="${CUDA_MAJOR}${CUDA_MINOR}"
else
    CUDA_VERSION_INT="128"  # 기본값
fi

echo "   컴파일용 CUDA 버전: $CUDA_VERSION_INT"

# 2. CUDA 라이브러리 경로 설정
echo "2. CUDA 라이브러리 경로 설정 중..."
CUDA_LIB_PATH="/usr/local/cuda/lib64"
if [ -d "$CUDA_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
    export CUDA_HOME="/usr/local/cuda"
    echo "   CUDA_HOME: $CUDA_HOME"
    echo "   LD_LIBRARY_PATH에 추가됨"
fi

# 3. 빌드 도구 확인
echo "3. 빌드 도구 확인 중..."
if ! command -v make &> /dev/null; then
    echo "   make 설치 중..."
    apt-get update && apt-get install -y build-essential
fi

# 4. 기존 bitsandbytes 제거
echo "4. 기존 bitsandbytes 제거 중..."
pip uninstall bitsandbytes -y 2>/dev/null || true

# 5. 소스 클론 및 컴파일
echo "5. bitsandbytes 소스에서 컴파일 중..."
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

# CUDA 버전에 맞게 컴파일
echo "   CUDA_VERSION=$CUDA_VERSION_INT로 컴파일 중..."
CUDA_VERSION=$CUDA_VERSION_INT make cuda12x

# 설치
pip install . --no-build-isolation

# 6. 정리
cd /
rm -rf "$TMP_DIR"

# 7. 테스트
echo "6. bitsandbytes 설치 확인 중..."
python -c "import bitsandbytes as bnb; print('bitsandbytes 설치 성공!')" || {
    echo "   경고: 설치 확인 실패"
    python -m bitsandbytes 2>&1 | head -20
}

echo ""
echo "=========================================="
echo "설정 완료!"
echo "=========================================="

