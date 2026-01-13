#!/bin/bash
# bitsandbytes CUDA 12.8 설정 수정 스크립트
# RunPod 환경에서 실행

set -e

echo "=========================================="
echo "bitsandbytes CUDA 12.8 설정 수정"
echo "=========================================="

# 1. CUDA 버전 확인
echo "1. CUDA 버전 확인 중..."
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
echo "   PyTorch CUDA 버전: $CUDA_VERSION"

# 2. CUDA 라이브러리 경로 확인
echo "2. CUDA 라이브러리 경로 확인 중..."
CUDA_LIB_PATH="/usr/local/cuda/lib64"
if [ -d "$CUDA_LIB_PATH" ]; then
    echo "   CUDA 라이브러리 경로 발견: $CUDA_LIB_PATH"
    export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
    echo "   LD_LIBRARY_PATH에 추가됨"
else
    echo "   경고: $CUDA_LIB_PATH를 찾을 수 없습니다"
    # 다른 가능한 경로 확인
    POSSIBLE_PATHS=(
        "/usr/local/cuda-12.8/lib64"
        "/usr/local/cuda-12.4/lib64"
        "/usr/local/cuda-12.1/lib64"
        "/usr/lib/x86_64-linux-gnu"
    )
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/libcudart.so" ]; then
            echo "   대체 경로 발견: $path"
            export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
            break
        fi
    done
fi

# 3. 기존 bitsandbytes 제거
echo "3. 기존 bitsandbytes 제거 중..."
pip uninstall bitsandbytes -y 2>/dev/null || true

# 4. 최신 bitsandbytes 설치 (CUDA 12.x 지원)
echo "4. bitsandbytes 재설치 중..."
# 최신 버전 설치 (CUDA 12.x 자동 감지)
pip install bitsandbytes --no-build-isolation --upgrade

# 5. bitsandbytes 테스트
echo "5. bitsandbytes 설치 확인 중..."
python -m bitsandbytes 2>&1 | head -20 || echo "   경고: bitsandbytes 테스트 실패"

# 6. 환경 변수 영구 설정 (선택사항)
echo ""
echo "=========================================="
echo "설정 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 환경 변수를 영구적으로 설정하려면:"
echo "  echo 'export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH' >> ~/.bashrc"
echo ""
echo "bitsandbytes 테스트:"
echo "  python -c 'import bitsandbytes as bnb; print(\"bitsandbytes 설치 성공!\")'"
echo ""

