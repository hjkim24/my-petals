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