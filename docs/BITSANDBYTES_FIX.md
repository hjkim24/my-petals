# bitsandbytes CUDA 설정 수정 가이드

RunPod 환경에서 bitsandbytes CUDA 설정 오류를 해결하는 방법입니다.

## 문제 상황

에러 메시지:
```
CUDA SETUP: Required library version not found: libbitsandbytes_cuda128.so
CUDA SETUP: Setup Failed!
RuntimeError: CUDA Setup failed despite GPU being available.
```

## 해결 방법

### 방법 1: 간단한 재설치 (권장 먼저 시도)

```bash
chmod +x scripts/fix_bitsandbytes.sh
./scripts/fix_bitsandbytes.sh
```

이 스크립트는:
1. CUDA 라이브러리 경로 확인 및 설정
2. 기존 bitsandbytes 제거
3. 최신 버전 재설치
4. 설치 확인

### 방법 2: 소스에서 컴파일 (방법 1이 실패한 경우)

```bash
chmod +x scripts/fix_bitsandbytes_from_source.sh
./scripts/fix_bitsandbytes_from_source.sh
```

이 스크립트는:
1. CUDA 12.8용으로 소스에서 컴파일
2. 환경 변수 자동 설정
3. 설치 확인

### 방법 3: 수동 설정

#### 1. CUDA 라이브러리 경로 확인
```bash
# CUDA 라이브러리 위치 확인
find /usr/local -name "libcudart.so*" 2>/dev/null
ls -la /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null
```

#### 2. 환경 변수 설정
```bash
# 현재 세션에만 적용
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# 영구적으로 설정 (선택사항)
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```

#### 3. bitsandbytes 재설치
```bash
# 기존 제거
pip uninstall bitsandbytes -y

# 최신 버전 설치
pip install bitsandbytes --no-build-isolation --upgrade

# 또는 소스에서 설치
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=128 make cuda12x
pip install . --no-build-isolation
```

#### 4. 설치 확인
```bash
# 간단한 테스트
python -c "import bitsandbytes as bnb; print('bitsandbytes 설치 성공!')"

# 상세 정보 확인
python -m bitsandbytes
```

## 설치 확인 후 모델 실행

bitsandbytes가 정상적으로 설치되면 양자화를 사용할 수 있습니다:

```bash
# 4-bit 양자화 사용
python -m src.main --model Qwen/Qwen2.5-72B-Instruct --splits "20,40,60" --dtype int4 --stage 1 ...

# 8-bit 양자화 사용
python -m src.main --model Qwen/Qwen2.5-72B-Instruct --splits "20,40,60" --dtype int8 --stage 1 ...
```

## 문제 해결

### 여전히 에러가 발생하는 경우

1. **CUDA 버전 확인**
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version
   ```

2. **PyTorch와 CUDA 호환성 확인**
   - PyTorch 2.4.x는 CUDA 12.1+ 지원
   - CUDA 12.8을 사용 중이라면 PyTorch 2.4.1 이상 필요

3. **양자화 없이 실행 (임시 해결책)**
   ```bash
   # 양자화 없이 실행 (bitsandbytes 불필요)
   python -m src.main --model Qwen/Qwen2.5-72B-Instruct --splits "20,40,60" --dtype bf16 --stage 1 ...
   ```

## 참고 자료

- [bitsandbytes GitHub](https://github.com/TimDettmers/bitsandbytes)
- [bitsandbytes CUDA 설정 가이드](https://github.com/TimDettmers/bitsandbytes/blob/main/how_to_use_nonpytorch_cuda.md)

