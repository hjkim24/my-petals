# My-PETALS: 분산 추론 시스템

PETALS 스타일의 분산 대규모 언어 모델 추론 시스템입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### GPU 1개 환경에서 테스트

GPU가 1개만 있어도 테스트 가능합니다. 다음 옵션 중 선택하세요:

#### 옵션 1: 같은 GPU 공유 (작은 모델만 가능)

**터미널 1 (Stage1):**
```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001 \
  --dht_initial_peers ""
```

**터미널 2 (Stage0):**
```bash
LOCAL_RANK=0 python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 0 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers "127.0.0.1:8000" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

#### 옵션 2: 하나는 GPU, 하나는 CPU

**터미널 1 (Stage1 - CPU):**
```bash
CUDA_VISIBLE_DEVICES="" python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001 \
  --dht_initial_peers ""
```

**터미널 2 (Stage0 - GPU):**
```bash
LOCAL_RANK=0 python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 0 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers "127.0.0.1:8000" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

#### 옵션 3: 둘 다 CPU (느리지만 테스트 가능)

**터미널 1 (Stage1):**
```bash
CUDA_VISIBLE_DEVICES="" python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001 \
  --dht_initial_peers ""
```

**터미널 2 (Stage0):**
```bash
CUDA_VISIBLE_DEVICES="" python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 0 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers "127.0.0.1:8000" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

### 단일 노드 테스트 (2개 터미널, GPU 2개 이상)

#### 1. Stage1 (서버) 실행 - 터미널 1

```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001 \
  --dht_initial_peers ""
```

#### 2. Stage0 (클라이언트) 실행 - 터미널 2

```bash
LOCAL_RANK=0 python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 0 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers "127.0.0.1:8000" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

### 멀티 노드 테스트

#### Node 1 (Stage1 서버)

```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001 \
  --dht_initial_peers ""
```

#### Node 2 (Stage0 클라이언트)

```bash
LOCAL_RANK=0 python mini_petals_stage1.py \
  --model gpt2 \
  --split_layer 6 \
  --stage 0 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers "NODE1_IP:8000" \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

## 인자 설명

- `--model`: Hugging Face 모델 이름 (예: `gpt2`, `meta-llama/Llama-2-7b-hf`)
- `--split_layer`: 모델을 나눌 레이어 인덱스 (Stage0: 0~k-1, Stage1: k~end)
- `--stage`: 실행할 스테이지 (0: 클라이언트, 1: 서버)
- `--prompt`: 입력 프롬프트 (기본값: "Hello, how are you?")
- `--dht_port`: DHT 포트 번호
- `--rpc_port`: RPC 포트 번호
- `--dht_initial_peers`: 초기 DHT 피어 목록 (콤마로 구분, 예: `"ip1:port1,ip2:port2"`)
- `--max_new_tokens`: 생성할 최대 토큰 수
- `--request_timeout`: RPC 요청 타임아웃 (초)

## 지원 모델

- GPT-2
- LLaMA / LLaMA-2
- Mistral
- Qwen2.5
- GPT-NeoX

## 테스트 스크립트

```bash
./test.sh [모델명] [split_layer] [max_tokens]
```

예시:
```bash
./test.sh gpt2 6 32
```

## 구조

- `partition.py`: 모델을 Stage0과 Stage1로 분할
- `rpc_transport.py`: 클라이언트 측 RPC 통신 (Stage0)
- `rpc_handler.py`: 서버 측 RPC 요청 처리 (Stage1)
- `mini_petals_stage1.py`: 메인 실행 스크립트

