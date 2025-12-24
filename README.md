# My-PETALS: 분산 추론 시스템

PETALS 스타일의 분산 대규모 언어 모델 추론 시스템입니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 4-스테이지 파이프라인 실행 예시 (로컬, 포트만 다르게)

스플릿 예: `--splits 10,20,30` (Stage0: 0~10, Stage1: 10~20, Stage2: 20~30, Stage3: 30~끝)

1) Stage1 (부트스트랩 DHT + 서버)
```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --splits 10,20,30 \
  --stage 1 \
  --dht_port 8000 \
  --rpc_port 8001
```
로그에 나온 `DHT visible multiaddrs`를 Stage2/3/0에 전달.

2) Stage2 (서버)
```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --splits 10,20,30 \
  --stage 2 \
  --dht_port 8002 \
  --rpc_port 8003 \
  --dht_initial_peers /ip4/<host>/tcp/8000/p2p/<DHT_PEER_FROM_STAGE1>
```

3) Stage3 (마지막 서버)
```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --splits 10,20,30 \
  --stage 3 \
  --dht_port 8004 \
  --rpc_port 8005 \
  --dht_initial_peers /ip4/<host>/tcp/8000/p2p/<DHT_PEER_FROM_STAGE1>
```

4) Stage0 (클라이언트)
```bash
python mini_petals_stage1.py \
  --model gpt2 \
  --splits 10,20,30 \
  --stage 0 \
  --dht_port 8006 \
  --rpc_port 8007 \
  --dht_initial_peers /ip4/<host>/tcp/8000/p2p/<DHT_PEER_FROM_STAGE1> \
  --prompt "Hello, how are you?" \
  --max_new_tokens 32
```

### 인자 요약
- `--model`: HF 모델 이름
- `--splits`: 3개의 증가하는 정수, 예: `10,20,30` (4 스테이지용)
- `--stage`: 0=클라이언트, 1/2=중간, 3=최종 서버
- `--dht_initial_peers`: 부트스트랩 DHT 멀티어드레스(최소 1개)
- `--dht_port`, `--rpc_port`: 각 노드의 포트
- `--prompt`, `--max_new_tokens`, `--request_timeout`: 생성 관련 옵션

## 테스트 스크립트
```bash
./test.sh [모델명] [splits] [max_tokens]
```
예: `./test.sh gpt2 10,20,30 32`

## 구조
- `partition.py`: 모델을 4개 스테이지로 슬라이싱
- `rpc_transport.py`: Stage0에서 다단계 RPC 체인
- `rpc_handler.py`: 각 스테이지 RPC 처리 및 (마지막이면 토큰 샘플링)
- `mini_petals_stage1.py`: 실행 스크립트 (stage0~3)
