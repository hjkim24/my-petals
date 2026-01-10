# 논문식 Full Load Balancing 사용 가이드

## 개요

논문 "Distributed Inference and Fine-tuning of Large Language Models Over The Internet"의 **Full Load Balancing** 알고리즘이 구현되었습니다.

## 구현된 기능

### ✅ 핵심 알고리즘 (논문 Appendix D)

1. **규칙 1: 새 서버 조인 시 최적 블록 선택**
   - `choose_best_blocks()`: 처리량 기반으로 최적 블록 구간 선택
   - Min-Max 알고리즘 사용 (병목 최소화)

2. **규칙 2: 주기적 재조정**
   - `should_choose_other_blocks()`: 주기적으로 블록 재조정 필요 여부 확인
   - 효율성 임계값(`balance_quality`) 사용하여 불필요한 재조정 방지

3. **처리량 기반 할당**
   - `compute_throughputs()`: 각 블록별 누적 처리량 계산
   - 여러 서버가 같은 블록을 담당할 경우 처리량 합산

4. **처리량 측정**
   - 컴퓨팅 처리량: Forward pass 성능 측정
   - 네트워크 처리량: 대역폭 기반 추정
   - 최종 처리량 = min(컴퓨팅, 네트워크)

## 사용 방법

### 기본 사용 (Load Balancing 활성화)

```bash
python -m src.main \
    --model meta-llama/Llama-2-7b-hf \
    --splits "10,20,30" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 4 \
    --total_blocks 32 \
    --dht_port 8000 \
    --rpc_port 8001
```

### 주요 옵션

- `--use_load_balancing`: Load Balancing 활성화 (필수)
- `--num_blocks`: 서버가 담당할 블록 개수 (기본값: 자동)
- `--total_blocks`: 모델의 전체 블록 개수 (기본값: 자동 탐지)
- `--balance_quality`: 품질 임계값 (기본값: 0.75 = 25% 이상 개선 시 재조정)
- `--mean_balance_check_period`: 재조정 체크 주기 초 (기본값: 120초)
- `--network_bandwidth_mbps`: 네트워크 대역폭 (기본값: 자동 추정)

### 고급 옵션

```bash
python -m src.main \
    --model meta-llama/Llama-2-7b-hf \
    --splits "10,20,30" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 6 \
    --total_blocks 32 \
    --balance_quality 0.99 \  # 1% 개선 시 재조정 (논문과 유사)
    --mean_balance_check_period 60 \  # 60초마다 체크 (논문과 동일)
    --network_bandwidth_mbps 1000 \  # 1 Gbps
    --dht_port 8000 \
    --rpc_port 8001
```

## 작동 방식

### 1. 서버 조인 시 (규칙 1)

1. DHT에서 현재 시스템의 모든 서버 정보 조회
2. 각 블록별 누적 처리량 계산
3. 연속된 `num_blocks` 길이의 구간 중, **병목이 가장 작은 구간** 선택
4. 선택된 블록 로드 및 DHT에 등록

### 2. 주기적 재조정 (규칙 2)

1. 평균 `mean_balance_check_period` 마다 (랜덤 지연 포함) 체크
2. 현재 서버의 처리량 재측정 및 DHT 업데이트
3. 자신의 블록을 제거한 상태에서 최적 위치 시뮬레이션
4. 반복적 최적화 수행 (다른 서버들도 함께 최적화)
5. 개선 여부 계산: `actual_quality = initial_throughput / new_throughput`
6. `actual_quality < balance_quality`이면 재조정 (블록 변경)

### 3. 재조정 시

- 현재 서버 종료
- 새로운 최적 블록 선택
- 새 블록 로드 및 다시 시작

## 구현 파일

### 핵심 모듈

- `src/load_balancing.py`: Load Balancing 알고리즘 (choose_best_blocks, should_choose_other_blocks)
- `src/throughput_measurement.py`: 처리량 측정 (컴퓨팅/네트워크)
- `src/dht_utils.py`: DHT 통합 (서버 정보 저장/조회)

### 통합

- `src/main.py`: 기존 코드와 통합
  - `run_stage_server_with_load_balancing()`: Load Balancing 버전
  - `_setup_and_run_server_with_rebalancing()`: 재조정 로직 포함

## 논문과의 비교

| 기능 | 논문 | 구현 | 상태 |
|------|------|------|------|
| 규칙 1: 새 서버 조인 | ✅ | ✅ | 완전 구현 |
| 규칙 2: 주기적 재조정 | ✅ | ✅ | 완전 구현 |
| 처리량 계산 | ✅ | ✅ | 완전 구현 |
| 처리량 측정 | ✅ | ✅ | 완전 구현 |
| 반복적 최적화 | ✅ | ✅ | 완전 구현 |
| 파이프라인 분리 방지 | ✅ | ✅ | 완전 구현 |
| 재조정 주기 | 60초 | 120초 (조정 가능) | ⚠️ 기본값 다름 |
| 효율성 임계값 | 1% | 25% (조정 가능) | ⚠️ 기본값 다름 |

## 주의사항

1. **의존성**: `numpy`가 필요합니다 (requirements.txt에 추가됨)
2. **성능**: 처리량 측정은 초기 실행 시 시간이 걸릴 수 있습니다
3. **DHT 연결**: 여러 서버 간 DHT 네트워크가 연결되어 있어야 합니다
4. **모델 정보**: `total_blocks`는 자동 탐지되지만, 명시적으로 지정하는 것을 권장합니다

## 예제 시나리오

### 시나리오 1: 여러 서버가 조인하는 경우

```bash
# 서버 1 (첫 번째 서버)
python -m src.main --model llama-2-7b --splits "10,20,30" --stage 1 \
    --use_load_balancing --num_blocks 4 --total_blocks 32

# 서버 2 (두 번째 서버 조인)
python -m src.main --model llama-2-7b --splits "10,20,30" --stage 1 \
    --use_load_balancing --num_blocks 4 --total_blocks 32 \
    --dht_initial_peers "/ip4/서버1IP/tcp/8000/p2p/PeerID1"

# 서버 3 (세 번째 서버 조인)
python -m src.main --model llama-2-7b --splits "10,20,30" --stage 1 \
    --use_load_balancing --num_blocks 4 --total_blocks 32 \
    --dht_initial_peers "/ip4/서버1IP/tcp/8000/p2p/PeerID1"
```

각 서버는 자동으로 최적의 블록 구간을 선택합니다.

### 시나리오 2: 서버 이탈 후 재조정

1. 여러 서버가 실행 중
2. 한 서버가 종료
3. 남은 서버들이 주기적 체크 시 불균형 감지
4. 자동으로 블록 재조정하여 파이프라인 복구

## 성능 기대값

논문의 실험 결과 (Section E):
- **Full load balancing**: Upper bound의 **85-90%** 처리량 유지
- **No load balancing**: 처리량 거의 0 (파이프라인 형성 실패)
- **Balancing new servers only**: 조인 시에만 좋은 성능, 이탈 시 병목 발생

현재 구현은 Full load balancing이므로, 논문과 유사한 성능을 기대할 수 있습니다.

