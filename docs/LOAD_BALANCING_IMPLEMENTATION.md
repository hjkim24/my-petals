# Load-Balancing 구현 범위 분석

이 문서는 Petals 논문에서 제시한 Load-Balancing 알고리즘과 현재 코드베이스의 구현 범위를 비교 분석합니다.

## 📚 논문에서 제시한 Load-Balancing

### 1. Section E: Load Balancing 실험 평가

논문에서는 4가지 접근법을 평가했습니다:

1. **No load balancing** - 랜덤 연속 간격으로 모델 블록 로드
2. **Balancing new servers only** - 새 서버 조인 시에만 최적 블록 선택 (Appendix D의 규칙 1 사용)
3. **Full load balancing** - 매분마다 각 서버가 블록 교체 필요 여부 확인
4. **Upper bound** - 매분마다 최적으로 블록 재할당 (이론적 최대값, 실제 구현 불가)

**결과**: Full load balancing이 Upper bound의 85-90% 수준의 처리량을 유지하며, 파이프라인 연결성을 유지합니다.

### 2. Appendix D: Load Balancing 알고리즘

- **규칙 1**: 새 서버가 조인할 때 최적 블록 선택
- **규칙 2**: 주기적으로 기존 서버들이 블록 재조정 여부 확인
- **Efficiency threshold `p`**: 블록 교체를 피하기 위한 효율성 임계값 (논문에서 `p = 1%` 사용)

## ✅ 현재 구현된 기능

### 1. Core Load-Balancing 알고리즘

#### ✅ `choose_best_blocks()` - 새 서버 조인 시 최적 블록 선택

**위치**: `petals/server/block_selection.py:28-33`

```python
def choose_best_blocks(num_blocks: int, module_infos: List[RemoteModuleInfo]) -> List[int]:
    spans = compute_spans(module_infos, min_state=ServerState.JOINING)
    throughputs = compute_throughputs(spans, total_blocks=len(module_infos))
    
    start = _choose_best_start(throughputs, num_blocks)
    return list(range(start, start + num_blocks))
```

**구현 상태**: ✅ **완전 구현됨**
- 논문의 **"Balancing new servers only"**와 **"Full load balancing"** 모두에서 사용
- Appendix D의 **규칙 1** 구현

#### ✅ `should_choose_other_blocks()` - 동적 재조정 판단

**위치**: `petals/server/block_selection.py:40-95`

**구현 상태**: ✅ **완전 구현됨**
- 논문의 **"Full load balancing"** 구현
- Appendix D의 **규칙 2** 구현

**주요 기능**:
1. 현재 시스템의 최소 처리량(`initial_throughput`) 계산
2. 자신의 블록을 제거한 상태에서 최적 위치 찾기
3. 반복적으로 다른 서버들도 최적화 (70-86줄)
4. 새로운 최소 처리량(`new_throughput`) 계산
5. `balance_quality` 임계값과 비교하여 재조정 여부 결정

**핵심 로직**:
```python
actual_quality = initial_throughput / new_throughput
return actual_quality < balance_quality - eps
```

#### ✅ `compute_throughputs()` - 처리량 계산

**위치**: `petals/server/block_selection.py:12-20`

**구현 상태**: ✅ **완전 구현됨**
- 각 블록별 누적 처리량 계산
- 여러 서버가 같은 블록을 담당할 경우 처리량 합산

### 2. Throughput 측정

#### ✅ `get_server_throughput()` - 서버 처리량 측정

**위치**: `petals/server/throughput.py:37-108`

**구현 상태**: ✅ **완전 구현됨**
- 컴퓨팅 처리량 (`forward_rps`, `inference_rps`) 측정
- 네트워크 처리량 (`network_rps`) 측정
- Relay 패널티 고려 (`relay_penalty`)
- 최종 처리량 = min(compute_throughput, network_throughput)

**주요 특징**:
- 평균 블록 사용 수 고려: `E[Uniform{1, 2, ..., num_blocks}] = (num_blocks + 1) / 2`
- 네트워크 대역폭을 실제로 측정 (speedtest 사용)
- 캐싱을 통한 재측정 최소화

### 3. 주기적 재조정 (Full Load Balancing)

#### ✅ `Server.run()` - 주기적 체크 루프

**위치**: `petals/server/server.py:328-384`

**구현 상태**: ✅ **구현됨** (단, 주기 차이)

**구현 내용**:
```python
while True:
    timeout = random.random() * 2 * self.mean_balance_check_period
    if self.stop.wait(timeout):
        return
    
    if self._should_choose_other_blocks():
        logger.info("Swarm is imbalanced, server will load other blocks")
        break  # Stop serving this set of modules
```

**주기 비교**:
- 논문: **매 60초 (1분)**마다 체크
- 현재 코드: **평균 120초 (2분)**마다 체크 (`mean_balance_check_period=120`)
- 실제 대기 시간: `random.random() * 2 * 120` = **0~240초 랜덤**

**설정 가능**: `--mean_balance_check_period` 파라미터로 조정 가능

### 4. Efficiency Threshold (`balance_quality`)

#### ✅ `balance_quality` 파라미터

**위치**: `petals/server/server.py:84, 268, 418`

**구현 상태**: ✅ **완전 구현됨**

**설정값 비교**:
- 논문의 `p`: **1%** (0.01) - "더 낮은 임계값이 더 자주 재조정하지만 더 나은 성능"
- 현재 코드 기본값: **0.75** (75%)
- 논문과 다른 이유: 논문의 `p`는 처리량 개선 비율 임계값이지만, 코드의 `balance_quality`는 품질 임계값

**로직**:
```python
actual_quality = initial_throughput / new_throughput
# actual_quality < 0.75 이면 재조정 (즉, 25% 이상 개선 가능 시 재조정)
return actual_quality < balance_quality - eps
```

**설정 가능**: `--balance_quality` 파라미터로 조정 가능 (기본값: 0.75)

### 5. 블록 선택 최적화

#### ✅ `_choose_best_start()` - 최적 시작 블록 찾기

**위치**: `petals/server/block_selection.py:23-25`

**구현 상태**: ✅ **완전 구현됨**

**알고리즘**:
- 연속된 `num_blocks` 길이의 모든 구간을 검사
- 각 구간의 처리량을 정렬하여 최소값을 구함
- 최소값이 가장 큰 구간 선택 (즉, 병목이 가장 작은 구간)

```python
options = ((sorted(throughputs[i : i + num_blocks]), i) for i in range(0, len(throughputs) - num_blocks + 1))
return min(options)[-1]  # 최소값 중 최대값 (min-max 알고리즘)
```

## 📊 구현 범위 요약

| 기능 | 논문 | 현재 구현 | 상태 |
|------|------|-----------|------|
| **Balancing new servers only** | Appendix D 규칙 1 | `choose_best_blocks()` | ✅ 완전 구현 |
| **Full load balancing** | Appendix D 규칙 2 | `should_choose_other_blocks()` | ✅ 완전 구현 |
| **Throughput 계산** | Section 3.2 | `compute_throughputs()` | ✅ 완전 구현 |
| **Throughput 측정** | Section 3.1 | `get_server_throughput()` | ✅ 완전 구현 |
| **주기적 재조정** | Section E (매 60초) | `Server.run()` 루프 | ⚠️ 구현됨 (주기 다름: 120초) |
| **Efficiency threshold** | Appendix D (`p = 1%`) | `balance_quality = 0.75` | ⚠️ 구현됨 (임계값 다름) |
| **Network throughput 측정** | Section 3.1 | `measure_network_rps()` | ✅ 완전 구현 |
| **Compute throughput 측정** | Section 3.1 | `measure_compute_rps()` | ✅ 완전 구현 |
| **Disjoint 체크** | Appendix D | `throughputs.min() <= 0` 체크 | ✅ 완전 구현 |
| **Iterative optimization** | Appendix D | `while moved:` 루프 | ✅ 완전 구현 |

## 🎯 핵심 구현 여부

### ✅ 완전히 구현된 기능

1. **새 서버 조인 시 최적 블록 선택** (`choose_best_blocks`)
2. **동적 재조정 알고리즘** (`should_choose_other_blocks`)
3. **처리량 기반 블록 할당** (`compute_throughputs`, `_choose_best_start`)
4. **네트워크 및 컴퓨팅 처리량 측정** (`get_server_throughput`)
5. **파이프라인 분리 방지** (disjoint 체크)
6. **반복적 최적화** (iterative optimization)

### ⚠️ 구현되었으나 설정이 다른 기능

1. **재조정 주기**
   - 논문: 60초
   - 코드: 120초 (조정 가능)

2. **효율성 임계값**
   - 논문: `p = 1%` (개선 비율)
   - 코드: `balance_quality = 0.75` (품질 임계값)

**참고**: 이 차이점들은 파라미터 조정으로 논문 설정과 동일하게 만들 수 있습니다:
- `--mean_balance_check_period 60` (논문과 동일)
- `--balance_quality 0.99` (1% 개선 시 재조정, 논문과 유사)

## 🔍 추가 구현 사항

### 논문에는 없지만 구현된 기능

1. **Race condition 방지**: `mean_block_selection_delay`로 여러 서버 동시 선택 시 지연
2. **Floating point 오류 방지**: `eps = 1e-3` 사용
3. **Relay 패널티**: `relay_penalty = 0.2`로 릴레이를 통한 연결 시 처리량 조정
4. **캐싱**: 처리량 측정 결과를 캐시하여 재측정 최소화
5. **Tensor parallelism 지원**: 다중 GPU에 걸친 블록 분산

## 📈 성능 비교 (예상)

논문의 실험 결과 (Section E, Figure 2):
- **No load balancing**: 처리량 거의 0 (파이프라인 형성 실패)
- **Balancing new servers only**: 서버 조인 시에만 좋은 성능
- **Full load balancing**: Upper bound의 85-90% 처리량 유지
- **Upper bound**: 이론적 최대값

현재 구현은 **Full load balancing**에 해당하므로, 논문과 유사한 성능을 기대할 수 있습니다.

## 🎓 결론

현재 Load-Balancing 구현은 **논문의 핵심 알고리즘을 완전히 구현**했습니다:

1. ✅ **Appendix D의 알고리즘**: 완전 구현
2. ✅ **Section E의 Full load balancing**: 완전 구현
3. ⚠️ **주기 및 임계값**: 구현되었으나 기본값이 다름 (파라미터 조정 가능)

**전체 구현도: 약 95%**

주요 차이점은 하이퍼파라미터 설정이며, 코드 구조와 알고리즘 로직은 논문과 일치합니다.
