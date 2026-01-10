# "No existing servers found" 메시지 설명

## ✅ 이 메시지는 정상입니다!

### 왜 나타나나요?

Stage1이 **첫 번째 서버**로 시작할 때:

1. **DHT 조회**: DHT에서 기존 서버 정보를 찾으려고 시도
2. **결과**: 아직 다른 서버가 없으므로 `0개` 반환
3. **동작**: "No existing servers found" 메시지 출력 후, 첫 번째 블록 `[0-7]` 선택
4. **등록**: 자신의 정보를 DHT에 등록하여 다음 서버(Stage2)가 찾을 수 있도록 함

### 코드 동작 흐름

```python
# 1. DHT에서 기존 서버 조회 (최대 3번 재시도)
module_infos = get_remote_module_infos(dht, args.model, total_blocks)
# 결과: [] (빈 리스트, 다른 서버 없음)

# 2. 서버가 없으면 첫 번째 블록 선택
if len(module_infos) == 0:
    block_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # 첫 8개 블록
    logger.info("No existing servers found, selecting first 8 blocks")
    
# 3. 서버가 있으면 Load Balancing으로 최적 블록 선택
else:
    block_indices = choose_best_blocks(...)  # 논문식 알고리즘
```

### Stage1 로그 해석

```
INFO:src.dht_utils:Retrieved 0 module infos from DHT (total_blocks=32)
INFO:__main__:Retrieved 0 module infos from DHT (attempt 1/3)
INFO:__main__:No existing servers found, waiting 2.0s for DHT propagation...
INFO:src.dht_utils:Retrieved 0 module infos from DHT (total_blocks=32)
INFO:__main__:Retrieved 0 module infos from DHT (attempt 2/3)
INFO:__main__:No existing servers found, waiting 3.0s for DHT propagation...
INFO:src.dht_utils:Retrieved 0 module infos from DHT (total_blocks=32)
INFO:__main__:Retrieved 0 module infos from DHT (attempt 3/3)
INFO:__main__:No existing servers found, selecting first 8 blocks: [0, 1, 2, 3, 4, 5, 6, 7]  ← 정상!
```

**이것은 문제가 아닙니다!** Stage1이 첫 번째 서버이므로 당연히 다른 서버를 찾지 못합니다.

### Stage2 시작 후 예상 로그

Stage2가 시작하면:

```
INFO:src.dht_utils:Retrieved 8 module infos from DHT (total_blocks=32)  ← Stage1 발견!
INFO:src.dht_utils:Block coverage: 0 to 7 (8 blocks found)  ← Stage1의 블록 발견
INFO:__main__:Found 1 unique server(s) in DHT: ['12D3KooWSfbNJ...']  ← Stage1 발견
INFO:__main__:Load balancing selected blocks: [8, 9, 10, 11, 12, 13, 14, 15]  ← 다른 블록 선택!
```

### 정리

| 상황 | DHT 조회 결과 | 동작 | 상태 |
|------|--------------|------|------|
| **Stage1 (첫 서버)** | 0개 | 첫 블록 [0-7] 선택 | ✅ 정상 |
| **Stage2 (두 번째 서버)** | 8개 (Stage1 발견) | Load Balancing으로 다른 블록 선택 | ✅ 정상 |
| **Stage3 (세 번째 서버)** | 16개 (Stage1,2 발견) | Load Balancing으로 또 다른 블록 선택 | ✅ 정상 |

### 문제가 되는 경우

다음 경우에만 문제입니다:

1. **Stage2가 Stage1을 찾지 못함**:
   ```
   INFO:src.dht_utils:Retrieved 0 module infos  ← 계속 0개
   INFO:__main__:No existing servers found  ← Stage1을 못 찾음
   INFO:__main__:Selected blocks: [0, 1, 2, 3, 4, 5, 6, 7]  ← Stage1과 중복!
   ```
   → **해결**: `--dht_initial_peers` 설정 확인, `--public_ip` 확인

2. **Stage1이 계속 0개를 반환하는 경우** (이미 다른 서버가 있는데):
   ```
   # Stage1이 재시작되었는데 Stage2가 이미 실행 중인 경우
   INFO:src.dht_utils:Retrieved 0 module infos  ← 이상함!
   ```
   → **해결**: DHT 연결 확인, `--dht_initial_peers` 설정 확인

### 결론

**Stage1에서 "No existing servers found"는 완전히 정상입니다!**

이 메시지는:
- ✅ "첫 번째 서버로 시작함"을 의미
- ✅ "다른 서버가 없어서 첫 번째 블록을 선택함"을 의미
- ✅ 문제가 아님

Stage2가 시작하면 Stage1을 찾고, Load Balancing이 작동할 것입니다!

