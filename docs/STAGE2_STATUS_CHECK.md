# Stage2 상태 확인 가이드

## 1. 최신 코드 확인

```bash
cd ~/my-petals
git log --oneline -5
# 최신 커밋이 "Fix peer ID matching" 또는 "Add DHT retry logic" 포함해야 함

git status
# "Your branch is up to date with 'origin/Jaewon'" 또는 최신 상태여야 함
```

최신 코드가 아니면:
```bash
git fetch origin
git reset --hard origin/Jaewon
```

## 2. Stage2 로그 확인 포인트

### ✅ 정상 작동 시나리오

**Case A: 다른 서버를 찾아서 다른 블록 선택**
```
INFO:__main__:Retrieved X module infos from DHT (attempt 1/3)
INFO:__main__:Found Y unique server(s) in DHT: ['12D3KooWJUa...']
INFO:__main__:Load balancing selected blocks: [8, 9, 10, 11, 12, 13, 14, 15]
INFO:__main__:Selected blocks: [8, 9, 10, 11, 12, 13, 14, 15] (start=8, end=16)
```
→ **다른 블록 선택됨! Stage3 실행 가능**

**Case B: 다른 서버를 찾지 못해서 첫 번째 블록 선택 (DHT 전파 지연)**
```
INFO:__main__:Retrieved 0 module infos from DHT (attempt 1/3)
INFO:__main__:No existing servers found, selecting first 8 blocks: [0, 1, 2, 3, 4, 5, 6, 7]
INFO:__main__:Selected blocks: [0, 1, 2, 3, 4, 5, 6, 7] (start=0, end=8)
```
→ **재시도 중... 기다려야 함**

### ⚠️ 문제 상황

**Case C: 같은 블록 선택 + WARN 메시지**
```
INFO:__main__:Selected blocks: [0, 1, 2, 3, 4, 5, 6, 7] (start=0, end=8)
WARN:src.load_balancing:Local peer ... not found in spans
```
→ **최신 코드가 아니거나 DHT 조회 실패. 재시작 필요**

## 3. 판단 기준

### ✅ Stage3 실행 가능 조건
- [ ] Stage2가 **다른 블록** 선택 (예: [8-15], [16-23], [24-31])
- [ ] Stage2 로그에 `Registered server ... on DHT` 메시지 있음
- [ ] Stage2가 정상적으로 실행 중 (에러 없음)

### ⏸️ 더 기다려야 하는 조건
- [ ] Stage2가 같은 블록 [0-7] 선택했지만, 재시도 로그가 보임 (`attempt 2/3`, `attempt 3/3`)
- [ ] `Retrieved 0 module infos` 메시지가 계속 나타남
- [ ] Stage2가 방금 시작됨 (DHT 전파 시간 필요)

### 🔄 재시작 필요한 조건
- [ ] 최신 코드가 아님 (git log 확인)
- [ ] `WARN:src.load_balancing:Local peer ... not found in spans` 반복됨
- [ ] 재시도 로그가 없음 (최신 코드 미사용)

## 4. 권장 순서

1. **Stage2 로그 확인** → 위 Case A/B/C 중 어디에 해당하는지 판단
2. **최신 코드 확인** → git status, git log 확인
3. **조치**:
   - Case A → Stage3 실행
   - Case B → 10-20초 더 대기 후 다시 확인
   - Case C → Stage2 재시작 (최신 코드 pull 후)


