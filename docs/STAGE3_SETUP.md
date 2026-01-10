# Stage3 실행 가이드

## Stage2 정보 확인

Stage2 로그에서 다음 정보 확인:

### 1. Stage2의 IP 주소
- Elice Cloud 인스턴스의 공인 IP 주소 확인
- 또는 Stage2 인스턴스에서: `hostname -I` 또는 `curl ifconfig.me`

### 2. Stage2의 DHT Peer ID
Stage2 로그에서:
```
INFO:src.dht_utils:Registered server 12D3KooWHG6QEACTQvZjTwWhs2xnyhFpf2mxPc27CRCyxhYAzb9V on DHT
```
→ **DHT Peer ID**: `12D3KooWHG6QEACTQvZjTwWhs2xnyhFpf2mxPc27CRCyxhYAzb9V`

### 3. Stage2의 DHT 포트
Stage2 실행 명령에서:
```bash
--dht_port 8004
```
→ **DHT 포트**: `8004`

## Stage3 실행 명령 (수정 필요)

### 옵션 1: Stage2 IP를 알고 있는 경우

```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --stage 1 \
    --use_load_balancing \
    --num_blocks 8 \
    --total_blocks 32 \
    --dht_initial_peers "/ip4/STAGE2_실제IP/tcp/8004/p2p/12D3KooWHG6QEACTQvZjTwWhs2xnyhFpf2mxPc27CRCyxhYAzb9V" \
    --dht_port 8006 \
    --mean_balance_check_period 120
```

**⚠️ `STAGE2_실제IP`를 Stage2 인스턴스의 실제 IP 주소로 바꾸세요!**

### 옵션 2: Stage2에서 IP 확인 방법

Stage2 인스턴스에서:
```bash
# 방법 1: 공인 IP 확인
curl ifconfig.me

# 방법 2: 로컬 IP 확인
hostname -I

# 방법 3: Elice Cloud에서 확인
# 인스턴스 정보에서 Public IP 확인
```

### 옵션 3: Stage1을 initial_peers로 사용 (만약 Stage1이 정상이라면)

Stage1의 정보를 사용할 수도 있습니다:
```bash
--dht_initial_peers "/ip4/STAGE1_IP/tcp/STAGE1_DHT_PORT/p2p/12D3KooWJUaPYC5b83xQ3md7Ln4zdXkLeXWcrxu6JL2sfaHdYxch"
```

## 예상 결과

Stage3 실행 시 다음 중 하나가 발생해야 합니다:

### 성공 시나리오 1: Stage2를 찾아서 다른 블록 선택
```
INFO:__main__:Retrieved X module infos from DHT (attempt 1/3)
INFO:__main__:Found Y unique server(s) in DHT: ['12D3KooWHG6QEACTQvZjTwWhs2xnyhFpf2mxPc27CRCyxhYAzb9V']
INFO:__main__:Load balancing selected blocks: [8, 9, 10, 11, 12, 13, 14, 15]
INFO:__main__:Selected blocks: [8, 9, 10, 11, 12, 13, 14, 15] (start=8, end=16)
```
→ ✅ **완벽! Load Balancing 작동 중!**

### 성공 시나리오 2: Stage2를 찾아서 다른 위치 선택
```
INFO:__main__:Load balancing selected blocks: [16, 17, 18, 19, 20, 21, 22, 23]
```
→ ✅ **좋음! 다른 블록 선택됨**

### 문제 시나리오: Stage2를 찾지 못함
```
INFO:__main__:Retrieved 0 module infos from DHT (attempt 1/3)
INFO:__main__:No existing servers found, selecting first 8 blocks: [0, 1, 2, 3, 4, 5, 6, 7]
```
→ ❌ **IP 주소나 포트 확인 필요**

## 주의사항

1. **IP 주소**: Stage2의 실제 공인 IP 주소 사용 (Elice Cloud에서 확인)
2. **포트**: Stage2의 DHT 포트와 일치해야 함 (8004)
3. **Peer ID**: Stage2의 DHT 등록 Peer ID 사용 (P2P Peer ID 아님)
4. **DHT 포트**: Stage3는 다른 포트 사용 (8006 권장)

## Stage2 IP 확인 명령어 (Stage2 인스턴스에서 실행)

```bash
# 공인 IP 확인
curl ifconfig.me

# 또는
wget -qO- ifconfig.me

# 또는 Elice Cloud 콘솔에서 확인
```


