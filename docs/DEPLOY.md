# 분산 GPU 인스턴스 배포 가이드

## 전제 조건
- 4개의 GPU 인스턴스 (각각 다른 서버/클라우드)
- 각 인스턴스의 공인 IP 주소 확인
- 방화벽에서 DHT/RPC 포트 오픈 (기본: 8000-8007)

**배포 방법 선택:**
- **방법 1 (Docker)**: Docker 및 nvidia-docker 설치 필요
- **방법 2 (직접 실행)**: Python 3.8+ 및 PyTorch 설치 필요 (권장: 이미 PyTorch가 설치된 경우)

---

## 방법 1: Docker를 사용한 배포

### 전제 조건
- Docker 및 nvidia-docker 설치
- 각 인스턴스에 Docker 설치 확인: `docker --version`

### 1. 코드 준비
각 인스턴스에 코드 업로드:

```bash
# Git에서 클론하거나 파일 전송
git clone <your-repo>
cd my-petals
```

### 2. Stage1 배포 (DHT 부트스트랩)
**인스턴스 1**에서 실행:

```bash
chmod +x deploy.sh
./deploy.sh 1 gpt2 "10,20,30" "" <PUBLIC_IP_1> 8000 8001

# DHT multiaddr 확인 (로그에서 추출)
docker logs mini-petals-stage1 | grep "DHT visible multiaddrs"
# 출력 예: INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [<Multiaddr /ip4/1.2.3.4/tcp/8000/p2p/12D3KooW...>]
```

**중요**: Stage1의 DHT multiaddr을 복사하세요. 형식: `/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>`

### 3. Stage2 배포
**인스턴스 2**에서 실행:

```bash
./deploy.sh 2 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_2> 8002 8003
```

### 4. Stage3 배포
**인스턴스 3**에서 실행:

```bash
./deploy.sh 3 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_3> 8004 8005
```

### 5. Stage0 배포 (클라이언트)
**인스턴스 4**에서 실행:

```bash
./deploy.sh 0 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_4> 8006 8007 "Hello, how are you?" 32
```

### 로그 확인 (Docker)
```bash
# 실시간 로그
docker logs -f mini-petals-stage<N>

# 최근 로그
docker logs --tail 100 mini-petals-stage<N>
```

---

## 방법 2: Docker 없이 직접 실행 (권장: 이미 PyTorch 설치된 경우)

### 전제 조건
- Python 3.8 이상
- PyTorch 설치 확인: `python -c "import torch; print(torch.__version__)"`
- CUDA 사용 가능 (GPU 인스턴스)

### 1. 코드 준비
각 인스턴스에 코드 업로드:

```bash
# Git에서 클론하거나 파일 전송
git clone <your-repo>
cd my-petals
```

### 2. Stage1 배포 (DHT 부트스트랩)
**인스턴스 1**에서 실행:

```bash
chmod +x deploy_direct.sh
./deploy_direct.sh 1 gpt2 "10,20,30" "" <PUBLIC_IP_1> 8000 8001

# DHT multiaddr 확인 (로그에서 추출)
tail -f stage1.log | grep "DHT visible multiaddrs"
# 출력 예: INFO:__main__:DHT visible multiaddrs (use for --dht_initial_peers): [<Multiaddr /ip4/1.2.3.4/tcp/8000/p2p/12D3KooW...>]
```

**중요**: Stage1의 DHT multiaddr을 복사하세요. 형식: `/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>`

### 3. Stage2 배포
**인스턴스 2**에서 실행:

```bash
./deploy_direct.sh 2 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_2> 8002 8003
```

### 4. Stage3 배포
**인스턴스 3**에서 실행:

```bash
./deploy_direct.sh 3 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_3> 8004 8005
```

### 5. Stage0 배포 (클라이언트)
**인스턴스 4**에서 실행:

```bash
./deploy_direct.sh 0 gpt2 "10,20,30" "/ip4/<PUBLIC_IP_1>/tcp/8000/p2p/<PEER_ID>" <PUBLIC_IP_4> 8006 8007 "Hello, how are you?" 32
```

### 로그 확인 (직접 실행)
```bash
# 실시간 로그
tail -f stage<N>.log

# 최근 로그
tail -n 100 stage<N>.log
```

### 프로세스 관리 (직접 실행)
```bash
# 실행 중인 프로세스 확인
ps aux | grep "src.main"

# 프로세스 종료
pkill -f "src.main --stage <N>"

# 또는 PID로 종료
kill <PID>
```

---

## 문제 해결

### 포트가 열려있지 않은 경우
각 인스턴스의 방화벽에서 포트 오픈:
- Stage1: 8000 (DHT), 8001 (RPC)
- Stage2: 8002 (DHT), 8003 (RPC)
- Stage3: 8004 (DHT), 8005 (RPC)
- Stage0: 8006 (DHT), 8007 (RPC)

### 공인 IP 확인
```bash
curl ifconfig.me
# 또는
curl ipinfo.io/ip
```

### Docker 컨테이너 재시작
```bash
docker restart mini-petals-stage<N>
```

### Docker 컨테이너 중지
```bash
docker stop mini-petals-stage<N>
docker rm mini-petals-stage<N>
```

---

## 빠른 참조

### deploy.sh 사용법 (Docker)
```bash
./deploy.sh <STAGE> <MODEL> <SPLITS> <DHT_INITIAL_PEERS> <PUBLIC_IP> <DHT_PORT> <RPC_PORT> [PROMPT] [MAX_TOKENS]
```

**예시:**
- Stage1: `./deploy.sh 1 gpt2 "10,20,30" "" 1.2.3.4 8000 8001`
- Stage2: `./deploy.sh 2 gpt2 "10,20,30" "/ip4/1.2.3.4/tcp/8000/p2p/12D3KooW..." 5.6.7.8 8002 8003`
- Stage0: `./deploy.sh 0 gpt2 "10,20,30" "/ip4/1.2.3.4/tcp/8000/p2p/12D3KooW..." 9.10.11.12 8006 8007 "Hello" 32`

### deploy_direct.sh 사용법 (직접 실행)
```bash
./deploy_direct.sh <STAGE> <MODEL> <SPLITS> <DHT_INITIAL_PEERS> <PUBLIC_IP> <DHT_PORT> <RPC_PORT> [PROMPT] [MAX_TOKENS]
```

**예시:**
- Stage1: `./deploy_direct.sh 1 gpt2 "10,20,30" "" 1.2.3.4 8000 8001`
- Stage2: `./deploy_direct.sh 2 gpt2 "10,20,30" "/ip4/1.2.3.4/tcp/8000/p2p/12D3KooW..." 5.6.7.8 8002 8003`
- Stage0: `./deploy_direct.sh 0 gpt2 "10,20,30" "/ip4/1.2.3.4/tcp/8000/p2p/12D3KooW..." 9.10.11.12 8006 8007 "Hello" 32`

### DHT Multiaddr 추출

**Docker 사용 시:**
```bash
docker logs mini-petals-stage1 2>&1 | grep -oP '<Multiaddr \K[^>]+' | head -1
```

**직접 실행 시:**
```bash
grep -oP '<Multiaddr \K[^>]+' stage1.log | head -1
```

또는 로그에서 직접 확인:
```bash
# Docker
docker logs mini-petals-stage1 | grep "DHT visible multiaddrs"

# 직접 실행
grep "DHT visible multiaddrs" stage1.log
```

### 네트워크 연결 확인

**Docker 사용 시:**
```bash
# Stage1에서 다른 stage들이 등록되었는지 확인
docker logs mini-petals-stage1 | grep "mini_petals:stage"

# Stage0에서 연결 상태 확인
docker logs mini-petals-stage0 | grep "Connected to"
```

**직접 실행 시:**
```bash
# Stage1에서 다른 stage들이 등록되었는지 확인
grep "mini_petals:stage" stage1.log

# Stage0에서 연결 상태 확인
grep "Connected to" stage0.log
```

---

## 배포 방법 비교

| 항목 | Docker | 직접 실행 |
|------|--------|----------|
| **설정 복잡도** | 중간 (Docker 설치 필요) | 낮음 (Python만 필요) |
| **시작 속도** | 느림 (이미지 빌드) | 빠름 (즉시 실행) |
| **리소스 사용** | 높음 (컨테이너 오버헤드) | 낮음 (네이티브 실행) |
| **환경 격리** | 좋음 | 없음 |
| **디버깅** | 어려움 (컨테이너 내부) | 쉬움 (직접 접근) |
| **권장 상황** | 여러 프로젝트 공존 | 단일 프로젝트, 이미 PyTorch 설치됨 |

**권장**: 이미 PyTorch가 설치된 GPU 인스턴스를 사용하는 경우, **방법 2 (직접 실행)**를 권장합니다.
