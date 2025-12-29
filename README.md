# Mini Petals - Distributed Inference System

분산 GPU 인스턴스에서 대규모 언어 모델을 실행하는 시스템입니다.

## 프로젝트 구조

```
my-petals/
├── src/                    # 핵심 기능 코드
│   ├── __init__.py
│   ├── main.py                # 메인 실행 파일
│   ├── partition.py           # 모델 분할 로직
│   ├── rpc_transport.py       # 클라이언트 RPC 통신
│   └── rpc_handler.py         # 서버 RPC 핸들러
├── scripts/               # 배포 및 실행 스크립트
│   ├── deploy_direct.sh      # 직접 실행 스크립트
│   ├── run_all.py            # 모든 stage 자동 실행
│   ├── auto_pull.sh          # 자동 git pull
│   └── ...
├── docs/                  # 문서
│   ├── README.md            # 이 파일
│   ├── DEPLOY.md            # 배포 가이드
│   ├── PORTS.md             # 포트 설정 가이드
│   └── AUTO_PULL.md         # 자동 pull 설정 가이드
├── requirements.txt        # Python 의존성
├── Dockerfile             # Docker 이미지 빌드
└── .dockerignore          # Docker 빌드 제외 파일
```

## 빠른 시작

### 1. 설치

```bash
git clone <your-repo>
cd my-petals
pip install -r requirements.txt
```

### 2. 실행

자세한 배포 방법은 [docs/DEPLOY.md](docs/DEPLOY.md)를 참고하세요.

**간단한 예시 (단일 머신):**
```bash
python -m src.main \
    --model gpt2 \
    --splits "10,20,30" \
    --stage 1 \
    --dht_port 8000 \
    --rpc_port 8001 \
    --dht_initial_peers ""
```

## 주요 기능

- **파이프라인 병렬화**: 모델을 여러 stage로 분할하여 분산 실행
- **DHT 기반 피어 발견**: Hivemind DHT를 사용한 자동 피어 연결
- **RPC 통신**: 효율적인 원격 프로시저 호출
- **KV Cache 관리**: 효율적인 추론을 위한 캐시 관리

## 문서

- [배포 가이드](docs/DEPLOY.md) - 분산 GPU 인스턴스 배포 방법
- [포트 설정](docs/PORTS.md) - 방화벽 포트 설정 가이드
- [자동 Pull 설정](docs/AUTO_PULL.md) - Git 자동 업데이트 설정

## 라이선스

MIT License

