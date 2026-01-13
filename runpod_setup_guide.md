# RunPod에서 설정하는 방법

## 1. RunPod 인스턴스의 공인 IP 확인

RunPod 콘솔에서 Pod를 클릭하면 IP 주소가 표시됩니다. 또는 Pod 내부에서 다음 명령어로 확인:

```bash
curl ifconfig.me
# 또는
curl ipinfo.io/ip
```

## 2. 포트 포워딩 확인

RunPod 콘솔의 Pod 설정에서:
- **TCP ports**: `22,8002,8003` (내부 포트)
- 각 내부 포트에 대해 RunPod가 자동으로 외부 포트를 할당합니다

포트 포워딩 정보는 RunPod 콘솔의 Pod 상세 페이지에서 확인할 수 있습니다.
일반적으로 형식은 다음과 같습니다:
- 내부 포트 8002 → 외부 포트 (예: 10798 또는 자동 할당)
- 내부 포트 8003 → 외부 포트 (예: 10799 또는 자동 할당)

## 3. 수정된 명령어

엘리스 클라우드 명령어:
```bash
python -m src.main --model meta-llama/Llama-3.1-8B --splits "8,16,24" --dht_port 8002 --rpc_port 8003 --public_ip 119.59.0.14 --public_dht_port 45485 --public_rpc_port 42622 --stage 1
```

**RunPod에서 수정된 명령어:**
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --dht_port 8002 \
    --rpc_port 8003 \
    --public_ip <RUNPOD_PUBLIC_IP> \
    --public_dht_port <RUNPOD_EXTERNAL_DHT_PORT> \
    --public_rpc_port <RUNPOD_EXTERNAL_RPC_PORT> \
    --stage 1
```

**예시 (공인 IP가 149.36.1.141이고, 외부 포트가 10798, 10799인 경우):**
```bash
python -m src.main \
    --model meta-llama/Llama-3.1-8B \
    --splits "8,16,24" \
    --dht_port 8002 \
    --rpc_port 8003 \
    --public_ip 149.36.1.141 \
    --public_dht_port 10798 \
    --public_rpc_port 10799 \
    --stage 1
```

## 4. deploy_direct.sh 스크립트 사용 (권장)

더 편리하게 사용하려면 `deploy_direct.sh` 스크립트를 사용하세요:

```bash
./scripts/deploy_direct.sh 1 meta-llama/Llama-3.1-8B "8,16,24" "" <RUNPOD_PUBLIC_IP> 8002 8003 <RUNPOD_EXTERNAL_DHT_PORT> <RUNPOD_EXTERNAL_RPC_PORT>
```

**예시:**
```bash
./scripts/deploy_direct.sh 1 meta-llama/Llama-3.1-8B "8,16,24" "" 149.36.1.141 8002 8003 10798 10799
```

## 5. 포트 포워딩 정보 확인 방법

RunPod 콘솔에서:
1. Pod를 클릭하여 상세 페이지로 이동
2. "Connect" 또는 "Ports" 탭에서 포트 포워딩 정보 확인
3. 또는 Pod 내부에서 `netstat -tuln` 명령어로 리스닝 포트 확인

## 주의사항

- `--dht_port`와 `--rpc_port`는 **내부 포트** (항상 8002, 8003)
- `--public_dht_port`와 `--public_rpc_port`는 **외부 포트** (RunPod가 할당한 포트)
- `--public_ip`는 RunPod 인스턴스의 **공인 IP 주소**
- 다른 인스턴스에서 연결할 때는 **외부 포트**를 사용한 multiaddr을 사용해야 합니다

