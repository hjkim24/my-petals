#!/usr/bin/env python3
"""
Fault Tolerance 테스트 스크립트

사용법:
1. 정상적으로 stage1, stage2, stage3를 실행
2. 이 스크립트를 실행하여 클라이언트가 생성 시작
3. 생성 중간에 stage 서버를 종료 (Ctrl+C 또는 kill)
4. 스크립트가 자동으로 복구를 시도하는지 확인
"""

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import run_rank0


def test_fault_tolerance():
    """Fault tolerance 테스트를 위한 메인 함수"""
    parser = argparse.ArgumentParser(description="Test fault tolerance with manual server kill")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--splits", type=str, default="8,16,24", help="Layer splits (e.g., 8,16,24)")
    parser.add_argument("--dht_initial_peers", type=str, default="", help="DHT initial peers")
    parser.add_argument("--dht_port", type=int, default=8000)
    parser.add_argument("--rpc_port", type=int, default=8001)
    parser.add_argument("--public_ip", type=str, default="", help="Public IP address for DHT announcement")
    parser.add_argument("--public_dht_port", type=int, default=None, help="Public DHT port (for port forwarding)")
    parser.add_argument("--public_rpc_port", type=int, default=None, help="Public RPC port (for port forwarding)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--top_k", type=int, default=50)
    
    args = parser.parse_args()
    
    # dtype 변환
    import torch
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]
    
    splits = [int(x) for x in args.splits.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("FAULT TOLERANCE 테스트")
    print("=" * 80)
    print(f"모델: {args.model}")
    print(f"Splits: {splits}")
    print(f"프롬프트: {args.prompt}")
    print(f"최대 토큰 수: {args.max_new_tokens}")
    print(f"DHT Port: {args.dht_port}")
    print(f"RPC Port: {args.rpc_port}")
    if args.public_ip:
        print(f"Public IP: {args.public_ip}")
        if args.public_dht_port:
            print(f"Public DHT Port: {args.public_dht_port}")
        if args.public_rpc_port:
            print(f"Public RPC Port: {args.public_rpc_port}")
    print("\n⚠️  테스트 방법:")
    print("1. 생성이 시작되면 중간에 stage 서버 중 하나를 종료하세요 (Ctrl+C 또는 kill)")
    print("2. 클라이언트가 자동으로 복구를 시도하는지 확인하세요")
    print("3. 복구 후 생성이 계속되는지 확인하세요")
    print("=" * 80)
    print("\n5초 후 시작합니다...")
    time.sleep(5)
    
    try:
        run_rank0(args, device, splits)
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_fault_tolerance()

