"""
서버 처리량 측정 모듈 (논문 Section 3.1)
네트워크 및 컴퓨팅 처리량을 측정하여 Load Balancing에 사용
"""

import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


def measure_compute_throughput(
    model,
    device: torch.device,
    hidden_size: int = 4096,
    num_blocks: int = 1,
    dtype: torch.dtype = torch.float16,
    warmup_steps: int = 2,
    benchmark_steps: int = 10,
) -> Dict[str, float]:
    """
    컴퓨팅 처리량 측정 (forward pass)
    
    Args:
        model: PyTorch 모델
        device: 디바이스
        hidden_size: Hidden state 크기
        num_blocks: 담당하는 블록 개수
        dtype: 데이터 타입
        warmup_steps: 워밍업 스텝 수
        benchmark_steps: 벤치마크 스텝 수
    
    Returns:
        처리량 측정 결과 딕셔너리
    """
    model.eval()
    
    # 더미 입력 생성
    batch_size = 1
    seq_len = 1  # autoregressive generation 시 seq_len=1
    
    try:
        # Forward pass 처리량 측정
        with torch.inference_mode():
            # 워밍업
            for _ in range(warmup_steps):
                dummy_input = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=device, dtype=dtype
                )
                try:
                    _ = model(dummy_input)
                except Exception as e:
                    logger.warning(f"Forward pass failed during warmup: {e}")
                    return {"forward_rps": 0.0, "inference_rps": 0.0}
            
            # 벤치마크
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            for _ in range(benchmark_steps):
                dummy_input = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=device, dtype=dtype
                )
                try:
                    _ = model(dummy_input)
                except Exception as e:
                    logger.warning(f"Forward pass failed during benchmark: {e}")
                    return {"forward_rps": 0.0, "inference_rps": 0.0}
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start_time
            
            forward_rps = benchmark_steps / elapsed if elapsed > 0 else 0.0
        
        # Inference RPS는 forward RPS와 동일 (autoregressive의 경우)
        inference_rps = forward_rps
        
    except Exception as e:
        logger.error(f"Error measuring compute throughput: {e}")
        forward_rps = 0.0
        inference_rps = 0.0
    
    return {
        "forward_rps": forward_rps,
        "inference_rps": inference_rps,
    }


def estimate_network_throughput(
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.float16,
    bandwidth_mbps: Optional[float] = None,
) -> float:
    """
    네트워크 처리량 추정 (논문 Section 3.1)
    
    Args:
        hidden_size: Hidden state 크기
        dtype: 데이터 타입
        bandwidth_mbps: 네트워크 대역폭 (Mbps), None이면 추정
    
    Returns:
        초당 처리 가능한 요청 수 (requests per second)
    """
    # Hidden state 크기 (bytes)
    element_size = torch.tensor(0, dtype=dtype).element_size()
    hidden_size_bytes = hidden_size * element_size
    
    # 하나의 요청당 전송 데이터 크기 (hidden state 하나)
    request_size_bytes = hidden_size_bytes
    
    # 기본 대역폭 추정 (Gbps -> Mbps -> bps)
    if bandwidth_mbps is None:
        # 일반적인 인터넷 연결: 100 Mbps ~ 1 Gbps
        bandwidth_mbps = 100.0  # 기본값: 100 Mbps
    
    bandwidth_bps = bandwidth_mbps * 1_000_000 / 8  # bytes per second
    
    # 네트워크 처리량 = 대역폭 / 요청당 크기
    network_rps = bandwidth_bps / request_size_bytes if request_size_bytes > 0 else 0.0
    
    return network_rps


def get_server_throughput(
    model,
    device: torch.device,
    num_blocks: int = 1,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.float16,
    network_bandwidth_mbps: Optional[float] = None,
    relay_penalty: float = 0.2,
) -> float:
    """
    서버 전체 처리량 계산 (논문 Section 3.1)
    
    최종 처리량 = min(컴퓨팅 처리량, 네트워크 처리량)
    
    Args:
        model: PyTorch 모델
        device: 디바이스
        num_blocks: 담당하는 블록 개수
        hidden_size: Hidden state 크기
        dtype: 데이터 타입
        network_bandwidth_mbps: 네트워크 대역폭 (Mbps)
        relay_penalty: Relay를 통한 연결 시 패널티 (0.0 ~ 1.0)
    
    Returns:
        서버 처리량 (requests per second)
    """
    # 컴퓨팅 처리량 측정
    compute_metrics = measure_compute_throughput(
        model=model,
        device=device,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        dtype=dtype,
    )
    
    compute_throughput = compute_metrics["inference_rps"]
    
    # 네트워크 처리량 추정
    network_throughput = estimate_network_throughput(
        hidden_size=hidden_size,
        dtype=dtype,
        bandwidth_mbps=network_bandwidth_mbps,
    )
    
    # Relay 패널티 적용 (선택적)
    if relay_penalty > 0:
        network_throughput *= (1.0 - relay_penalty)
    
    # 최종 처리량 = min(컴퓨팅, 네트워크)
    final_throughput = min(compute_throughput, network_throughput)
    
    logger.debug(
        f"Server throughput: compute={compute_throughput:.2f} rps, "
        f"network={network_throughput:.2f} rps, "
        f"final={final_throughput:.2f} rps"
    )
    
    return final_throughput

