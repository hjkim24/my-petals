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
        model: PyTorch 모델 (StageSegment, StageLast 등)
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
    
    # 모델 타입 확인 (StageSegment/StageLast는 position_ids, attention_mask 필요)
    model_class_name = model.__class__.__name__
    requires_position_ids = model_class_name in ["StageSegment", "StageLast"]
    
    try:
        # Forward pass 처리량 측정
        with torch.inference_mode():
            # 워밍업
            warmup_success = False
            for warmup_step in range(warmup_steps):
                hidden_states = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=device, dtype=dtype
                )
                
                if requires_position_ids:
                    # StageSegment/StageLast는 position_ids와 attention_mask 필요
                    position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)
                    try:
                        output = model(
                            hidden_states,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            use_cache=False
                        )
                        # 출력이 tuple인 경우 처리 (hidden_states, past_key_values)
                        if isinstance(output, tuple):
                            _ = output[0]
                        else:
                            _ = output
                        warmup_success = True
                    except Exception as e:
                        logger.debug(f"Forward pass failed during warmup step {warmup_step+1}/{warmup_steps}: {e}")
                        # 마지막 워밍업에서도 실패하면 경고
                        if warmup_step == warmup_steps - 1:
                            logger.warning(f"All warmup steps failed, proceeding anyway")
                else:
                    # 일반 모델 (Stage0 등)
                    try:
                        _ = model(hidden_states)
                        warmup_success = True
                    except Exception as e:
                        logger.debug(f"Forward pass failed during warmup: {e}")
            
            if not warmup_success and requires_position_ids:
                logger.warning("Warmup failed but continuing with benchmark")
            
            # 벤치마크
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            successful_steps = 0
            for step in range(benchmark_steps):
                hidden_states = torch.randn(
                    batch_size, seq_len, hidden_size,
                    device=device, dtype=dtype
                )
                
                if requires_position_ids:
                    position_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)
                    try:
                        output = model(
                            hidden_states,
                            position_ids=position_ids,
                            attention_mask=attention_mask,
                            past_key_values=None,
                            use_cache=False
                        )
                        # 출력이 tuple인 경우 처리 (hidden_states, past_key_values)
                        if isinstance(output, tuple):
                            _ = output[0]
                        else:
                            _ = output
                        successful_steps += 1
                    except Exception as e:
                        logger.warning(f"Forward pass failed at step {step+1}/{benchmark_steps}: {e}")
                        # 계속 진행하되 실패한 스텝은 카운트하지 않음
                        continue
                else:
                    try:
                        _ = model(hidden_states)
                        successful_steps += 1
                    except Exception as e:
                        logger.warning(f"Forward pass failed at step {step+1}/{benchmark_steps}: {e}")
                        continue
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            elapsed = time.time() - start_time
            
            if successful_steps == 0:
                logger.warning("All benchmark steps failed, returning 0.0 rps")
                forward_rps = 0.0
            else:
                forward_rps = successful_steps / elapsed if elapsed > 0 else 0.0
        
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
    # 처리량 측정 실패 시 (0.0) 네트워크 처리량 사용
    if compute_throughput <= 0:
        logger.warning(
            f"Compute throughput measurement failed (got {compute_throughput:.2f} rps), "
            f"using network throughput estimate: {network_throughput:.2f} rps"
        )
        final_throughput = network_throughput
    else:
        final_throughput = min(compute_throughput, network_throughput)
    
    # 최종 fallback: 둘 다 0이면 기본값 사용
    if final_throughput <= 0:
        logger.warning("Both compute and network throughput are 0, using default: 10.0 rps")
        final_throughput = 10.0
    
    logger.info(
        f"Server throughput: compute={compute_throughput:.2f} rps, "
        f"network={network_throughput:.2f} rps, "
        f"final={final_throughput:.2f} rps"
    )
    
    return final_throughput

