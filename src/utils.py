import logging
import time
from typing import Optional, Tuple, Iterable, Dict, Union

import torch

logger = logging.getLogger(__name__)


def measure_network_throughput(config) -> float:
    """
    Measure network throughput using speedtest API.
    Returns: tokens per second
    """
    try:
        import speedtest
    except ImportError:
        logger.warning(
            "speedtest-cli not installed. Install with: pip install speedtest-cli==2.1.3"
        )
        # Default to 100 Mbit/s
        bits_per_token = config.hidden_size * 16  # 16-bit tensors
        default_speed = 100e6  # 100 Mbit/s
        return default_speed / bits_per_token

    bits_per_token = config.hidden_size * 16  # 16-bit tensors
    try:
        logger.info("Measuring network throughput using speedtest...")
        st = speedtest.Speedtest()
        st.get_servers()
        st.get_best_server()
        st.download()
        st.upload()
        results = st.results.dict()
        network_bps = min(results['download'], results['upload'])
        network_rps = network_bps / bits_per_token
        
        logger.info(
            f"Network throughput: {network_rps:.1f} tokens/sec "
            f"({results['download'] / 1e6:.2f} Mbit/s download, "
            f"{results['upload'] / 1e6:.2f} Mbit/s upload)"
        )
        return network_rps
    except Exception as e:
        logger.warning(f"Network benchmark failed: {e}, using default 100 Mbit/s")
        default_speed = 100e6  # 100 Mbit/s
        return default_speed / bits_per_token


def measure_gpu_throughput(
    stage_model,
    device: torch.device,
    config,
    num_layers: int,
    quant_type,
) -> float:
    """
    Measure GPU throughput by running benchmark forward passes.
    Returns: tokens per second
    """
    # 벤치마크 파라미터
    n_warmup = 3
    n_steps = 20
    seq_len = 128  # Prefill 시뮬레이션
    
    with torch.inference_mode():
        # 더미 입력 생성
        dummy_input = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=torch.float16)
        dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        
        # Warmup
        logger.info(f"GPU benchmark: warming up ({n_warmup} steps)...")
        for _ in range(n_warmup):
            # StageSegment or StageLast (hidden_states 사용)
            if hasattr(stage_model, 'embed_tokens'):
                # Stage0 (input_ids 사용) - 서버에서는 사용하지 않지만 호환성을 위해 처리
                dummy_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
                _ = stage_model(
                    dummy_ids,
                    dummy_pos,
                    None,
                    None,
                    use_cache=False,
                )
            else:
                # StageSegment or StageLast
                _ = stage_model(
                    dummy_input,
                    dummy_pos,
                    None,
                    None,
                    use_cache=False,
                )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        
        # 측정
        logger.info(f"GPU benchmark: measuring ({n_steps} steps)...")
        start_time = time.perf_counter()
        for _ in range(n_steps):
            # StageSegment or StageLast (hidden_states 사용)
            if hasattr(stage_model, 'embed_tokens'):
                # Stage0 (input_ids 사용) - 서버에서는 사용하지 않지만 호환성을 위해 처리
                dummy_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
                _ = stage_model(
                    dummy_ids,
                    dummy_pos,
                    None,
                    None,
                    use_cache=False,
                )
            else:
                # StageSegment or StageLast
                _ = stage_model(
                    dummy_input,
                    dummy_pos,
                    None,
                    None,
                    use_cache=False,
                )
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start_time
        
        gpu_rps = (n_steps * seq_len) / elapsed
        
        logger.info(
            f"GPU throughput: {gpu_rps:.1f} tokens/sec "
            f"({seq_len} tokens/batch, {n_steps} steps, {elapsed:.2f}s elapsed)"
        )
        return gpu_rps


def benchmark_server_performance(
    stage_model,
    device: torch.device,
    config,
    num_layers: int,
    quant_type,
) -> Dict[str, float]:
    """
    Benchmark both network and GPU throughput.
    Returns: {
        "network_rps": float,
        "gpu_rps": float,
        "throughput": float,  # min(network_rps, gpu_rps)
    }
    """
    logger.info("Starting server performance benchmark...")
    
    # 네트워크 처리량 측정
    network_rps = measure_network_throughput(config)
    
    # GPU 처리량 측정
    gpu_rps = measure_gpu_throughput(stage_model, device, config, num_layers, quant_type)
    
    # 둘 중 최소값을 throughput으로 사용
    throughput = min(network_rps, gpu_rps)
    
    logger.info(
        f"Server benchmark complete: "
        f"network={network_rps:.1f} tokens/s, "
        f"gpu={gpu_rps:.1f} tokens/s, "
        f"final throughput={throughput:.1f} tokens/s"
    )
    
    return {
        "network_rps": network_rps,
        "gpu_rps": gpu_rps,
        "throughput": throughput,
    }


def extract_kv_tuple(output: Iterable, layer_idx: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a transformer layer output, return (key, value) tuple if present.
    Handles both legacy tuple caches and transformers Cache/DynamicCache objects.
    Expected LLaMA-style outputs:
      - (hidden_states, past_key_value)
      - (hidden_states, attentions, past_key_value) when output_attentions=True
    """
    try:
        from transformers.cache_utils import Cache  # type: ignore
    except Exception:
        Cache = None

    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return None
    candidate = output[-1] if len(output) > 2 else output[1]

    # Handle new transformers Cache objects
    if Cache is not None and isinstance(candidate, Cache):
        try:
            if layer_idx is not None:
                return candidate[layer_idx]
            # Fallback to legacy conversion if no layer index provided
            legacy = candidate.to_legacy_cache()
            if layer_idx is None and len(legacy) > 0:
                return legacy[-1]
        except Exception:
            return None

    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        if all(isinstance(t, torch.Tensor) for t in candidate):
            return candidate  # (key, value)
    return None


def default_position_ids(layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]], seq_len: int, device) -> torch.Tensor:
    """
    Build position_ids using past KV length if available; otherwise start at 0.
    """
    past_len = 0
    if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
        if layer_past[0] is not None and layer_past[0].dim() >= 3:
            past_len = layer_past[0].shape[2]
    return torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)


def normalize_cache(past):
    """
    Convert transformers Cache/DynamicCache to legacy tuple if needed.
    """
    try:
        from transformers.cache_utils import Cache  # type: ignore
    except Exception:
        Cache = None
    if Cache is not None and isinstance(past, Cache):
        try:
            return past.to_legacy_cache()
        except Exception:
            return past
    return past
