import argparse
import asyncio
import os
from uuid import uuid4
import time

import torch
from transformers import AutoTokenizer
import logging
from hivemind import DHT, get_dht_time
from hivemind.p2p import P2P
from hivemind.utils.logging import get_logger

# Import 경로 처리: 패키지로 실행되거나 직접 실행될 때 모두 지원
try:
    # 패키지로 실행될 때 (python -m src.main)
    from .llama_partition import load_stage_model, Stage0, StageSegment, StageLast
    from .rpc_transport import RpcTransport
    from .rpc_handler import StageConnectionHandler
except ImportError:
    # 직접 실행될 때 (python src/main.py)
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.llama_partition import load_stage_model, Stage0, StageSegment, StageLast
    from src.rpc_transport import RpcTransport
    from src.rpc_handler import StageConnectionHandler

logger = get_logger(__name__)
# Ensure logs are emitted when running from terminal
logging.basicConfig(level=logging.INFO)


def build_masks(seq_len: int, device, dtype=None):
    # batch=1, no padding
    mask_dtype = dtype if dtype is not None else torch.float
    attn = torch.ones(1, seq_len, device=device, dtype=mask_dtype)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq]
    return attn, pos


def parse_splits(splits_str: str):
    """Parse comma-separated splits string into list of integers."""
    return [int(x.strip()) for x in splits_str.split(",")]


def _format_initial_peers(dht_initial_peers: str) -> list:
    """Format DHT initial peers from comma-separated string."""
    if not dht_initial_peers:
        return []
    peers = [p.strip() for p in dht_initial_peers.split(",") if p.strip()]
    return peers


def _get_local_ip() -> str:
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@torch.inference_mode() # gradient 계산 비활성화하여 오버헤드 제거
def run_rank0(args, device, splits):
    """Run Stage0 (client side)."""

    try:
        from transformers.cache_utils import Cache  # type: ignore
    except Exception:
        Cache = None

    def _describe_past(past):
        if past is None:
            return "None"
        if Cache is not None and isinstance(past, Cache):
            try:
                l0 = past.get_seq_length(0)
            except Exception:
                l0 = "err"
            return f"{type(past).__name__}(len={l0})"
        if isinstance(past, (list, tuple)):
            if len(past) == 0:
                return "empty tuple"
            first = past[0]
            if isinstance(first, (list, tuple)) and first and first[0] is not None:
                return f"tuple(len={len(past)}, first_shape={tuple(first[0].shape)})"
            return f"tuple(len={len(past)}, first={first})"
        return str(type(past))

    # 1. Initialize
    full = load_stage_model(args.model, device, role="stage0", end=splits[0], dtype=args.dtype)
    s0 = Stage0(full, splits[0]).to(device) # load to GPU

    # connect to DHT Network with initial(stage1) DHT peer address
    dht_peers = _format_initial_peers(args.dht_initial_peers)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tx = RpcTransport(
        device=device,
        stage=0,
        dht_initial_peers=dht_peers,
        dht_port=args.dht_port,
        rpc_port=args.rpc_port,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # 기존에 DHT network에 등록되어있는 keys
        stage_keys=[f"mini_petals:stage{i}" for i in range(1, 4)],
    )

    prompt = args.prompt
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # [1, L]
    L = input_ids.size(1)

    # 입력 토큰 차원에 맞게 attention mask, position embedding tensor 생성
    model_dtype = s0.embed_tokens.weight.dtype
    attn, pos = None, torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

    past0 = None # 첫 호출이므로 KV Cache 없음
    prompt_ids = input_ids.clone()

    # 2. Prefill: 입력 시퀀스 전체를 처리하고 첫 번째 토큰 생성
    t_prefill_start = time.perf_counter()
    from src.utils import normalize_cache
    hidden, past0 = s0(input_ids, pos, attn, past0, use_cache=True)  # [1, L, H]
    past0 = normalize_cache(past0)
    # logger.info(f"Stage0 prefill: prompt_len={L}, past summary: {_describe_past(past0)}")
    # logger.info(f"Stage0 prefill hidden stats: shape={hidden.shape}, min={hidden.min().item():.4f}, max={hidden.max().item():.4f}, mean={hidden.mean().item():.4f}, std={hidden.std().item():.4f}")

    session_id = str(uuid4())
    max_length = L + args.max_new_tokens

    tx.send_prefill(L, hidden, session_id=session_id, max_length=max_length)  # [1, L, H]
    next_id = tx.recv_token()
    generated = [next_id]
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    logger.info(f"Prefill completed in {prefill_time:.3f}s")
    logger.info(f"Generated token: {next_id} ({tok.decode([next_id], skip_special_tokens=True)})")

    # 3. Decoding (for newly generated token)
    cur_len = L + 1
    t_decode_start = time.perf_counter()
    eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    
    # 연속 반복 감지용
    consecutive_repeat_count = 0
    last_token = None
    
    # 성능 통계용
    decode_total_times = []
    
    for _ in range(args.max_new_tokens - 1): # Prefill에서 1개 토큰 생성했으므로 max - 1 생성
        # Stage0 cache가 비어 있으면 전체 히스토리를 재계산하여 캐시를 복원한다.
        need_full_recompute = (past0 is None) or (isinstance(past0, (list, tuple)) and len(past0) == 0)

        if need_full_recompute:
            logger.warning("Stage0 cache missing; recomputing from full history.")
            full_input = torch.cat([prompt_ids, torch.tensor([generated], device=device, dtype=torch.long)], dim=1)
            full_pos = torch.arange(full_input.shape[1], device=device, dtype=torch.long).unsqueeze(0)
            attn = None
            hidden_full, past0 = s0(full_input, full_pos, attn, None, use_cache=True)
            past0 = normalize_cache(past0)
            hidden = hidden_full[:, -1:, :]  # 마지막 토큰 hidden만 사용
            past_len = full_input.shape[1] - 1
            # logger.info(f"Stage0 recompute: total_len={full_input.shape[1]}, past_len={past_len}, hidden_shape={hidden.shape}, past summary: {_describe_past(past0)}")
        else:
            new_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
            if isinstance(past0, (list, tuple)) and isinstance(past0[0], (list, tuple)) and past0[0][0] is not None:
                past_len = past0[0][0].shape[-2]
            elif Cache is not None and isinstance(past0, Cache):
                past_len = past0.get_seq_length(0)
            else:
                past_len = cur_len - 1
                logger.warning(f"Stage0 past tuple missing first entry, fallback past_len={past_len}, past summary: {_describe_past(past0)}")
            attn = None
            pos = torch.tensor([[past_len]], device=device, dtype=torch.long)
            # logger.info(f"Stage0 decode step {cur_len-L}: past_len={past_len}, pos_ids={pos.tolist()}, input_token={next_id}")
            hidden, past0 = s0(new_input, pos, attn, past0, use_cache=True)  # [1,1,H]
            past0 = normalize_cache(past0)
            # if isinstance(past0, (list, tuple)) and past0 and isinstance(past0[0], (list, tuple)) and past0[0][0] is not None:
            #     new_past_len = past0[0][0].shape[-2]
            # elif Cache is not None and isinstance(past0, Cache):
            #     new_past_len = past0.get_seq_length(0)
            # else:
            #     new_past_len = "N/A"
            # logger.info(f"Stage0: hidden shape={hidden.shape}, new_past_len={new_past_len}, past summary: {_describe_past(past0)}")

        # send_prefill과 동일 (generated_tokens 전달)
        tx.send_decode_step(cur_len, hidden, session_id=session_id, max_length=max_length, generated_tokens=generated)  # [1,1,H]
        next_id = tx.recv_token()
        
        # 출력 품질 확인: 각 토큰 디코딩 및 부분 텍스트 출력
        # next_token_text = tok.decode([next_id], skip_special_tokens=True)
        # partial_text = tok.decode(generated + [next_id], skip_special_tokens=True)
        # logger.info(f"Stage0: received token={next_id} ('{next_token_text}') | Partial: '{partial_text[-50:]}'")

        # EOS 토큰 체크 - 생성 중단
        if eos_token_id is not None and next_id == eos_token_id:
            logger.info("EOS token generated, stopping generation")
            break
        
        # 연속 반복 체크 (같은 토큰이 5번 연속 나오면 중단)
        if next_id == last_token:
            consecutive_repeat_count += 1
            if consecutive_repeat_count >= 5:
                next_token_text = tok.decode([next_id], skip_special_tokens=True)
                logger.warning(f"Consecutive repetition detected (token {next_id}='{next_token_text}'), stopping generation")
                break
        else:
            consecutive_repeat_count = 0
            last_token = next_id

        if tx.last_decode_total is not None:
            decode_total_times.append(tx.last_decode_total)

        generated.append(next_id)
        cur_len += 1

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    total_time = t_decode_end - t_prefill_start

    # 생성된 모든 토큰을 한 번에 디코딩하여 출력
    generated_text = tok.decode(generated, skip_special_tokens=True)
    
    # ========== 출력 품질 평가 ==========
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"GENERATED: {generated_text}")
    print(f"{'='*80}\n")
    
    # 품질 메트릭 계산
    prompt_tokens = len(input_ids[0])
    generated_tokens = len(generated) - prompt_tokens  # generated에는 prompt도 포함될 수 있음
    actual_generated = generated[prompt_tokens:] if len(generated) > prompt_tokens else generated
    actual_generated_text = tok.decode(actual_generated, skip_special_tokens=True)
    
    # 반복 토큰 비율 계산
    unique_tokens = len(set(actual_generated))
    repetition_ratio = 1.0 - (unique_tokens / len(actual_generated)) if len(actual_generated) > 0 else 0.0
    
    # EOS 전에 끝났는지 확인
    ended_with_eos = eos_token_id is not None and generated[-1] == eos_token_id if generated else False
    
    logger.info(f"\n{'='*80}")
    logger.info(f"OUTPUT QUALITY METRICS:")
    logger.info(f"  Prompt: '{prompt}'")
    logger.info(f"  Generated text: '{actual_generated_text}'")
    logger.info(f"  Generated tokens: {len(actual_generated)}")
    logger.info(f"  Unique tokens: {unique_tokens}")
    logger.info(f"  Repetition ratio: {repetition_ratio:.2%}")
    logger.info(f"  Ended with EOS: {ended_with_eos}")
    logger.info(f"  Consecutive repeats detected: {consecutive_repeat_count >= 5}")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nDecode completed in {decode_time:.3f}s")
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"TTFT (Time to First Token): {prefill_time:.3f}s")
    logger.info(f"Throughput: {len(actual_generated) / total_time:.2f} tokens/s")

    # 성능 통계 출력
    if hasattr(tx, 'decode_stage_times') and tx.decode_stage_times:
        stage_lines = []
        for stage_num in [1, 2, 3]:
            stage_times = [t for t in tx.decode_stage_times if t.get('stage') == stage_num]
            if stage_times:
                avg_time = sum(t['time'] for t in stage_times) / len(stage_times)
                stage_lines.append(f"Stage{stage_num}={avg_time*1000:.1f}ms")
        if stage_lines:
            logger.info("Decode stages (avg): " + ", ".join(stage_lines))

    
    # Cleanup
    tx.shutdown()


@torch.inference_mode()
def run_stage_server(args, device, splits):
    """Run a server stage (1, 2, or 3) with optional Load Balancing."""
    # Load Balancing 활성화 여부 확인
    use_load_balancing = getattr(args, 'use_load_balancing', False)
    num_blocks = getattr(args, 'num_blocks', None)  # Load Balancing 시 담당할 블록 개수
    total_blocks = getattr(args, 'total_blocks', None)  # 전체 블록 개수
    
    if use_load_balancing:
        return run_stage_server_with_load_balancing(
            args, device, splits, num_blocks, total_blocks
        )
    
    # 기존 고정 splits 방식
    return run_stage_server_fixed(args, device, splits)


def run_stage_server_fixed(args, device, splits):
    """Run a server stage with fixed splits (original implementation)."""
    # CPU 오프로딩 옵션
    use_cpu_offload = getattr(args, 'use_cpu_offload', False)
    keep_layers_on_gpu = getattr(args, 'keep_layers_on_gpu', 0)
    
    if args.stage == 1:
        start, end = splits[0], splits[1]
        full = load_stage_model(
            args.model, device, role="segment", 
            start=start, end=end, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageSegment(
            full, start, end,
            gpu_device=device,
            keep_layers_on_gpu=keep_layers_on_gpu
        )
        final_stage = False
    elif args.stage == 2:
        start, end = splits[1], splits[2]
        full = load_stage_model(
            args.model, device, role="segment",
            start=start, end=end, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageSegment(
            full, start, end,
            gpu_device=device,
            keep_layers_on_gpu=keep_layers_on_gpu
        )
        final_stage = False
    elif args.stage == 3:
        start = splits[2]
        full = load_stage_model(
            args.model, device, role="last",
            start=start, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageLast(
            full, start,
            gpu_device=device,
            keep_layers_on_gpu=keep_layers_on_gpu
        )
        final_stage = True
    else:
        raise ValueError("stage must be 1, 2, or 3 for server")
    
    return _setup_and_run_server(args, device, stage_model, final_stage)


def run_stage_server_with_load_balancing(args, device, splits, num_blocks, total_blocks):
    """Run a server stage with Load Balancing (논문식 Full Load Balancing)."""
    try:
        from .load_balancing import (
            choose_best_blocks, should_choose_other_blocks, ServerState
        )
        from .dht_utils import (
            get_remote_module_infos, register_server_on_dht,
            register_blocks_on_dht, update_server_throughput_on_dht
        )
        from .throughput_measurement import get_server_throughput
    except ImportError as e:
        logger.error(f"Failed to import Load Balancing modules: {e}")
        logger.error("Falling back to fixed splits mode")
        return run_stage_server_fixed(args, device, splits)
    
    import random
    import asyncio
    
    # DHT 초기화
    initial_peers_list = _format_initial_peers(args.dht_initial_peers)
    local_ip = _get_local_ip()
    announce_ip = args.public_ip if args.public_ip else local_ip
    public_dht_port = args.public_dht_port if args.public_dht_port is not None else args.dht_port
    public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port
    
    if args.public_ip:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.dht_port}"]
        announce_maddrs = [f"/ip4/{args.public_ip}/tcp/{public_dht_port}"]
    else:
        host_maddrs = [f"/ip4/{local_ip}/tcp/{args.dht_port}"]
        announce_maddrs = None
    
    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
    )
    
    # DHT 초기화 후 multiaddr 정보 로깅 (Stage2 연결용)
    visible = dht.get_visible_maddrs()
    dht_peer_id = str(dht.peer_id)
    
    if args.public_ip:
        visible_str = [str(m) for m in visible] if visible else []
        has_public_ip = any(args.public_ip in str(m) for m in visible_str)
        
        if not has_public_ip:
            # 공인 IP와 외부 포트를 사용한 multiaddr을 명시적으로 생성
            public_maddr = f"/ip4/{args.public_ip}/tcp/{public_dht_port}/p2p/{dht_peer_id}"
            logger.warning(
                f"DHT visible multiaddrs do not contain public IP {args.public_ip}. "
                f"Use this multiaddr for --dht_initial_peers: {public_maddr}"
            )
            logger.info(f"DHT visible multiaddrs (may contain private IP): {visible}")
        else:
            logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    elif visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    else:
        # 공인 IP가 없고 visible maddrs도 없으면 fallback
        fallback_maddr = f"/ip4/{announce_ip}/tcp/{public_dht_port}/p2p/{dht_peer_id}"
        logger.info(
            f"DHT visible multiaddrs not available; try fallback: {fallback_maddr}"
        )
    
    # 기본값 설정
    if num_blocks is None:
        num_blocks = 4  # 기본값: 4개 블록
    if total_blocks is None:
        # 모델에서 전체 레이어 수 추론 (간단한 휴리스틱)
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model)
            total_blocks = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 32
        except:
            total_blocks = 32
    
    # Load Balancing 메인 루프
    balance_quality = getattr(args, 'balance_quality', 0.75)
    mean_balance_check_period = getattr(args, 'mean_balance_check_period', 120.0)
    
    while True:
        try:
            # 1. 현재 시스템 상태 조회 (DHT 전파 시간 고려하여 재시도)
            module_infos = []
            max_retries = 3
            retry_delay = 2.0  # 초
            
            for retry in range(max_retries):
                module_infos = get_remote_module_infos(dht, args.model, total_blocks)
                logger.info(f"Retrieved {len(module_infos)} module infos from DHT (attempt {retry+1}/{max_retries})")
                
                if len(module_infos) > 0 or retry == max_retries - 1:
                    break
                
                # 다른 서버 정보를 찾지 못했지만 첫 번째 시도가 아닌 경우 잠시 대기
                if retry < max_retries - 1:
                    logger.info(f"No existing servers found, waiting {retry_delay}s for DHT propagation...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # 지수 백오프
            
            # 조회된 서버 정보 로깅
            if len(module_infos) > 0:
                unique_servers = set()
                for info in module_infos:
                    if info.server_info:
                        unique_servers.add(str(info.server_info.peer_id)[:16])
                logger.info(f"Found {len(unique_servers)} unique server(s) in DHT: {list(unique_servers)}")
            
            # 2. 최적 블록 선택 (규칙 1 또는 규칙 2)
            if len(module_infos) == 0:
                # 첫 서버이거나 다른 서버 정보 없음 - 처음부터 시작
                block_indices = list(range(min(num_blocks, total_blocks)))
                logger.info(f"No existing servers found, selecting first {num_blocks} blocks: {block_indices}")
            else:
                # 논문식 최적 블록 선택
                block_indices = choose_best_blocks(num_blocks, module_infos, total_blocks)
                logger.info(f"Load balancing selected blocks: {block_indices}")
            
            start_block = min(block_indices)
            end_block = max(block_indices) + 1
            
            logger.info(f"Selected blocks: {block_indices} (start={start_block}, end={end_block})")
            
            # 3. 모델 로드
            use_cpu_offload = getattr(args, 'use_cpu_offload', False)
            keep_layers_on_gpu = getattr(args, 'keep_layers_on_gpu', 0)
            
            if end_block >= total_blocks:
                # 마지막 블록 포함 - final stage
                full = load_stage_model(
                    args.model, device, role="last",
                    start=start_block, dtype=args.dtype,
                    use_cpu_offload=use_cpu_offload
                )
                stage_model = StageLast(
                    full, start_block,
                    gpu_device=device,
                    keep_layers_on_gpu=keep_layers_on_gpu
                )
                final_stage = True
            else:
                # 중간 세그먼트
                full = load_stage_model(
                    args.model, device, role="segment",
                    start=start_block, end=end_block, dtype=args.dtype,
                    use_cpu_offload=use_cpu_offload
                )
                stage_model = StageSegment(
                    full, start_block, end_block,
                    gpu_device=device,
                    keep_layers_on_gpu=keep_layers_on_gpu
                )
                final_stage = False
            
            # 4. 처리량 측정
            try:
                hidden_size = getattr(stage_model.config, 'hidden_size', 4096)
                throughput = get_server_throughput(
                    stage_model, device, num_blocks=len(block_indices),
                    hidden_size=hidden_size, dtype=args.dtype,
                    network_bandwidth_mbps=getattr(args, 'network_bandwidth_mbps', None),
                )
            except Exception as e:
                logger.warning(f"Failed to measure throughput, using default: {e}")
                throughput = 10.0  # 기본값
            
            # 5. DHT에 등록
            peer_id = dht.peer_id
            register_server_on_dht(
                dht, peer_id, start_block, end_block, throughput,
                model_name=args.model, server_address=announce_ip,
                state=ServerState.ONLINE
            )
            register_blocks_on_dht(dht, peer_id, block_indices, args.model)
            
            # 6. 서버 실행 (재조정 로직 포함)
            should_rebalance = _setup_and_run_server_with_rebalancing(
                args, device, stage_model, final_stage, dht, peer_id,
                block_indices, throughput, balance_quality, mean_balance_check_period,
                args.model, total_blocks
            )
            
            # 재조정이 필요하면 루프 계속, 아니면 종료
            if not should_rebalance:
                logger.info("Server shutting down normally")
                break
            
            # 재조정 루프에서 나왔다면 다시 블록 선택
            logger.info("Re-evaluating block assignment...")
            time.sleep(1)  # 잠시 대기
            
        except KeyboardInterrupt:
            logger.info("Shutting down Load Balancing server...")
            break
        except Exception as e:
            logger.error(f"Error in Load Balancing loop: {e}", exc_info=True)
            time.sleep(5)  # 에러 후 재시도 전 대기


def _setup_and_run_server(args, device, stage_model, final_stage):
    """기존 서버 설정 및 실행 로직 (고정 splits용)."""
    dht_peers = args.dht_initial_peers.split(",") if args.dht_initial_peers else []
    initial_peers_list = _format_initial_peers(args.dht_initial_peers)
    local_ip = _get_local_ip()
    
    # 공인 IP가 제공되면 announce_maddrs에 사용, 없으면 local_ip 사용
    announce_ip = args.public_ip if args.public_ip else local_ip
    
    # 외부 포트가 제공되면 사용, 없으면 내부 포트 사용 (포트 포워딩 시나리오 대응)
    public_dht_port = args.public_dht_port if args.public_dht_port is not None else args.dht_port
    public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port

    # 공인 IP가 제공되면 모든 인터페이스(0.0.0.0)에서 리스닝하여 외부 접근 허용
    # announce_maddrs에 공인 IP와 외부 포트를 설정하여 다른 피어에게 알림
    if args.public_ip:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.dht_port}"]
        announce_maddrs = [f"/ip4/{args.public_ip}/tcp/{public_dht_port}"]
    else:
        host_maddrs = [f"/ip4/{local_ip}/tcp/{args.dht_port}"]
        announce_maddrs = None

    # Initialize DHT Network
    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
    )

    # 초기화된 DHT 네트워크의 multiaddr 리스트 반환
    visible = dht.get_visible_maddrs()
    peer_id = str(dht.peer_id)
    
    # 공인 IP가 제공되었을 때, visible maddrs에 공인 IP가 포함되어 있는지 확인
    if args.public_ip:
        # visible maddrs를 문자열로 변환하여 공인 IP 포함 여부 확인
        visible_str = [str(m) for m in visible] if visible else []
        has_public_ip = any(args.public_ip in str(m) for m in visible_str)
        
        if not has_public_ip:
            # 공인 IP와 외부 포트를 사용한 multiaddr을 명시적으로 생성
            public_maddr = f"/ip4/{args.public_ip}/tcp/{public_dht_port}/p2p/{peer_id}"
            logger.warning(
                f"DHT visible multiaddrs do not contain public IP {args.public_ip}. "
                f"Use this multiaddr for --dht_initial_peers: {public_maddr}"
            )
            logger.info(f"DHT visible multiaddrs (may contain private IP): {visible}")
        else:
            logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    elif visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    else:
        # 공인 IP가 없고 visible maddrs도 없으면 local_ip 사용
        fallback_ip = announce_ip
        fallback_dht_port = public_dht_port  # 외부 포트가 있으면 사용, 없으면 내부 포트
        fallback = [f"/ip4/{fallback_ip}/tcp/{fallback_dht_port}/p2p/{peer_id}"]
        logger.info(
            f"DHT visible multiaddrs not available; try fallback: {fallback}"
        )


    handler = StageConnectionHandler(
        dht=dht,
        stage_model=stage_model,
        device=device,
        request_timeout=args.request_timeout,
        final_stage=final_stage,
    )

    async def setup_and_run():
        p2p = None
        try:
            logger.info(f"Initializing P2P for Stage{args.stage}...")
            # P2P Daemon 생성(hivemind 내장)
            # 공인 IP가 제공되면 모든 인터페이스(0.0.0.0)에서 리스닝하여 외부 접근 허용
            if args.public_ip:
                p2p_host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.rpc_port}"]
            else:
                p2p_host_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}"]
            p2p = await P2P.create(host_maddrs=p2p_host_maddrs)
            logger.info(f"P2P initialized successfully, PeerID: {p2p.peer_id}")
            # get_visible_maddrs is async in this hivemind version
            visible_maddrs = await p2p.get_visible_maddrs() # P2P가 자동으로 감지한 외부 접근 가능한 multiaddr 리스트
            p2p_maddr = getattr(p2p, "daemon_listen_maddr", None) # P2P Daemon이 실제로 리스닝하는 주소
            p2p_maddrs = [str(m) for m in visible_maddrs] if visible_maddrs else []
            # visible_maddrs와 p2p_maddr를 병합
            if p2p_maddr:
                p2p_maddrs.append(str(p2p_maddr))
            
            # 공인 IP가 제공되었을 때, visible maddrs에 공인 IP가 포함되어 있는지 확인
            if args.public_ip:
                has_public_ip = any(args.public_ip in m for m in p2p_maddrs)
                if not has_public_ip:
                    # 공인 IP와 외부 RPC 포트를 사용한 multiaddr을 명시적으로 생성
                    public_p2p_maddr = f"/ip4/{args.public_ip}/tcp/{public_rpc_port}/p2p/{p2p.peer_id}"
                    logger.warning(
                        f"Stage{args.stage} P2P visible maddrs do not contain public IP {args.public_ip}. "
                        f"Use this multiaddr: {public_p2p_maddr}"
                    )
                    p2p_maddrs.append(public_p2p_maddr)
            
            if p2p_maddrs:
                logger.info(f"Stage{args.stage} P2P listen maddrs: {p2p_maddrs}")
            else:
                # Fallback to announced addr using rpc_port (공인 IP와 외부 포트 사용)
                p2p_maddrs = [f"/ip4/{announce_ip}/tcp/{public_rpc_port}/p2p/{p2p.peer_id}"]
                logger.warning(f"Stage{args.stage} P2P listen maddrs unknown; using fallback {p2p_maddrs}")

            peer_info = {
                "peer_id": str(p2p.peer_id),          # 서버 고유 ID (subkey로도 사용)
                "timestamp": get_dht_time(),
                "stage": args.stage,
            }
            if p2p_maddrs:
                peer_info["p2p_maddrs"] = p2p_maddrs

            # ✅ (핵심) 같은 stage_key 아래에 여러 서버가 공존하도록 subkey 사용
            STAGE_KEY = f"mini_petals:stage{args.stage}"
            SUBKEY = str(p2p.peer_id)

            # ✅ (핵심) 죽은 서버가 오래 남지 않게 TTL 짧게 + heartbeat 갱신
            TTL = 45  # seconds (권장: 30~60)

            def _store_once():
                peer_info["timestamp"] = get_dht_time()
                dht.store(
                    key=STAGE_KEY,
                    subkey=SUBKEY,
                    value=peer_info,
                    expiration_time=get_dht_time() + TTL,
                )

            # 최초 1회 등록
            _store_once()
            logger.info(f"Stage{args.stage} registered in DHT: key={STAGE_KEY}, subkey={SUBKEY[:8]}..., ttl={TTL}s")

            # 주기적 heartbeat (TTL/3 마다 갱신)
            async def heartbeat():
                while True:
                    try:
                        _store_once()
                    except Exception as e:
                        logger.warning(f"Stage{args.stage} heartbeat failed: {e}")
                    await asyncio.sleep(TTL / 3)

            hb_task = asyncio.create_task(heartbeat())


            # P2P Daemon에 StageConnectionHandler의 rpc_* 메서드 등록
            await handler.add_p2p_handlers(p2p)
            logger.info(f"Stage{args.stage} handlers registered, waiting for requests...")

            # P2P 초기화 대기
            await asyncio.sleep(0.5)

            # 무한 대기 (요청 처리)
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info(f"Stage{args.stage} shutting down...")
        finally:
            if 'hb_task' in locals():
                hb_task.cancel()
            if p2p:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")

    asyncio.run(setup_and_run())


def _setup_and_run_server_with_rebalancing(
    args, device, stage_model, final_stage, dht, peer_id,
    block_indices, throughput, balance_quality, mean_balance_check_period,
    model_name, total_blocks
) -> bool:
    """
    Load Balancing을 지원하는 서버 실행 (주기적 재조정 포함)
    
    Returns:
        True이면 재조정 필요 (블록 변경 필요), False이면 정상 종료
    """
    try:
        from .load_balancing import should_choose_other_blocks
        from .dht_utils import get_remote_module_infos, update_server_throughput_on_dht
    except ImportError as e:
        logger.error(f"Failed to import Load Balancing modules: {e}")
        return False
    
    import random
    
    handler = StageConnectionHandler(
        dht=dht,
        stage_model=stage_model,
        device=device,
        request_timeout=args.request_timeout,
        final_stage=final_stage,
    )
    
    should_rebalance = False
    stop_event = asyncio.Event()
    
    async def setup_and_run_with_rebalancing():
        nonlocal should_rebalance
        p2p = None
        hb_task = None
        rebalance_task = None
        
        try:
            local_ip = _get_local_ip()
            announce_ip = args.public_ip if args.public_ip else local_ip
            public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port
            
            # P2P 초기화
            if args.public_ip:
                p2p_host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.rpc_port}"]
            else:
                p2p_host_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}"]
            
            p2p = await P2P.create(host_maddrs=p2p_host_maddrs)
            logger.info(f"P2P initialized for Load Balancing server, PeerID: {p2p.peer_id}")
            
            # Heartbeat 로직
            STAGE_KEY = f"mini_petals:stage{args.stage}"
            SUBKEY = str(p2p.peer_id)
            TTL = 45
            
            def _store_once():
                peer_info = {
                    "peer_id": str(p2p.peer_id),
                    "timestamp": get_dht_time(),
                    "stage": args.stage,
                    "blocks": block_indices,
                    "throughput": throughput,
                }
                dht.store(
                    key=STAGE_KEY,
                    subkey=SUBKEY,
                    value=peer_info,
                    expiration_time=get_dht_time() + TTL,
                )
            
            _store_once()
            
            async def heartbeat():
                while not stop_event.is_set():
                    try:
                        _store_once()
                        await asyncio.sleep(TTL / 3)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                        await asyncio.sleep(TTL / 3)
            
            hb_task = asyncio.create_task(heartbeat())
            
            # 주기적 재조정 체크 (논문 Appendix D 규칙 2)
            async def rebalance_check():
                nonlocal should_rebalance
                while not stop_event.is_set():
                    try:
                        # 랜덤 대기 시간 (평균 mean_balance_check_period)
                        timeout = random.random() * 2 * mean_balance_check_period
                        await asyncio.sleep(timeout)
                        
                        if stop_event.is_set():
                            break
                        
                        # 처리량 업데이트
                        try:
                            from .throughput_measurement import get_server_throughput
                            hidden_size = getattr(stage_model.config, 'hidden_size', 4096)
                            new_throughput = get_server_throughput(
                                stage_model, device, num_blocks=len(block_indices),
                                hidden_size=hidden_size, dtype=args.dtype,
                            )
                            update_server_throughput_on_dht(dht, peer_id, new_throughput, model_name)
                        except Exception as e:
                            logger.debug(f"Failed to update throughput: {e}")
                        
                        # 재조정 필요 여부 확인
                        module_infos = get_remote_module_infos(dht, model_name, total_blocks)
                        if should_choose_other_blocks(peer_id, module_infos, balance_quality, total_blocks):
                            logger.info("Load balancing detected imbalance, will rebalance blocks")
                            should_rebalance = True
                            stop_event.set()
                            break
                            
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Rebalance check failed: {e}")
                        await asyncio.sleep(10)
            
            rebalance_task = asyncio.create_task(rebalance_check())
            
            # Handler 등록
            await handler.add_p2p_handlers(p2p)
            logger.info(f"Load Balancing server ready, blocks={block_indices}, throughput={throughput:.2f} rps")
            
            # 무한 대기 (재조정 또는 종료 신호까지)
            await stop_event.wait()
            
        except KeyboardInterrupt:
            logger.info("Shutting down Load Balancing server...")
        except Exception as e:
            logger.error(f"Error in Load Balancing server: {e}", exc_info=True)
        finally:
            stop_event.set()
            if hb_task:
                hb_task.cancel()
            if rebalance_task:
                rebalance_task.cancel()
            if p2p:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")
    
    try:
        asyncio.run(setup_and_run_with_rebalancing())
    except KeyboardInterrupt:
        logger.info("Load Balancing server interrupted")
        return False
    
    return should_rebalance


def _setup_and_run_server_with_rebalancing(
    args, device, stage_model, final_stage, dht, peer_id,
    block_indices, throughput, balance_quality, mean_balance_check_period,
    model_name, total_blocks
) -> bool:
    """
    Load Balancing을 지원하는 서버 실행 (주기적 재조정 포함)
    
    Returns:
        True이면 재조정 필요 (블록 변경 필요), False이면 정상 종료
    """
    try:
        from .load_balancing import should_choose_other_blocks
        from .dht_utils import get_remote_module_infos, update_server_throughput_on_dht
    except ImportError as e:
        logger.error(f"Failed to import Load Balancing modules: {e}")
        return False
    
    import random
    import asyncio
    
    handler = StageConnectionHandler(
        dht=dht,
        stage_model=stage_model,
        device=device,
        request_timeout=args.request_timeout,
        final_stage=final_stage,
    )
    
    should_rebalance = False
    stop_event = None  # asyncio.Event로 생성
    
    async def setup_and_run_with_rebalancing():
        nonlocal should_rebalance, stop_event
        stop_event = asyncio.Event()  # async 함수 내에서 생성
        p2p = None
        hb_task = None
        rebalance_task = None
        
        try:
            local_ip = _get_local_ip()
            announce_ip = args.public_ip if args.public_ip else local_ip
            public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port
            
            # P2P 초기화 (기존 로직 재사용)
            if args.public_ip:
                p2p_host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.rpc_port}"]
            else:
                p2p_host_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}"]
            
            p2p = await P2P.create(host_maddrs=p2p_host_maddrs)
            logger.info(f"P2P initialized for Load Balancing server, PeerID: {p2p.peer_id}")
            
            # 기존 heartbeat 로직
            STAGE_KEY = f"mini_petals:stage{args.stage}"
            SUBKEY = str(p2p.peer_id)
            TTL = 45
            
            def _store_once():
                peer_info = {
                    "peer_id": str(p2p.peer_id),
                    "timestamp": get_dht_time(),
                    "stage": args.stage,
                    "blocks": block_indices,
                    "throughput": throughput,
                }
                dht.store(
                    key=STAGE_KEY,
                    subkey=SUBKEY,
                    value=peer_info,
                    expiration_time=get_dht_time() + TTL,
                )
            
            _store_once()
            
            async def heartbeat():
                while not stop_event.is_set():
                    try:
                        _store_once()
                        await asyncio.sleep(TTL / 3)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                        await asyncio.sleep(TTL / 3)
            
            hb_task = asyncio.create_task(heartbeat())
            
            # 주기적 재조정 체크 (논문 Appendix D 규칙 2)
            async def rebalance_check():
                nonlocal should_rebalance
                while not stop_event.is_set():
                    try:
                        # 랜덤 대기 시간 (평균 mean_balance_check_period)
                        timeout = random.random() * 2 * mean_balance_check_period
                        await asyncio.sleep(timeout)
                        
                        if stop_event.is_set():
                            break
                        
                        # 처리량 업데이트
                        try:
                            from .throughput_measurement import get_server_throughput
                            hidden_size = getattr(stage_model.config, 'hidden_size', 4096)
                            new_throughput = get_server_throughput(
                                stage_model, device, num_blocks=len(block_indices),
                                hidden_size=hidden_size, dtype=args.dtype,
                            )
                            update_server_throughput_on_dht(dht, peer_id, new_throughput, model_name)
                        except Exception as e:
                            logger.debug(f"Failed to update throughput: {e}")
                        
                        # 재조정 필요 여부 확인
                        module_infos = get_remote_module_infos(dht, model_name, total_blocks)
                        if should_choose_other_blocks(peer_id, module_infos, balance_quality, total_blocks):
                            logger.info("Load balancing detected imbalance, will rebalance blocks")
                            should_rebalance = True
                            stop_event.set()
                            break
                            
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Rebalance check failed: {e}")
                        await asyncio.sleep(10)
            
            rebalance_task = asyncio.create_task(rebalance_check())
            
            # Handler 등록
            await handler.add_p2p_handlers(p2p)
            logger.info(f"Load Balancing server ready, blocks={block_indices}, throughput={throughput:.2f} rps")
            
            # 무한 대기 (재조정 또는 종료 신호까지)
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            logger.info("Shutting down Load Balancing server...")
        except Exception as e:
            logger.error(f"Error in Load Balancing server: {e}", exc_info=True)
        finally:
            stop_event.set()
            if hb_task:
                hb_task.cancel()
            if rebalance_task:
                rebalance_task.cancel()
            if p2p:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")
    
    try:
        asyncio.run(setup_and_run_with_rebalancing())
    except KeyboardInterrupt:
        logger.info("Load Balancing server interrupted")
        return False
    
    return should_rebalance


def main():
    parser = argparse.ArgumentParser(description="Mini Petals: Distributed Inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True,
                       help="Comma-separated cut points for 4-stage pipeline, e.g., 10,20,30")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                       help="Model dtype: fp16 (default), bf16, fp32")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for text generation")
    parser.add_argument("--dht_initial_peers", type=str, default="",
                       help='Comma-separated list of initial DHT peers (e.g., full multiaddrs /ip4/host/tcp/port/p2p/PeerID)')
    parser.add_argument("--public_ip", type=str, default="",
                       help="Public IP address for DHT announcement (required for cross-instance connections)")
    parser.add_argument("--public_dht_port", type=int, default=None,
                       help="Public DHT port (for port forwarding scenarios like RunPod). If not provided, uses --dht_port")
    parser.add_argument("--public_rpc_port", type=int, default=None,
                       help="Public RPC port (for port forwarding scenarios like RunPod). If not provided, uses --rpc_port")
    parser.add_argument("--dht_port", type=int, default=8000)
    parser.add_argument("--rpc_port", type=int, default=8001)
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2, 3],
                       help='Stage number (0=client, 1/2 mid, 3 final server)')
    parser.add_argument('--request_timeout', type=float, default=30.0,
                       help='Timeout for RPC requests in seconds')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (client->stage3)')
    parser.add_argument('--top_p', type=float, default=0.92, help='Nucleus sampling p')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (0=disabled)')
    parser.add_argument('--use_cpu_offload', action='store_true',
                       help='Enable CPU offloading: keep model parameters on CPU and move to GPU only when needed')
    parser.add_argument('--keep_layers_on_gpu', type=int, default=0,
                       help='Number of recent layers to keep on GPU when using CPU offloading (default: 0)')
    # Load Balancing 옵션
    parser.add_argument('--use_load_balancing', action='store_true',
                       help='Enable Full Load Balancing (논문식)')
    parser.add_argument('--num_blocks', type=int, default=None,
                       help='Number of blocks to serve (for Load Balancing, default: auto)')
    parser.add_argument('--total_blocks', type=int, default=None,
                       help='Total number of blocks in the model (for Load Balancing, default: auto)')
    parser.add_argument('--balance_quality', type=float, default=0.75,
                       help='Load balancing quality threshold (0.75 = 25%% improvement needed, default: 0.75)')
    parser.add_argument('--mean_balance_check_period', type=float, default=120.0,
                       help='Mean period for balance check in seconds (default: 120)')
    parser.add_argument('--network_bandwidth_mbps', type=float, default=None,
                       help='Network bandwidth in Mbps for throughput estimation (default: auto estimate)')
    
    args = parser.parse_args()

    # Get Local Rank for multi GPU environment(if Single, 0)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    args.dtype = dtype_map[args.dtype]
    
    splits = parse_splits(args.splits)
    if args.stage == 0:
        run_rank0(args, device, splits)
    else:
        run_stage_server(args, device, splits)


if __name__ == "__main__":
    main()