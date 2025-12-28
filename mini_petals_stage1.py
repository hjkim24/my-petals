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
from partition import load_stage_model, Stage0, StageSegment, StageLast
from rpc_transport import RpcTransport
from rpc_handler import StageConnectionHandler

logger = get_logger(__name__)
# Ensure logs are emitted when running from terminal
logging.basicConfig(level=logging.INFO)


def build_masks(seq_len: int, device, dtype=None):
    # batch=1, no padding
    mask_dtype = dtype if dtype is not None else torch.float
    attn = torch.ones(1, seq_len, device=device, dtype=mask_dtype)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq]
    return attn, pos


# dht_initial_peer에 대한 Multiaddr formatting(/p2p 이외의 주소의 경우 ValueError 처리)
def _format_initial_peers(dht_initial_peers):
    initial_peers_list = []
    for peer in dht_initial_peers:
        peer = peer.strip()
        if not peer:
            continue
        if "/p2p/" in peer:
            initial_peers_list.append(peer)
        elif ":" in peer:
            raise ValueError(
                f"dht_initial_peers entry '{peer}' is missing '/p2p/<peer_id>'. "
                "Use the full multiaddr printed by Stage1 (e.g., /ip4/127.0.0.1/tcp/8000/p2p/<peer_id>)."
            )
        else:
            initial_peers_list.append(peer)
    return initial_peers_list

# arg로 받은 layer split parsing(str -> int)
def parse_splits(s: str):
    parts = [int(x) for x in s.split(",") if x.strip()]
    if len(parts) != 3 or sorted(parts) != parts:
        raise ValueError("splits must be three increasing integers, e.g., 10,20,30")
    return parts

# DHT를 통해 다른 네트워크에 존재하는 노드가 찾을 수 있도록 현재 노드의 IP 주소 저장
def _get_local_ip():
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) # 외부로 연결 시도
        ip = s.getsockname()[0] # 외부 연결에 실제로 사용된 인터페이스의 IP 주소를 반환
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@torch.inference_mode() # gradient 계산 비활성화하여 오버헤드 제거
def run_rank0(args, device, splits):
    """Run Stage0 (client side)."""

    # 1. Initialize
    full = load_stage_model(args.model, device, role="stage0", end=splits[0], dtype=args.dtype)
    s0 = Stage0(full, splits[0]).to(device) # load to GPU

    # connect to DHT Network with initial(stage1) DHT peer address
    dht_peers = args.dht_initial_peers.split(',') if args.dht_initial_peers else []
    
    # initialize RpcTransport for connection
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


    # 2. Input Process (prompt -> token)
    tok = AutoTokenizer.from_pretrained(args.model) # model에 맞는 tokenizer 로드
    prompt = args.prompt
    # tokenized된 token을 GPU에 로드
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # [1, L]
    L = input_ids.size(1)

    # 입력 토큰 차원에 맞게 attention mask, position embedding tensor 생성
    model_dtype = s0.embed_tokens.weight.dtype
    attn, pos = None, torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

    past0 = None # 첫 호출이므로 KV Cache 없음

    # forward pass
    hidden, past0 = s0(input_ids, pos, attn, past0, use_cache=True) # nn.Module.__call__ method -> forward pass

    # 3. Prefill (for initial input tokens)
    session_id = str(uuid4())
    max_length = L + args.max_new_tokens

    t_prefill_start = time.perf_counter()
    # send prefill hidden to rank1
    tx.send_prefill(L, hidden, session_id=session_id, max_length=max_length)  # [1,L,H]

    # load _last_token after prefill
    next_id = tx.recv_token()
    ttft = time.perf_counter() - t_prefill_start
    generated = [next_id]

    # metrics containers
    decode_total_times = []

    # 4. Decoding (for newly generated token)
    cur_len = L + 1
    t_decode_start = time.perf_counter()
    for _ in range(args.max_new_tokens - 1): # Prefill에서 1개 토큰 생성했으므로 max - 1 생성
        new_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
        # past cache 길이에 맞춰 attention mask/position_ids 생성
        past_len = past0[0][0].shape[-2] if past0 and past0[0] is not None else cur_len - 1
        attn = None
        pos = torch.tensor([[past_len]], device=device, dtype=torch.long)
        hidden, past0 = s0(new_input, pos, attn, past0, use_cache=True)  # [1,1,H]

        # send_prefill과 동일
        tx.send_decode_step(cur_len, hidden, session_id=session_id, max_length=max_length)  # [1,1,H]
        next_id = tx.recv_token()


        if tx.last_decode_total is not None:
            decode_total_times.append(tx.last_decode_total)
        generated.append(next_id)
        cur_len += 1

    total_decode_time = time.perf_counter() - t_decode_start
    total_dec_tokens = len(decode_total_times)
    decode_tps = (total_dec_tokens / total_decode_time) if total_decode_time > 0 else 0.0

    # 5. Result
    text = tok.decode(generated, skip_special_tokens=True)
    print("Generated:", text)

    # Log simple timing summary
    print("=== Timing Summary ===")
    print(f"TTFT: {ttft*1000:.2f} ms")
    if tx.last_prefill_stage_times:
        prefill_stage_ms = ", ".join(f"{k}={v*1000:.2f} ms" for k, v in tx.last_prefill_stage_times)
        print(f"Prefill stages: {prefill_stage_ms}")
    if tx.last_prefill_total is not None:
        print(f"Prefill total: {tx.last_prefill_total*1000:.2f} ms")
    print(f"Decode tokens: {total_dec_tokens}, decode throughput: {decode_tps:.2f} tok/s")
    if tx.decode_stage_history:
        agg = {}
        for stages in tx.decode_stage_history:
            for k, v in stages:
                agg.setdefault(k, []).append(v)
        stage_lines = []
        for k, vals in agg.items():
            avg = sum(vals) / len(vals)
            stage_lines.append(f"{k} avg={avg*1000:.2f} ms")
        print("Decode stages (avg): " + ", ".join(stage_lines))

    
    # Cleanup
    tx.shutdown()


@torch.inference_mode()
def run_stage_server(args, device, splits):
    """Run a server stage (1, 2, or 3)."""
    if args.stage == 1:
        start, end = splits[0], splits[1]
        full = load_stage_model(args.model, device, role="segment", start=start, end=end, dtype=args.dtype)
        stage_model = StageSegment(full, start, end).to(device)
        final_stage = False
    elif args.stage == 2:
        start, end = splits[1], splits[2]
        full = load_stage_model(args.model, device, role="segment", start=start, end=end, dtype=args.dtype)
        stage_model = StageSegment(full, start, end).to(device)
        final_stage = False
    elif args.stage == 3:
        start = splits[2]
        full = load_stage_model(args.model, device, role="last", start=start, dtype=args.dtype)
        stage_model = StageLast(full, start).to(device)
        final_stage = True
    else:
        raise ValueError("stage must be 1, 2, or 3 for server")

    dht_peers = args.dht_initial_peers.split(",") if args.dht_initial_peers else []
    initial_peers_list = _format_initial_peers(dht_peers)
    local_ip = _get_local_ip()

    # Initialize DHT Network
    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=[f"/ip4/{local_ip}/tcp/{args.dht_port}"],
    )

    # 초기화된 DHT 네트워크의 multiaddr 리스트 반환
    visible = dht.get_visible_maddrs()
    peer_id = str(dht.peer_id)
    if visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    else:
        fallback = [f"/ip4/{local_ip}/tcp/{args.dht_port}/p2p/{peer_id}"]
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
            p2p = await P2P.create(host_maddrs=[f"/ip4/{local_ip}/tcp/{args.rpc_port}"])
            logger.info(f"P2P initialized successfully, PeerID: {p2p.peer_id}")
            # get_visible_maddrs is async in this hivemind version
            visible_maddrs = await p2p.get_visible_maddrs() # P2P가 자동으로 감지한 외부 접근 가능한 multiaddr 리스트
            p2p_maddr = getattr(p2p, "daemon_listen_maddr", None) # P2P Daemon이 실제로 리스닝하는 주소
            p2p_maddrs = [str(m) for m in visible_maddrs] if visible_maddrs else []
            # visible_maddrs와 p2p_maddr를 병합
            if p2p_maddr:
                p2p_maddrs.append(str(p2p_maddr))
            if p2p_maddrs:
                logger.info(f"Stage{args.stage} P2P listen maddrs: {p2p_maddrs}")
            else:
                # Fallback to announced addr using rpc_port
                p2p_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}/p2p/{p2p.peer_id}"]
                logger.warning(f"Stage{args.stage} P2P listen maddrs unknown; using fallback {p2p_maddrs}")

            peer_info = {
                "peer_id": str(p2p.peer_id),
                "ip": local_ip,
                "rpc_port": args.rpc_port,
                "dht_port": args.dht_port,
                "timestamp": get_dht_time(),
            }
            # P2P Multiaddr 존재하면 peer_info에 추가
            if p2p_maddrs:
                peer_info["p2p_maddrs"] = p2p_maddrs

            # DHT Network에 rpc 통신에 필요한 정보 저장
            dht.store(f"mini_petals:stage{args.stage}", peer_info, expiration_time=get_dht_time() + 3600)

            # P2P Daemon에 StageConnectionHandler의 rpc_* 메서드 등록
            await handler.add_p2p_handlers(p2p)
            logger.info(f"Stage{args.stage} handlers registered, waiting for requests...")

            # 요청 들어올때까지 대기
            await asyncio.Event().wait()
        finally:
            handler.shutdown()
            if p2p is not None:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")
            dht.shutdown()

    asyncio.run(setup_and_run())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True,
                       help="Comma-separated cut points for 4-stage pipeline, e.g., 10,20,30")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                       help="Model dtype: fp16 (default), bf16, fp32")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for text generation")
    
    # RPC-related arguments
    parser.add_argument('--dht_initial_peers', type=str, default='', 
                       help='Comma-separated list of initial DHT peers (e.g., full multiaddrs /ip4/host/tcp/port/p2p/PeerID)')
    parser.add_argument('--dht_port', type=int, default=8000, help='DHT port')
    parser.add_argument('--rpc_port', type=int, default=8001, help='RPC port')
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2, 3],
                       help='Stage number (0=client, 1/2 mid, 3 final server)')
    parser.add_argument('--request_timeout', type=float, default=30.0,
                       help='Timeout for RPC requests in seconds')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (client->stage3)')
    parser.add_argument('--top_p', type=float, default=0.92, help='Nucleus sampling p')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (0=disabled)')
    
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

#atest