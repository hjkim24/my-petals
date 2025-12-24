import argparse
import asyncio
import os
from uuid import uuid4

import torch
from transformers import AutoTokenizer
import logging
from hivemind import DHT, get_dht_time
from hivemind.p2p import P2P
from hivemind.utils.logging import get_logger
from partition import load_partition, Stage0, Stage1
from rpc_transport import RpcTransport
from rpc_handler import Stage1ConnectionHandler

logger = get_logger(__name__)
# Ensure logs are emitted when running from terminal
logging.basicConfig(level=logging.INFO)


def build_masks(seq_len: int, device):
    # batch=1, no padding
    attn = torch.ones(1, seq_len, device=device, dtype=torch.float)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq]
    return attn, pos


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


def _get_local_ip():
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@torch.inference_mode()
def run_rank0(args, device):
    """Run Stage0 (client side)."""
    full = load_partition(args.model, args.split_layer, device)
    s0 = Stage0(full, args.split_layer).to(device)

    dht_peers = args.dht_initial_peers.split(',') if args.dht_initial_peers else []
    tx = RpcTransport(
        device=device,
        stage=0,
        dht_initial_peers=dht_peers,
        dht_port=args.dht_port,
        rpc_port=args.rpc_port,
    )

    # prompt
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = args.prompt
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # [1, L]
    L = input_ids.size(1)

    attn, pos = build_masks(L, device)

    past0 = None
    hidden, past0 = s0(input_ids, pos, attn, past0, use_cache=True)

    session_id = str(uuid4())
    max_length = L + args.max_new_tokens

    # send prefill hidden to rank1
    tx.send_prefill(L, hidden, session_id=session_id, max_length=max_length)  # [1,L,H]

    # receive first next_token_id (after stage1 prefill)
    next_id = tx.recv_token()
    generated = [next_id]

    # decode loop
    cur_len = L + 1
    for _ in range(args.max_new_tokens - 1):
        # new token
        new_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
        attn, pos = build_masks(cur_len, device)  # simple, batch=1
        # position_ids should be full length; many HF impl accept this
        hidden, past0 = s0(new_input, pos[:, -1:], attn, past0, use_cache=True)  # [1,1,H]

        tx.send_decode_step(cur_len, hidden, session_id=session_id, max_length=max_length)  # [1,1,H]
        next_id = tx.recv_token()
        generated.append(next_id)
        cur_len += 1

    text = tok.decode(generated, skip_special_tokens=True)
    print("Generated:", text)
    
    # Cleanup
    tx.shutdown()


@torch.inference_mode()
def run_rank1(args, device):
    """Run Stage1 (server side)."""
    full = load_partition(args.model, args.split_layer, device)
    s1 = Stage1(full, args.split_layer).to(device)

    dht_peers = args.dht_initial_peers.split(",") if args.dht_initial_peers else []
    initial_peers_list = _format_initial_peers(dht_peers)
    local_ip = _get_local_ip()

    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=[f"/ip4/{local_ip}/tcp/{args.dht_port}"],
    )
    visible = dht.get_visible_maddrs()
    peer_id = str(dht.peer_id)
    if visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    else:
        fallback = [f"/ip4/{local_ip}/tcp/{args.dht_port}/p2p/{peer_id}"]
        logger.info(
            f"DHT visible multiaddrs not available; try fallback: {fallback}"
        )

    handler = Stage1ConnectionHandler(
        dht=dht,
        stage1_model=s1,
        device=device,
        request_timeout=args.request_timeout,
    )

    async def setup_and_run():
        p2p = None
        try:
            logger.info("Initializing P2P for Stage1...")
            p2p = await P2P.create(host_maddrs=[f"/ip4/{local_ip}/tcp/{args.rpc_port}"])
            logger.info(f"P2P initialized successfully, PeerID: {p2p.peer_id}")
            # get_visible_maddrs is async in this hivemind version
            visible_maddrs = await p2p.get_visible_maddrs()
            p2p_maddr = getattr(p2p, "daemon_listen_maddr", None)
            p2p_maddrs = [str(m) for m in visible_maddrs] if visible_maddrs else []
            if p2p_maddr:
                p2p_maddrs.append(str(p2p_maddr))
            if p2p_maddrs:
                logger.info(f"Stage1 P2P listen maddrs: {p2p_maddrs}")
            else:
                # Fallback to announced addr using rpc_port
                p2p_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}/p2p/{p2p.peer_id}"]
                logger.warning(f"Stage1 P2P listen maddrs unknown; using fallback {p2p_maddrs}")

            peer_info = {
                "peer_id": str(p2p.peer_id),
                "ip": local_ip,
                "rpc_port": args.rpc_port,
                "dht_port": args.dht_port,
                "timestamp": get_dht_time(),
            }
            if p2p_maddrs:
                peer_info["p2p_maddrs"] = p2p_maddrs
            dht.store("mini_petals:stage1", peer_info, expiration_time=get_dht_time() + 3600)

            await handler.add_p2p_handlers(p2p)
            logger.info("Stage1 handlers registered, waiting for requests...")
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
    parser.add_argument("--split_layer", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for text generation")
    
    # RPC-related arguments
    parser.add_argument('--dht_initial_peers', type=str, default='', 
                       help='Comma-separated list of initial DHT peers (e.g., full multiaddrs /ip4/host/tcp/port/p2p/PeerID)')
    parser.add_argument('--dht_port', type=int, default=8000, help='DHT port')
    parser.add_argument('--rpc_port', type=int, default=8001, help='RPC port')
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1],
                       help='Stage number (0 for Stage0/client, 1 for Stage1/server)')
    parser.add_argument('--request_timeout', type=float, default=30.0,
                       help='Timeout for RPC requests in seconds')
    
    args = parser.parse_args()

    # Get device
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    
    if args.stage == 0:
        run_rank0(args, device)
    else:
        run_rank1(args, device)


if __name__ == "__main__":
    main()
