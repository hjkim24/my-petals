import argparse
import asyncio
import os
import torch
from transformers import AutoTokenizer
from hivemind import get_dht_time
from hivemind.p2p import P2P
from partition import load_partition, Stage0, Stage1
from rpc_transport import RpcTransport
from rpc_handler import Stage1ConnectionHandler


# --- Transport abstraction ---

class Transport:
    """A minimal interface for stage-to-stage communication."""

    def send_prefill(self, L: int, hidden: torch.Tensor):
        raise NotImplementedError

    def recv_prefill(self, H: int, dtype: torch.dtype):
        raise NotImplementedError

    def send_decode_step(self, cur_len: int, hidden: torch.Tensor):
        raise NotImplementedError

    def recv_decode_step(self, H: int, dtype: torch.dtype):
        raise NotImplementedError

    def send_token(self, token_id: int):
        raise NotImplementedError

    def recv_token(self) -> int:
        raise NotImplementedError


def build_masks(seq_len: int, device):
    # batch=1, no padding
    attn = torch.ones(1, seq_len, device=device, dtype=torch.long)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)  # [1, seq]
    return attn, pos


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
    prompt = "Hello, how are you?"
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)  # [1, L]
    L = input_ids.size(1)

    attn, pos = build_masks(L, device)

    past0 = None
    hidden, past0 = s0(input_ids, pos, attn, past0, use_cache=True)

    # send prefill hidden to rank1
    tx.send_prefill(L, hidden)  # [1,L,H]

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

        tx.send_decode_step(cur_len, hidden)  # [1,1,H]
        next_id = tx.recv_token()
        generated.append(next_id)
        cur_len += 1

    text = tok.decode(generated, skip_special_tokens=True)
    print("Generated:", text)
    
    # Cleanup
    tx.shutdown()


async def run_rank1_async(args, device, handler):
    """Run Stage1 server (async)."""
    # Wait for requests - the handler will process them
    # This is a simple implementation - in practice, you'd have a proper server loop
    try:
        # Keep the server running
        await asyncio.sleep(3600)  # Run for 1 hour or until interrupted
    except KeyboardInterrupt:
        pass
    finally:
        handler.shutdown()


@torch.inference_mode()
def run_rank1(args, device):
    """Run Stage1 (server side)."""
    full = load_partition(args.model, args.split_layer, device)
    s1 = Stage1(full, args.split_layer).to(device)

    dht_peers = args.dht_initial_peers.split(',') if args.dht_initial_peers else []
    
    # Initialize RPC transport for DHT and peer registration
    tx = RpcTransport(
        device=device,
        stage=1,
        dht_initial_peers=dht_peers,
        dht_port=args.dht_port,
        rpc_port=args.rpc_port,
    )
    
    # Create connection handler
    handler = Stage1ConnectionHandler(
        dht=tx.dht,
        stage1_model=s1,
        device=device,
        request_timeout=args.request_timeout,
    )
    
    # Start P2P server
    initial_peers_list = []
    for peer in dht_peers:
        if ':' in peer:
            ip, port = peer.split(':')
            initial_peers_list.append(f"/ip4/{ip}/tcp/{port}")
        else:
            initial_peers_list.append(peer)
    
    p2p = P2P(
        initial_peers=initial_peers_list if initial_peers_list else None,
        dht=tx.dht,
        host_maddrs=[f"/ip4/{tx.local_ip}/tcp/{args.rpc_port}"],
    )
    p2p.start()
    
    # Update DHT with our PeerID
    peer_info = {
        "ip": tx.local_ip,
        "rpc_port": args.rpc_port,
        "dht_port": args.dht_port,
        "stage": 1,
        "peer_id": str(p2p.peer_id),
        "timestamp": get_dht_time(),
    }
    key = "petals_peer_stage_1"
    tx.dht.store(key, peer_info, expiration_time=get_dht_time() + 3600)
    
    # Add P2P handlers
    async def setup_and_run():
        await handler.add_p2p_handlers(p2p)
        await run_rank1_async(args, device, handler)
    
    # Run async server
    try:
        asyncio.run(setup_and_run())
    except KeyboardInterrupt:
        pass
    finally:
        handler.shutdown()
        p2p.shutdown()
        tx.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split_layer", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    
    # RPC-related arguments
    parser.add_argument('--dht_initial_peers', type=str, default='', 
                       help='Comma-separated list of initial DHT peers (e.g., "ip1:port1,ip2:port2")')
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
