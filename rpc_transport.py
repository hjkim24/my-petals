"""
RPC transport using hivemind P2P for Petals-like distributed inference.
Uses P2P protobuf handlers for client-side RPC calls.
"""
import asyncio
import concurrent.futures
import socket
from typing import List, Optional

import torch
from hivemind import DHT, MSGPackSerializer, PeerID, serialize_torch_tensor
from hivemind.compression.serialization import deserialize_torch_tensor
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

logger = get_logger(__name__)


class RpcTransport:
    """RPC-based transport using hivemind P2P for distributed inference (client side)."""

    def __init__(
        self,
        device: torch.device,
        stage: int,
        dht_initial_peers: List[str],
        dht_port: int = 8000,
        rpc_port: int = 8001,
        timeout: float = 30.0,
    ):
        """
        Args:
            device: PyTorch device (cuda:0, cuda:1, etc.)
            stage: Current stage (0 for Stage0, 1 for Stage1)
            dht_initial_peers: List of initial DHT peers (e.g., ["ip1:port1", "ip2:port2"])
            dht_port: Port for DHT
            rpc_port: Port for RPC server
            timeout: Timeout for RPC calls in seconds
        """
        self.device = device
        self.stage = stage
        self.dht_port = dht_port
        self.rpc_port = rpc_port
        self.timeout = timeout
        self.local_ip = self._get_local_ip()

        self._last_token: Optional[int] = None
        self.remote_peer_id: Optional[PeerID] = None
        self.remote_maddrs = []

        initial_peers_list = self._format_initial_peers(dht_initial_peers)

        logger.info(f"Initializing DHT on {self.local_ip}:{dht_port}")
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers_list if initial_peers_list else None,
            # Use default host/announce to avoid strict multiaddr parsing issues
        )

        self.p2p: Optional[P2P] = None
        self.peer_id: Optional[PeerID] = None

        if self.stage == 0:
            self.p2p = self._run_async(self._create_p2p())
            self.peer_id = self.p2p.peer_id
        else:
            logger.info("Stage1 server side does not rely on RpcTransport; only client helpers are active")

        logger.info(f"RpcTransport initialized: stage={stage}, peer_id={self.peer_id}")

    def _format_initial_peers(self, dht_initial_peers: List[str]) -> List[str]:
        initial_peers_list = []
        for peer in dht_initial_peers:
            peer = peer.strip()
            if not peer:
                continue
            # Require full multiaddr with peer ID to avoid invalid p2p multiaddr errors
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

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    async def _create_p2p(self) -> P2P:
        # Use default listen/announce addrs to avoid multiaddr parsing issues across platforms
        return await P2P.create()

    async def _discover_stage1_peer(self, max_retries: int = 10, retry_delay: float = 1.0) -> PeerID:
        """Find Stage1 server peer_id via DHT."""
        key = "mini_petals:stage1"
        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                result = await loop.run_in_executor(None, lambda: self.dht.get(key, latest=True))
                if result is not None and result.value is not None:
                    peer_id_str = result.value.get("peer_id")
                    if peer_id_str:
                        peer_id = PeerID.from_base58(peer_id_str)
                        maddrs = result.value.get("p2p_maddrs") or []
                        self.remote_maddrs = maddrs
                        logger.info(f"Found stage1 peer: {peer_id}, maddrs={maddrs}")
                        return peer_id
                    logger.warning("Stage1 peer info missing peer_id, retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to find stage1 peer failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        raise RuntimeError("Could not find stage1 peer via DHT key 'mini_petals:stage1'")

    async def _ensure_ready(self):
        if self.stage != 0:
            raise RuntimeError("RpcTransport client helpers should only be used on stage0")
        if self.p2p is None:
            self.p2p = await self._create_p2p()
            self.peer_id = self.p2p.peer_id
        if self.remote_peer_id is None:
            self.remote_peer_id = await self._discover_stage1_peer()
            if self.remote_maddrs:
                try:
                    from multiaddr import Multiaddr
                    # Filter out unix/unsupported addrs; keep tcp/quic/ip4/ip6
                    filtered = []
                    for m in self.remote_maddrs:
                        try:
                            base = m.split("/p2p/")[0]  # strip peer component if present
                            ma = Multiaddr(base)
                            if ma.protocols()[0].name in ("ip4", "ip6"):
                                filtered.append(ma)
                            elif ma.protocols()[0].name in ("tcp", "quic"):
                                filtered.append(ma)
                            else:
                                logger.debug(f"Skipping non-tcp/ip multiaddr {m}")
                        except Exception as e:
                            logger.debug(f"Failed to parse multiaddr {m}: {e}")
                    if filtered:
                        await self.p2p._client.connect(self.remote_peer_id, filtered)
                        logger.info(f"Connected to stage1 peer via maddrs: {self.remote_maddrs}")
                    else:
                        logger.warning(f"No usable tcp/ip multiaddrs to connect: {self.remote_maddrs}")
                except Exception as e:
                    logger.warning(f"Could not connect to stage1 via maddrs {self.remote_maddrs}: {e}")
            else:
                logger.warning("Stage1 p2p_maddrs missing in DHT; relying on auto peer discovery")
            try:
                await asyncio.wait_for(self.p2p.wait_for_at_least_n_peers(1), timeout=5)
                logger.info(f"P2P connected peers: {await self.p2p.list_peers()}")
            except Exception as e:
                logger.warning(f"P2P did not see remote peer after connect attempt: {e}")

    def _extract_token_id(self, response: runtime_pb2.ExpertResponse) -> Optional[int]:
        metadata = MSGPackSerializer.loads(response.metadata) if response.metadata else {}
        token_id = metadata.get("token_id")
        if token_id is not None:
            return int(token_id)
        if response.tensors:
            try:
                return int(deserialize_torch_tensor(response.tensors[0]).item())
            except Exception as e:
                logger.warning(f"Failed to deserialize token tensor: {e}")
        return None

    async def _forward_unary(
        self, uid: str, serialized_tensors: list, metadata: bytes, timeout: float
    ) -> int:
        """Unary RPC call for forward."""
        await self._ensure_ready()
        request = runtime_pb2.ExpertRequest(uid=uid, tensors=serialized_tensors, metadata=metadata)
        response = await asyncio.wait_for(
            self.p2p.call_protobuf_handler(
                self.remote_peer_id,
                "Stage1ConnectionHandler.rpc_forward",
                request,
                runtime_pb2.ExpertResponse,
            ),
            timeout=timeout,
        )

        token_id = self._extract_token_id(response)
        if token_id is None:
            raise ValueError("rpc_forward returned no token")
        return token_id

    async def _forward_stream(
        self, uid: str, serialized_tensors: list, metadata: bytes, timeout: float
    ) -> int:
        """Stream RPC call for forward (for large payloads)."""
        await self._ensure_ready()
        parts = (
            runtime_pb2.ExpertRequest(uid=uid, tensors=[part], metadata=metadata)
            for tensor in serialized_tensors
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        )

        outputs = self.p2p.iterate_protobuf_handler(
            self.remote_peer_id,
            "Stage1ConnectionHandler.rpc_forward_stream",
            iter_as_aiter(parts),
            runtime_pb2.ExpertResponse,
        )

        token_id: Optional[int] = None
        fallback_tensors = []
        async for response in aiter_with_timeout(outputs, timeout):
            token_id = self._extract_token_id(response)
            if token_id is not None:
                break
            fallback_tensors.extend(response.tensors)

        if token_id is None:
            if fallback_tensors:
                token_id = int(deserialize_torch_tensor(fallback_tensors[0]).item())
            else:
                raise ValueError("rpc_forward_stream returned no token")

        return token_id

    def send_prefill(self, L: int, hidden: torch.Tensor, session_id: str, max_length: int):
        """Send prefill hidden states to peer (stage0 -> stage1)."""
        if self.stage != 0:
            raise RuntimeError("send_prefill should only be called by stage0")

        async def _send():
            hidden_cpu = hidden.cpu().detach()
            serialized = serialize_torch_tensor(hidden_cpu)
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": L,
                    "cur_len": L,
                    "is_prefill": True,
                    "max_length": max_length,
                }
            )

            size = hidden_cpu.element_size() * hidden_cpu.nelement()
            forward_fn = self._forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._forward_unary

            token_id = await forward_fn("stage1", [serialized], metadata, self.timeout)
            return token_id

        self._last_token = self._run_async(_send())

    def send_decode_step(self, cur_len: int, hidden: torch.Tensor, session_id: str, max_length: int):
        """Send decode step hidden states to peer (stage0 -> stage1)."""
        if self.stage != 0:
            raise RuntimeError("send_decode_step should only be called by stage0")

        async def _send():
            hidden_cpu = hidden.cpu().detach()
            serialized = serialize_torch_tensor(hidden_cpu)
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": 1,
                    "cur_len": cur_len,
                    "is_prefill": False,
                    "max_length": max_length,
                }
            )

            size = hidden_cpu.element_size() * hidden_cpu.nelement()
            forward_fn = self._forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._forward_unary

            token_id = await forward_fn("stage1", [serialized], metadata, self.timeout)
            return token_id

        self._last_token = self._run_async(_send())

    def send_token(self, token_id: int):
        """Send generated token to peer (stage1 -> stage0).

        Note: In RPC mode, token is returned in the response, not sent separately.
        This method is a no-op for RPC mode.
        """
        if self.stage != 1:
            raise RuntimeError("send_token should only be called by stage1")
        pass

    def recv_token(self) -> int:
        """Receive token from peer (stage0 receives from stage1)."""
        if self.stage != 0:
            raise RuntimeError("recv_token should only be called by stage0")

        if self._last_token is None:
            raise RuntimeError("No token received. Call send_prefill or send_decode_step first.")

        token_id = self._last_token
        self._last_token = None
        return token_id

    def shutdown(self):
        """Shutdown P2P and DHT."""
        if self.p2p is not None:
            try:
                self._run_async(self.p2p.shutdown())
            except Exception as e:
                logger.warning(f"Error shutting down P2P: {e}")
        if hasattr(self, "dht") and self.dht is not None:
            try:
                self.dht.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down DHT: {e}")
