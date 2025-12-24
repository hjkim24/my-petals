"""
RPC transport using hivemind P2P for Petals-like distributed inference.
Uses P2P.get_stub() for client-side RPC calls and ConnectionHandler for server-side.
"""
import asyncio
import concurrent.futures
import socket
import time
from typing import List, Optional, Tuple

import torch
from hivemind import DHT, MSGPackSerializer, PeerID, get_dht_time, serialize_torch_tensor
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.control import MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class RpcTransport:
    """RPC-based transport using hivemind P2P for distributed inference.
    
    Client side (stage 0): Uses P2P.get_stub() to get stub and call remote RPC methods.
    Server side (stage 1): Uses ConnectionHandler (separate file: rpc_handler.py).
    """

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
        
        # Get local IP address
        self.local_ip = self._get_local_ip()
        
        # Convert initial_peers to proper format
        initial_peers_list = []
        for peer in dht_initial_peers:
            if ':' in peer:
                ip, port = peer.split(':')
                initial_peers_list.append(f"/ip4/{ip}/tcp/{port}")
            else:
                initial_peers_list.append(peer)
        
        # Initialize DHT
        logger.info(f"Initializing DHT on {self.local_ip}:{dht_port}")
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers_list if initial_peers_list else None,
            host_maddrs=[f"/ip4/{self.local_ip}/tcp/{dht_port}"],
        )
        
        # Register this peer in DHT
        self._register_peer()
        
        # Initialize P2P (for client-side RPC calls)
        if stage == 0:
            logger.info(f"Initializing P2P on {self.local_ip}:{rpc_port}")
            self.p2p = P2P(
                initial_peers=initial_peers_list if initial_peers_list else None,
                dht=self.dht,
                host_maddrs=[f"/ip4/{self.local_ip}/tcp/{rpc_port}"],
            )
            self.p2p.start()
            
            # Update DHT with our PeerID
            peer_info = {
                "ip": self.local_ip,
                "rpc_port": self.rpc_port,
                "dht_port": self.dht_port,
                "stage": self.stage,
                "peer_id": str(self.p2p.peer_id),
                "timestamp": get_dht_time(),
            }
            key = f"petals_peer_stage_{self.stage}"
            self.dht.store(key, peer_info, expiration_time=get_dht_time() + 3600)
            
            # Find peer and create stub
            self.peer_id = self._find_peer()
            self.stub = self.p2p.get_stub(self.peer_id)
        else:
            # Server side: P2P and handler are managed separately
            self.p2p = None
            self.stub = None
            self.peer_id = None
        
        logger.info(f"RpcTransport initialized: stage={stage}, peer_id={self.peer_id}")
    
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
    
    def _register_peer(self):
        """Register this peer's information in DHT."""
        # Get our own PeerID from P2P (if available) or generate/store it
        peer_id_str = None
        if self.stage == 0 and hasattr(self, 'p2p') and self.p2p is not None:
            # For client, P2P is already initialized
            peer_id_str = str(self.p2p.peer_id)
        elif self.stage == 1:
            # For server, we'll store it after P2P is created in run_rank1
            # For now, we'll update it later
            pass
        
        peer_info = {
            "ip": self.local_ip,
            "rpc_port": self.rpc_port,
            "dht_port": self.dht_port,
            "stage": self.stage,
            "timestamp": get_dht_time(),
        }
        if peer_id_str:
            peer_info["peer_id"] = peer_id_str
        
        key = f"petals_peer_stage_{self.stage}"
        self.dht.store(key, peer_info, expiration_time=get_dht_time() + 3600)
        logger.info(f"Registered peer info: {peer_info}")
    
    def _find_peer(self, max_retries: int = 10, retry_delay: float = 1.0) -> PeerID:
        """Find the peer for the other stage via DHT."""
        target_stage = 1 - self.stage  # 0 -> 1, 1 -> 0
        key = f"petals_peer_stage_{target_stage}"
        
        for attempt in range(max_retries):
            try:
                result = self.dht.get(key, latest=True)
                if result is not None and result.value is not None:
                    peer_info = result.value
                    # Get PeerID from DHT - store it when registering
                    peer_id_str = peer_info.get("peer_id")
                    if peer_id_str:
                        peer_id = PeerID.from_base58(peer_id_str)
                        logger.info(f"Found peer: {peer_id}")
                        return peer_id
                    else:
                        # If peer_id not in DHT, we need to get it from P2P
                        # Try to find peer by multiaddr
                        multiaddr = f"/ip4/{peer_info['ip']}/tcp/{peer_info['rpc_port']}"
                        # Use P2P to find peer - this requires P2P to be running
                        # For now, we'll wait and retry
                        logger.warning(f"PeerID not found in DHT for {multiaddr}, retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to find peer failed: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        raise RuntimeError(f"Could not find peer for stage {target_stage} after {max_retries} attempts")
    
    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use ThreadPoolExecutor to run in a new thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(coro)
    
    async def _forward_unary(
        self, uid: str, serialized_tensors: list, metadata: bytes, timeout: float
    ) -> int:
        """Unary RPC call for forward."""
        request = runtime_pb2.ExpertRequest(
            uid=uid,
            tensors=serialized_tensors,
            metadata=metadata,
        )
        response = await self.stub.rpc_forward(request, timeout=timeout)
        
        # Parse response to get token ID
        response_metadata = MSGPackSerializer.loads(response.metadata) if response.metadata else {}
        token_id = response_metadata.get("token_id")
        
        if token_id is None:
            # Fallback: deserialize token from tensor
            if response.tensors:
                token_tensor = deserialize_torch_tensor(response.tensors[0])
                token_id = int(token_tensor.item())
        
        return token_id
    
    async def _forward_stream(
        self, uid: str, serialized_tensors: list, metadata: bytes, timeout: float
    ) -> int:
        """Stream RPC call for forward (for large payloads)."""
        # Split tensors for streaming
        parts = (
            runtime_pb2.ExpertRequest(uid=uid, tensors=[part], metadata=metadata)
            for tensor in serialized_tensors
            for part in split_for_streaming(tensor, MAX_UNARY_PAYLOAD_SIZE)
        )
        
        # Call stream RPC
        outputs = await asyncio.wait_for(
            self.stub.rpc_forward_stream(iter_as_aiter(parts)), timeout=timeout
        )
        outputs = aiter_with_timeout(outputs, timeout)
        
        # Deserialize response
        response_tensors = await deserialize_tensor_stream(msg.tensors async for msg in outputs)
        
        if response_tensors:
            token_tensor = response_tensors[0]
            token_id = int(token_tensor.item())
        else:
            raise ValueError("No response tensors received from stream")
        
        return token_id
    # TODO: implement backward unary/stream for fine-tuning
    
    def send_prefill(self, L: int, hidden: torch.Tensor):
        """Send prefill hidden states to peer (stage0 -> stage1)."""
        if self.stage != 0:
            raise RuntimeError("send_prefill should only be called by stage0")
        
        async def _send():
            # Serialize tensor
            hidden_cpu = hidden.cpu().detach()
            serialized = serialize_torch_tensor(hidden_cpu, compression=None)
            
            # Create metadata
            metadata = MSGPackSerializer.dumps({
                "seq_len": L,
                "is_prefill": True,
            })
            
            # Choose unary or stream based on size
            size = hidden_cpu.element_size() * hidden_cpu.nelement()
            # Use // 2 since hivemind serializes bfloat16 tensors in float32, so they take 2x more space
            forward_fn = self._forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._forward_unary
            
            token_id = await forward_fn("stage1", [serialized], metadata, self.timeout)
            return token_id
        
        # Store token for recv_token to retrieve
        self._last_token = self._run_async(_send())
    
    def recv_prefill(self, H: int, dtype: torch.dtype) -> Tuple[int, torch.Tensor]:
        """Receive prefill hidden states from peer (stage1 receives from stage0).
        
        Note: This is handled by the ConnectionHandler on the server side.
        This method should not be called directly in RPC mode.
        """
        raise NotImplementedError("recv_prefill is handled by ConnectionHandler on server side")
    
    def send_decode_step(self, cur_len: int, hidden: torch.Tensor):
        """Send decode step hidden states to peer (stage0 -> stage1)."""
        if self.stage != 0:
            raise RuntimeError("send_decode_step should only be called by stage0")
        
        async def _send():
            # Serialize tensor
            hidden_cpu = hidden.cpu().detach()
            serialized = serialize_torch_tensor(hidden_cpu, compression=None)
            
            # Create metadata
            metadata = MSGPackSerializer.dumps({
                "seq_len": 1,  # Decode step processes one token
                "cur_len": cur_len,
                "is_prefill": False,
            })
            
            # Choose unary or stream based on size
            # Decode step typically has small tensors (1 token), so usually unary
            size = hidden_cpu.element_size() * hidden_cpu.nelement()
            forward_fn = self._forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._forward_unary
            
            token_id = await forward_fn("stage1", [serialized], metadata, self.timeout)
            return token_id
        
        # Store token for recv_token to retrieve
        self._last_token = self._run_async(_send())
    
    def recv_decode_step(self, H: int, dtype: torch.dtype) -> Tuple[int, torch.Tensor]:
        """Receive decode step hidden states from peer (stage1 receives from stage0).
        
        Note: This is handled by the ConnectionHandler on the server side.
        This method should not be called directly in RPC mode.
        """
        raise NotImplementedError("recv_decode_step is handled by ConnectionHandler on server side")
    
    def send_token(self, token_id: int):
        """Send generated token to peer (stage1 -> stage0).
        
        Note: In RPC mode, token is returned in the response, not sent separately.
        This method is a no-op for RPC mode.
        """
        if self.stage != 1:
            raise RuntimeError("send_token should only be called by stage1")
        # In RPC mode, token is returned in rpc_forward response
        # This method is kept for interface compatibility but does nothing
        pass
    
    def recv_token(self) -> int:
        """Receive token from peer (stage0 receives from stage1)."""
        if self.stage != 0:
            raise RuntimeError("recv_token should only be called by stage0")
        
        if not hasattr(self, '_last_token'):
            raise RuntimeError("No token received. Call send_prefill or send_decode_step first.")
        
        token_id = self._last_token
        delattr(self, '_last_token')
        return token_id
    
    def shutdown(self):
        """Shutdown P2P and DHT."""
        if self.p2p is not None:
            self.p2p.shutdown()
        if hasattr(self, 'dht'):
            self.dht.shutdown()
