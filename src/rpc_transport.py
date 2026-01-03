"""
RPC transport using hivemind P2P for Petals-like distributed inference.
Uses P2P protobuf handlers for client-side RPC calls.
"""
import asyncio
import concurrent.futures
import socket
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from hivemind import DHT, PeerID, serialize_torch_tensor, MSGPackSerializer
from hivemind.compression.serialization import deserialize_torch_tensor
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
try:
    from hivemind.p2p.p2p_daemon_bindings.utils import P2PDaemonError, P2PHandlerError
except ImportError:
    # Fallback for different hivemind versions
    from hivemind.p2p.p2p_daemon_bindings import P2PDaemonError, P2PHandlerError
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
        temperature: float = 1.0,
        top_p: float = 0.92,
        top_k: int = 50,
        stage_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            device: PyTorch device (cuda:0, cuda:1, etc.)
            stage: Current stage (0 for Stage0, 1 for Stage1)
            dht_initial_peers: List of initial DHT peers (e.g., ["ip1:port1", "ip2:port2"])
            dht_port: Port for DHT
            rpc_port: Port for RPC server
            timeout: Timeout for RPC calls in seconds
            stage_keys: Ordered list of stage DHT keys (excluding stage0). Default: stage1, stage2, stage3
        """
        self.device = device
        self.stage = stage
        self.dht_port = dht_port
        self.rpc_port = rpc_port
        self.timeout = timeout
        self.local_ip = self._get_local_ip()
        self.stage_keys = stage_keys or ["mini_petals:stage1", "mini_petals:stage2", "mini_petals:stage3"]
        self.sampling = {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)}

        self._last_token: Optional[int] = None # 마지막으로 받은 토큰 ID
        self.remote_info: Dict[str, Dict] = {} # 원격 스테이지 연결 정보
        self.last_prefill_stage_times: List[Tuple[str, float]] = [] # Prefill 단계 Stage별 소요 시간
        self.last_prefill_total: Optional[float] = None # Prefill 단계 전체 소요 시간
        self.last_decode_stage_times: List[Tuple[str, float]] = [] # 마지막 Decode Step에서 Stage별 소요 시간
        self.last_decode_total: Optional[float] = None # 마지막 Decode Step 전체 소요 시간
        self.decode_stage_history: List[List[Tuple[str, float]]] = [] # 모든 Decode Step의 Stage별 소요 시간 히스토리
        self.decode_total_times: List[float] = [] # 모든 Decode Step의 전체 소요 시간 리스트
        
        # Fault tolerance: Client-side cache (Petals 논문 방식)
        # 각 stage로 보낸 과거 입력(hidden states)을 저장하여 서버 실패 시 복구에 사용
        # Session별로 분리하여 여러 session 동시 처리 지원
        self.client_cache: Dict[str, Dict[str, List[torch.Tensor]]] = {}  # {stage_key: {session_id: [past_inputs]}}
        self.failed_stages: Set[str] = set()  # 실패한 stage 추적
        self.failed_peers: Dict[str, Set[str]] = {}  # {stage_key: {failed_peer_id, ...}} - 실패한 서버 추적

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
            logger.info("Server stages do not rely on RpcTransport; only client helpers are active")

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

    async def _discover_peer(
        self, 
        stage_key: str, 
        max_retries: int = 10, 
        retry_delay: float = 1.0,
        exclude_peer_ids: Optional[Set[str]] = None
    ) -> Tuple[PeerID, List[str]]:
        """
        Find server peer_id via DHT for a given stage key.
        
        Args:
            stage_key: DHT key for the stage (e.g., "mini_petals:stage2")
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            exclude_peer_ids: Set of peer IDs to exclude (failed servers)
        """
        if exclude_peer_ids is None:
            exclude_peer_ids = set()
        
        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                # DHT에서 여러 후보를 가져오기 위해 latest=False로 시도
                # 또는 여러 번 시도하여 다른 서버 찾기
                result = await loop.run_in_executor(None, lambda: self.dht.get(stage_key, latest=True))
                if result is not None and result.value is not None:
                    entry = result.value
                    if isinstance(entry, dict):
                        peer_id_str = entry.get("peer_id")
                        if peer_id_str:
                            # 실패한 서버는 제외
                            if peer_id_str in exclude_peer_ids:
                                logger.warning(f"{stage_key} found excluded peer {peer_id_str[:8]}..., retrying...")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay)
                                    continue
                            
                            peer_id = PeerID.from_base58(peer_id_str)
                            maddrs = entry.get("p2p_maddrs") or []
                            return peer_id, maddrs
                    logger.warning(f"{stage_key} missing peer info, retrying...")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to find peer for {stage_key} failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        raise RuntimeError(f"Could not find peer via DHT key '{stage_key}' (excluded {len(exclude_peer_ids)} failed peers)")

    async def _ensure_ready(self, stage_key: str):
        if self.stage != 0:
            raise RuntimeError("RpcTransport client helpers should only be used on stage0")
        if self.p2p is None:
            self.p2p = await self._create_p2p()
            self.peer_id = self.p2p.peer_id
        info = self.remote_info.get(stage_key)
        if info is None:
            peer_id, maddrs = await self._discover_peer(stage_key)
            info = {"peer_id": peer_id, "maddrs": maddrs}
            self.remote_info[stage_key] = info
            if maddrs:
                try:
                    from multiaddr import Multiaddr
                    filtered = []
                    for m in maddrs:
                        try:
                            base = m.split("/p2p/")[0]
                            ma = Multiaddr(base)
                            if ma.protocols()[0].name in ("ip4", "ip6", "tcp", "quic"):
                                filtered.append(ma)
                            else:
                                logger.debug(f"Skipping non-tcp/ip multiaddr {m}")
                        except Exception as e:
                            logger.debug(f"Failed to parse multiaddr {m}: {e}")
                    if filtered:
                        await self.p2p._client.connect(peer_id, filtered)
                        logger.info(f"Connected to {stage_key} via maddrs: {maddrs}")
                    else:
                        logger.warning(f"No usable tcp/ip multiaddrs to connect: {maddrs}")
                except Exception as e:
                    logger.warning(f"Could not connect to {stage_key} via maddrs {maddrs}: {e}")
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

    async def _call_stage_unary(
        self, stage_key: str, serialized_tensors: list, metadata: bytes, timeout: float, expect_hidden: bool
    ):
        """Unary RPC call for a given stage."""
        await self._ensure_ready(stage_key)
        peer_id = self.remote_info[stage_key]["peer_id"]
        request = runtime_pb2.ExpertRequest(uid=stage_key, tensors=serialized_tensors, metadata=metadata)
        response = await asyncio.wait_for(
            self.p2p.call_protobuf_handler(
                peer_id,
                "StageConnectionHandler.rpc_forward",
                request,
                runtime_pb2.ExpertResponse,
            ),
            timeout=timeout,
        )

        if expect_hidden:
            if not response.tensors:
                raise ValueError(f"{stage_key} returned no tensors")
            return deserialize_torch_tensor(response.tensors[0])
        token_id = self._extract_token_id(response)
        if token_id is None:
            raise ValueError(f"{stage_key} returned no token")
        return token_id

    async def _call_stage_stream(
        self, stage_key: str, serialized_tensors: list, metadata: bytes, timeout: float, expect_hidden: bool
    ):
        """Stream RPC call for a given stage."""
        await self._ensure_ready(stage_key)
        peer_id = self.remote_info[stage_key]["peer_id"]
        parts = (
            runtime_pb2.ExpertRequest(uid=stage_key, tensors=[part], metadata=metadata)
            for tensor in serialized_tensors
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        )

        outputs = self.p2p.iterate_protobuf_handler(
            peer_id,
            "StageConnectionHandler.rpc_forward_stream",
            iter_as_aiter(parts),
            runtime_pb2.ExpertResponse,
        )

        tensors = []
        token_id: Optional[int] = None
        async for response in aiter_with_timeout(outputs, timeout):
            if expect_hidden:
                tensors.extend(response.tensors)
            else:
                token_id = self._extract_token_id(response)
                if token_id is not None:
                    break
                tensors.extend(response.tensors)

        if expect_hidden:
            if not tensors:
                raise ValueError(f"{stage_key} stream returned no tensors")
            return deserialize_torch_tensor(tensors[0])
        else:
            if token_id is None:
                if tensors:
                    token_id = int(deserialize_torch_tensor(tensors[0]).item())
                else:
                    raise ValueError(f"{stage_key} stream returned no token")
            return token_id

    async def _call_stage_with_recovery(
        self,
        stage_key: str,
        serialized_tensors: list,
        metadata: bytes,
        timeout: float,
        expect_hidden: bool,
        is_replay: bool = False,
        session_id: Optional[str] = None,
    ):
        """
        RPC 호출 + 실패 시 자동 복구 (Petals fault tolerance).
        
        Args:
            is_replay: True면 replay 모드 (복구 중), 실패 시 재시도 안 함
            session_id: Replay 시 필요
        """
        max_recovery_attempts = 3
        
        for attempt in range(max_recovery_attempts):
            try:
                # 기존 RPC 호출
                # serialized_tensors[0]는 runtime_pb2.Tensor protobuf 메시지이므로 크기 계산 방법이 다름
                if serialized_tensors:
                    first_tensor = serialized_tensors[0]
                    # runtime_pb2.Tensor인 경우 ByteSize() 사용, bytes인 경우 len() 사용
                    if hasattr(first_tensor, 'ByteSize'):
                        size = first_tensor.ByteSize()
                    elif isinstance(first_tensor, bytes):
                        size = len(first_tensor)
                    else:
                        # fallback: SerializeToString()으로 변환 후 크기 확인
                        try:
                            size = len(first_tensor.SerializeToString())
                        except AttributeError:
                            size = 0
                else:
                    size = 0
                forward_fn = self._call_stage_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._call_stage_unary
                return await forward_fn(stage_key, serialized_tensors, metadata, timeout, expect_hidden)
            except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError, P2PDaemonError, P2PHandlerError) as e:
                if is_replay:
                    # Replay 중 실패는 복구 불가
                    logger.error(f"Replay failed for {stage_key}: {e}")
                    raise
                
                logger.warning(f"Stage {stage_key} failed (attempt {attempt+1}/{max_recovery_attempts}): {e}")
                
                # 실패한 stage 기록
                self.failed_stages.add(stage_key)
                
                # 실패한 peer_id 기록
                if stage_key in self.remote_info:
                    failed_peer_id = str(self.remote_info[stage_key].get("peer_id", ""))
                    if stage_key not in self.failed_peers:
                        self.failed_peers[stage_key] = set()
                    self.failed_peers[stage_key].add(failed_peer_id)
                    logger.info(f"Marked peer {failed_peer_id[:8]}... as failed for {stage_key}")
                
                # 새 서버 찾기 (실패한 서버 제외)
                try:
                    exclude_peers = self.failed_peers.get(stage_key, set())
                    new_peer_id, new_maddrs = await self._discover_peer(
                        stage_key, 
                        max_retries=5, 
                        retry_delay=0.5,
                        exclude_peer_ids=exclude_peers
                    )
                    
                    # remote_info 업데이트
                    self.remote_info[stage_key] = {
                        "peer_id": new_peer_id,
                        "maddrs": new_maddrs
                    }
                    
                    # P2P 연결 재설정
                    if self.p2p and new_maddrs:
                        try:
                            from multiaddr import Multiaddr
                            filtered = []
                            for m in new_maddrs:
                                try:
                                    base = m.split("/p2p/")[0]
                                    ma = Multiaddr(base)
                                    if ma.protocols()[0].name in ("ip4", "ip6", "tcp", "quic"):
                                        filtered.append(ma)
                                except Exception:
                                    pass
                            if filtered:
                                await self.p2p._client.connect(new_peer_id, filtered)
                                logger.info(f"Reconnected to new {stage_key} server")
                        except Exception as conn_e:
                            logger.warning(f"Failed to reconnect to {stage_key}: {conn_e}")
                    
                    # Client-side cache에서 past_inputs 재전송 (replay)
                    if (stage_key in self.client_cache and 
                        session_id in self.client_cache[stage_key] and 
                        len(self.client_cache[stage_key][session_id]) > 0):
                        past_count = len(self.client_cache[stage_key][session_id])
                        logger.info(f"Replaying {past_count} past inputs to {stage_key} for session {session_id[:8]}")
                        await self._replay_past_inputs(stage_key, session_id, metadata)
                    
                    # 실패한 stage 제거 (복구 시도 완료)
                    self.failed_stages.discard(stage_key)
                    
                    # 재시도
                    if attempt < max_recovery_attempts - 1:
                        await asyncio.sleep(0.5)  # 잠시 대기 후 재시도
                        continue
                    else:
                        raise RuntimeError(f"Failed to recover {stage_key} after {max_recovery_attempts} attempts") from e
                except Exception as recovery_e:
                    logger.error(f"Recovery failed for {stage_key}: {recovery_e}")
                    if attempt < max_recovery_attempts - 1:
                        await asyncio.sleep(1.0)
                        continue
                    else:
                        raise RuntimeError(f"Failed to recover {stage_key}: {recovery_e}") from recovery_e

    async def _replay_past_inputs(
        self,
        stage_key: str,
        session_id: str,
        base_metadata: bytes
    ):
        """
        Client-side cache의 past_inputs를 새 서버에 재전송하여 KV 캐시 복구.
        Petals 논문의 fault tolerance 메커니즘.
        """
        if (stage_key not in self.client_cache or 
            session_id not in self.client_cache[stage_key] or 
            len(self.client_cache[stage_key][session_id]) == 0):
            logger.warning(f"No past inputs to replay for {stage_key} (session {session_id[:8]})")
            return
        
        past_inputs = self.client_cache[stage_key][session_id]
        logger.info(f"Replaying {len(past_inputs)} inputs to {stage_key} for session {session_id[:8]}")
        
        base_metadata_dict = MSGPackSerializer.loads(base_metadata)
        
        # cur_len 누적 계산 (prefill + decode steps)
        cumulative_len = 0
        for idx, past_input in enumerate(past_inputs):
            # 각 past_input을 순서대로 재전송
            # metadata에서 is_prefill, seq_len 등을 적절히 설정
            replay_metadata_dict = base_metadata_dict.copy()
            seq_len = past_input.shape[1]
            
            if idx == 0:
                # 첫 번째는 prefill
                replay_metadata_dict["is_prefill"] = True
                cumulative_len = seq_len
            else:
                # 이후는 decode steps
                replay_metadata_dict["is_prefill"] = False
                cumulative_len += seq_len
            
            replay_metadata_dict["seq_len"] = seq_len
            replay_metadata_dict["cur_len"] = cumulative_len
            replay_metadata_dict["session_id"] = session_id
            replay_metadata_dict["is_replay"] = True  # replay 플래그
            
            replay_metadata = MSGPackSerializer.dumps(replay_metadata_dict)
            serialized = serialize_torch_tensor(past_input)
            size = len(serialized)
            
            # 재전송 (응답은 무시, KV 캐시 복구만 목적)
            try:
                forward_fn = self._call_stage_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._call_stage_unary
                await forward_fn(
                    stage_key,
                    [serialized],
                    replay_metadata,
                    self.timeout,
                    expect_hidden=True
                )
            except Exception as e:
                logger.warning(f"Replay step {idx}/{len(past_inputs)} failed for {stage_key}: {e}, continuing...")
                # 일부 실패해도 계속 진행 (부분 복구)
        
        logger.info(f"Replay completed for {stage_key} (session {session_id[:8]})")

    def send_prefill(self, L: int, hidden: torch.Tensor, session_id: str, max_length: int):
        """Send prefill hidden states through all remote stages."""
        if self.stage != 0:
            raise RuntimeError("send_prefill should only be called by stage0")

        async def _send():
            start_all = time.perf_counter()
            # GPU에 있던 tensor를 CPU로 이동(for sending)
            hidden_cpu = hidden.cpu().detach()
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": L,
                    "cur_len": L,
                    "is_prefill": True,
                    "max_length": max_length,
                    **self.sampling,
                }
            )

            cur = hidden_cpu
            stage_times: List[Tuple[str, float]] = []
            for idx, stage_key in enumerate(self.stage_keys):
                expect_hidden = idx < len(self.stage_keys) - 1
                
                # Client-side cache에 입력 저장 (fault tolerance, session별 관리)
                if stage_key not in self.client_cache:
                    self.client_cache[stage_key] = {}
                if session_id not in self.client_cache[stage_key]:
                    self.client_cache[stage_key][session_id] = []
                self.client_cache[stage_key][session_id].append(cur.clone().cpu())  # 과거 입력 저장
                
                stage_start = time.perf_counter()
                serialized = serialize_torch_tensor(cur) # 1. Serialize Tensor
                size = cur.element_size() * cur.nelement()
                
                # 복구 가능한 RPC 호출
                result = await self._call_stage_with_recovery(
                    stage_key,
                    [serialized],
                    metadata,
                    self.timeout,
                    expect_hidden=expect_hidden,
                    is_replay=False,
                    session_id=session_id,
                )
                
                stage_times.append((stage_key, time.perf_counter() - stage_start)) # 4. Record time
                if expect_hidden: # Not last stage
                    cur = result
                else: # Last stage
                    self.last_prefill_stage_times = stage_times
                    self.last_prefill_total = time.perf_counter() - start_all
                    return result

            self.last_prefill_stage_times = stage_times
            self.last_prefill_total = time.perf_counter() - start_all
            raise RuntimeError("No final stage returned a token")

        # _last_token에 토큰 저장
        self._last_token = self._run_async(_send())

    def send_decode_step(self, cur_len: int, hidden: torch.Tensor, session_id: str, max_length: int, generated_tokens: Optional[List[int]] = None):
        """Send decode step hidden states through all remote stages."""
        if self.stage != 0:
            raise RuntimeError("send_decode_step should only be called by stage0")

        async def _send():
            start_all = time.perf_counter()
            hidden_cpu = hidden.cpu().detach()
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": 1,
                    "cur_len": cur_len,
                    "is_prefill": False,
                    "max_length": max_length,
                    "generated_tokens": (generated_tokens[-50:] if generated_tokens else []),  # 최근 50개 전달 (반복 체크 범위 확대)
                    **self.sampling,
                }
            )

            cur = hidden_cpu
            stage_times: List[Tuple[str, float]] = []
            for idx, stage_key in enumerate(self.stage_keys):
                expect_hidden = idx < len(self.stage_keys) - 1
                
                # Client-side cache에 입력 저장 (fault tolerance, session별 관리)
                if stage_key not in self.client_cache:
                    self.client_cache[stage_key] = {}
                if session_id not in self.client_cache[stage_key]:
                    self.client_cache[stage_key][session_id] = []
                self.client_cache[stage_key][session_id].append(cur.clone().cpu())  # 과거 입력 저장
                
                stage_start = time.perf_counter()
                serialized = serialize_torch_tensor(cur)
                size = cur.element_size() * cur.nelement()
                
                # 복구 가능한 RPC 호출
                result = await self._call_stage_with_recovery(
                    stage_key,
                    [serialized],
                    metadata,
                    self.timeout,
                    expect_hidden=expect_hidden,
                    is_replay=False,
                    session_id=session_id,
                )
                
                stage_times.append((stage_key, time.perf_counter() - stage_start))
                if expect_hidden:
                    cur = result
                else:
                    total = time.perf_counter() - start_all
                    self.last_decode_stage_times = stage_times
                    self.last_decode_total = total
                    self.decode_stage_history.append(stage_times)
                    self.decode_total_times.append(total)
                    return result

            self.last_decode_stage_times = stage_times
            total = time.perf_counter() - start_all
            self.last_decode_total = total
            self.decode_stage_history.append(stage_times)
            self.decode_total_times.append(total)
            raise RuntimeError("No final stage returned a token")

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
