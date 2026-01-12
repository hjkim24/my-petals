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
        - ✅ 여러 후보(subkey들)를 가져와서
        - ✅ exclude_peer_ids(실패 서버) 제외하고
        - ✅ 그 중 하나를 선택해서 반환
        """
        if exclude_peer_ids is None:
            exclude_peer_ids = set()

        loop = asyncio.get_running_loop()

        def _get_candidates_sync():
            """Get candidate peers from DHT, handling different hivemind versions."""
            candidates = []
            
            # hivemind 버전에 따라 return_metadata 지원 여부가 다름
            # 먼저 return_metadata 없이 시도
            try:
                res = self.dht.get(stage_key)
            except TypeError:
                # return_metadata가 필수인 경우를 대비해 다시 시도
                try:
                    res = self.dht.get(stage_key, return_metadata=True)
                except TypeError:
                    # return_metadata를 지원하지 않는 버전
                    res = self.dht.get(stage_key)
            
            if res is None:
                logger.warning(f"{stage_key}: DHT.get returned None (no value for this key)")
                return []
            
            # res가 다양한 형태일 수 있음
            value = None
            if hasattr(res, 'value'):
                value = res.value
            elif isinstance(res, dict):
                value = res
            elif isinstance(res, (list, tuple)) and len(res) > 0:
                # 일부 버전은 리스트/튜플로 반환
                value = res[0] if isinstance(res[0], dict) else res
            
            if value is None:
                logger.warning(f"{stage_key}: DHT.get returned object without usable value (type={type(res).__name__})")
                return []

            # hivemind 버전에 따라 value가 dict(subkey->entry)일 수 있음
            if isinstance(value, dict):
                # subkey가 있는 경우 (여러 서버)
                debug_entries = []
                excluded_peers = []
                for subk, v in value.items():
                    entry = v
                    # ValueWithExpiration 객체 처리 (hivemind의 일반적인 반환 형태)
                    if hasattr(v, 'value'):
                        entry = v.value
                    # 어떤 버전은 (entry, expiration, ...) 튜플로 줄 때가 있음
                    elif isinstance(v, tuple) and len(v) > 0:
                        entry = v[0]
                    # dict에 "value" 키가 있는 경우
                    elif isinstance(v, dict) and "value" in v:
                        entry = v.get("value", v)

                    if not isinstance(entry, dict):
                        logger.debug(f"{stage_key}: Skipping non-dict entry for subkey {subk}, type={type(entry).__name__}, value={entry}")
                        continue

                    peer_id_str = entry.get("peer_id") or str(subk)
                    if not peer_id_str:
                        continue

                    # 실패한 peer 제외
                    if peer_id_str in exclude_peer_ids:
                        excluded_peers.append(peer_id_str)
                        logger.debug(f"{stage_key}: Excluding failed peer {peer_id_str[:8]}... (subkey={subk})")
                        continue

                    # maddrs
                    maddrs = entry.get("p2p_maddrs") or []
                    ts = entry.get("timestamp", 0)
                    throughput = entry.get("throughput", 0.0)  # 기본값 0 (성능 정보 없음)

                    candidates.append((peer_id_str, maddrs, ts, throughput))
                    debug_entries.append(
                        {
                            "subkey": str(subk),
                            "peer_id": peer_id_str,
                            "maddrs": maddrs,
                            "timestamp": ts,
                            "throughput": throughput,
                        }
                    )

                # 상세 로그 출력
                logger.info(
                    f"{stage_key}: DHT discovery - "
                    f"total_entries={len(value)}, "
                    f"candidates={len(candidates)}, "
                    f"excluded={len(excluded_peers)} (excluded_peer_ids={list(exclude_peer_ids)}), "
                    f"candidate_peers={[c[0][:8]+'...' for c in candidates]}"
                )
                if debug_entries:
                    logger.debug(f"{stage_key}: DHT entries={debug_entries}")
                if excluded_peers:
                    logger.debug(f"{stage_key}: Excluded peers={[p[:8]+'...' for p in excluded_peers]}")
                if not candidates and len(value) > 0:
                    logger.warning(
                        f"{stage_key}: DHT dict value has {len(value)} entries but no usable candidates "
                        f"(excluded={len(excluded_peers)}, exclude_peer_ids={list(exclude_peer_ids)})"
                    )
            else:
                # 단일 entry 형태 (subkey 없음)
                if isinstance(value, dict):
                    peer_id_str = value.get("peer_id")
                    if peer_id_str and peer_id_str not in exclude_peer_ids:
                        maddrs = value.get("p2p_maddrs") or []
                        ts = value.get("timestamp", 0)
                        throughput = value.get("throughput", 0.0)  # 기본값 0 (성능 정보 없음)
                        candidates.append((peer_id_str, maddrs, ts, throughput))
                        logger.info(
                            f"{stage_key}: DHT single entry="
                            f"{{'peer_id': {peer_id_str}, 'maddrs': {maddrs}, 'timestamp': {ts}}}"
                        )
                else:
                    logger.warning(
                        f"{stage_key}: Unexpected DHT value type {type(value).__name__}, raw value={value}"
                    )

            return candidates

        for attempt in range(max_retries):
            try:
                candidates = await loop.run_in_executor(None, _get_candidates_sync)

                if candidates:
                    # 선택 정책: throughput 기반 선택
                    # 1) throughput 높은 순으로 정렬
                    # 2) 최고 성능 서버 선택 (동일 throughput일 경우 랜덤 선택)
                    candidates.sort(key=lambda x: x[3], reverse=True)  # x[3] = throughput
                    
                    if candidates:
                        top_throughput = candidates[0][3]
                        # 동일 throughput을 가진 서버들 중에서 선택
                        top_candidates = [c for c in candidates if c[3] == top_throughput]
                        peer_id_str, maddrs, _ts, _throughput = random.choice(top_candidates)

                        logger.info(
                            f"{stage_key}: Selected peer from {len(candidates)} candidates - "
                            f"peer_id={peer_id_str[:8]}..., "
                            f"timestamp={_ts}, "
                            f"throughput={_throughput:.1f} tokens/s, "
                            f"maddrs_count={len(maddrs)}, "
                            f"top_candidates={len(top_candidates)}"
                        )
                    else:
                        # Fallback (should not happen)
                        peer_id_str, maddrs, _ts, _throughput = candidates[0]
                        logger.warning(
                            f"{stage_key}: Fallback selection - "
                            f"peer_id={peer_id_str[:8]}..., "
                            f"throughput={_throughput:.1f} tokens/s"
                        )
                    peer_id = PeerID.from_base58(peer_id_str)
                    return peer_id, maddrs

                logger.warning(
                    f"{stage_key}: no candidates found (excluded={len(exclude_peer_ids)}, "
                    f"excluded_peer_ids={[p[:8]+'...' for p in exclude_peer_ids] if exclude_peer_ids else 'none'}), "
                    f"retrying... (attempt {attempt+1}/{max_retries})"
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to find peer for {stage_key} failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        raise RuntimeError(
            f"Could not find peer via DHT key '{stage_key}' (excluded {len(exclude_peer_ids)} failed peers)"
        )

    async def _ensure_ready(self, stage_key: str, force_refresh: bool = False):
        if self.stage != 0:
            raise RuntimeError("RpcTransport client helpers should only be used on stage0")
        if self.p2p is None:
            self.p2p = await self._create_p2p()
            self.peer_id = self.p2p.peer_id

        if force_refresh:
            self.remote_info.pop(stage_key, None)

        info = self.remote_info.get(stage_key)
        if info is None:
            exclude = self.failed_peers.get(stage_key, set())
            peer_id, maddrs = await self._discover_peer(stage_key, exclude_peer_ids=exclude)
            info = {"peer_id": peer_id, "maddrs": maddrs}
            self.remote_info[stage_key] = info

            # connect 시도
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
                        except Exception:
                            pass
                    if filtered:
                        await self.p2p._client.connect(peer_id, filtered)
                        logger.info(f"Connected to {stage_key} via maddrs: {maddrs}")
                    else:
                        logger.warning(f"No usable tcp/ip multiaddrs to connect: {maddrs}")
                except Exception as e:
                    logger.warning(f"Could not connect to {stage_key} via maddrs {maddrs}: {e}")

        # optional: peers 확인
        try:
            await asyncio.wait_for(self.p2p.wait_for_at_least_n_peers(1), timeout=5)
        except Exception:
            pass


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
                    pid_obj = self.remote_info[stage_key].get("peer_id", None)
                    # PeerID -> base58 string으로 통일
                    if pid_obj is not None:
                        try:
                            failed_peer_id = pid_obj.to_base58()
                        except Exception:
                            failed_peer_id = str(pid_obj)
                    else:
                        failed_peer_id = ""

                    if failed_peer_id:
                        if stage_key not in self.failed_peers:
                            self.failed_peers[stage_key] = set()
                        self.failed_peers[stage_key].add(failed_peer_id)
                        logger.info(f"Marked peer {failed_peer_id[:8]}... as failed for {stage_key}")

                
                # 새 서버 찾기 (실패한 서버 제외)
                try:
                    exclude_peers = self.failed_peers.get(stage_key, set())
                    logger.info(
                        f"{stage_key}: Attempting recovery - "
                        f"excluded_peers={[p[:8]+'...' for p in exclude_peers] if exclude_peers else 'none'}, "
                        f"failed_peers_count={len(exclude_peers)}"
                    )
                    new_peer_id, new_maddrs = await self._discover_peer(
                        stage_key, 
                        max_retries=5, 
                        retry_delay=0.5,
                        exclude_peer_ids=exclude_peers
                    )
                    logger.info(
                        f"{stage_key}: Found replacement server - "
                        f"peer_id={new_peer_id.to_base58()[:8]}..., "
                        f"maddrs={new_maddrs}"
                    )
                    
                    # remote_info 업데이트
                    self.remote_info[stage_key] = {
                        "peer_id": new_peer_id,
                        "maddrs": new_maddrs
                    }

                    # ✅ force_refresh로 새 peer로 강제 재연결 준비
                    await self._ensure_ready(stage_key, force_refresh=True)

                    
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
                        try:
                            await self._replay_past_inputs(stage_key, session_id, metadata)
                            logger.info(f"Replay completed successfully for {stage_key} (session {session_id[:8]})")
                        except Exception as replay_e:
                            logger.error(f"Replay failed for {stage_key} (session {session_id[:8]}): {replay_e}")
                            # Replay 실패는 recovery 실패로 처리
                            raise RuntimeError(f"Replay failed for {stage_key}: {replay_e}") from replay_e
                    
                    # 실패한 stage 제거 (복구 시도 완료)
                    self.failed_stages.discard(stage_key)
                    
                    # 재시도 전에 서버가 준비될 시간을 줌
                    await asyncio.sleep(0.2)
                    
                    # 재시도
                    if attempt < max_recovery_attempts - 1:
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
            
            # protobuf Tensor 크기 계산 (hivemind 버전 호환)
            if hasattr(serialized, 'ByteSize'):
                size = serialized.ByteSize()
            elif isinstance(serialized, bytes):
                size = len(serialized)
            else:
                # fallback: SerializeToString()으로 변환 후 크기 확인
                try:
                    size = len(serialized.SerializeToString())
                except AttributeError:
                    size = 0
            
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
                logger.debug(f"Replay step {idx+1}/{len(past_inputs)} completed for {stage_key} (session {session_id[:8]})")
            except Exception as e:
                logger.error(f"Replay step {idx+1}/{len(past_inputs)} failed for {stage_key} (session {session_id[:8]}): {e}")
                # Replay 실패는 치명적이므로 예외를 다시 발생시켜서 recovery 실패로 처리
                raise RuntimeError(f"Replay failed at step {idx+1}/{len(past_inputs)} for {stage_key}: {e}") from e
        
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
                # 통과한 스테이지 및 해당 스테이지의 maddrs 로그
                info = self.remote_info.get(stage_key, {})
                maddrs = info.get("maddrs") or []
                logger.info(f"Prefill pass: stage_key={stage_key}, maddrs={maddrs}")

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
                # 통과한 스테이지 및 해당 스테이지의 maddrs 로그
                info = self.remote_info.get(stage_key, {})
                maddrs = info.get("maddrs") or []
                logger.info(f"Decode pass: stage_key={stage_key}, maddrs={maddrs}")

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