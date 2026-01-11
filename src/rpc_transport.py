"""
RPC transport using hivemind P2P for Petals-like distributed inference.
Uses P2P protobuf handlers for client-side RPC calls.

[Full LB 핵심 변경]
- routing="stage"  : 기존 mini_petals:stage1/2/3 체인
- routing="module" : petals:module:<model>:block_* 기반으로 route를 DHT에서 계산해서 호출

[Fix 포함]
- hivemind DHT.get()의 latest 인자/return 형태 버전 차이 안전 처리 (TypeError 방지)
- Full LB에서 remote_info에 pin만 해둔 peer도 반드시 connect 시도 (peer table empty 방지)
- _run_async가 코루틴 내부 예외를 event-loop 에러로 오인해서 재실행하지 않도록 안전화
"""
import asyncio
import concurrent.futures
import random
import socket
import time
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

import torch
from hivemind import DHT, PeerID, serialize_torch_tensor, MSGPackSerializer
from hivemind.compression.serialization import deserialize_torch_tensor
from hivemind.p2p import P2P
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE

try:
    from hivemind.p2p.p2p_daemon_bindings.utils import P2PDaemonError, P2PHandlerError
except ImportError:
    from hivemind.p2p.p2p_daemon_bindings import P2PDaemonError, P2PHandlerError

from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

from .dht_utils import get_module_key

if TYPE_CHECKING:
    from multiaddr import Multiaddr

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
        # ✅ Full LB
        routing: str = "stage",  # "stage" | "module"
        model_name: str = "default",
        total_blocks: Optional[int] = None,
        start_block: int = 0,
    ):
        """
        Args:
            device: PyTorch device (cuda:0, cuda:1, etc.)
            stage: Current stage (0 for Stage0)
            dht_initial_peers: List of initial DHT peers (full multiaddrs recommended)
            dht_port: Port for DHT
            rpc_port: Port for RPC server
            timeout: Timeout for RPC calls in seconds

            routing:
                - "stage": 기존 mini_petals stage 체인
                - "module": petals:module 기반 route 계산 후 호출
            model_name/total_blocks/start_block:
                routing="module"에서 필수
        """
        self.device = device
        self.stage = stage
        self.dht_port = dht_port
        self.rpc_port = rpc_port
        self.timeout = timeout
        self.local_ip = self._get_local_ip()

        self.routing = routing
        self.model_name = model_name
        self.total_blocks = total_blocks
        self.start_block = start_block

        self.stage_keys = stage_keys or ["mini_petals:stage1", "mini_petals:stage2", "mini_petals:stage3"]
        self.sampling = {"temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k)}

        self._last_token: Optional[int] = None
        self.remote_info: Dict[str, Dict[str, Any]] = {}  # key(stage_key/module_key) -> {"peer_id": PeerID, "maddrs":[...]}
        self.last_prefill_stage_times: List[Tuple[str, float]] = []
        self.last_prefill_total: Optional[float] = None
        self.last_decode_stage_times: List[Tuple[str, float]] = []
        self.last_decode_total: Optional[float] = None
        self.decode_stage_history: List[List[Tuple[str, float]]] = []
        self.decode_total_times: List[float] = []

        # Fault tolerance: Client-side cache (Petals 논문 방식)
        self.client_cache: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self.failed_stages: Set[str] = set()
        self.failed_peers: Dict[str, Set[str]] = {}

        # Full LB: session별 route 캐시
        self.session_routes: Dict[str, List[Tuple[str, bool]]] = {}  # session_id -> [(key, expect_hidden), ...]

        initial_peers_list = self._format_initial_peers(dht_initial_peers)

        logger.info(f"Initializing DHT on {self.local_ip}:{dht_port}")
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers_list if initial_peers_list else None,
        )

        self.p2p: Optional[P2P] = None
        self.peer_id: Optional[PeerID] = None

        if self.stage == 0:
            self.p2p = self._run_async(self._create_p2p())
            self.peer_id = self.p2p.peer_id
        else:
            logger.info("Server stages do not rely on RpcTransport; only client helpers are active")

        logger.info(f"RpcTransport initialized: stage={stage}, peer_id={self.peer_id}, routing={self.routing}")

    # ----------------------------
    # Small helpers (compat/safety)
    # ----------------------------

    def _format_initial_peers(self, dht_initial_peers: List[str]) -> List[str]:
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

    def _get_local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # ✅ FIX: event loop 관련 RuntimeError만 처리하고, 코루틴 내부 예외는 재실행하지 않음
    def _run_async(self, coro):
        """
        Run an async coroutine from sync context safely.
        - If we're already inside an event loop: run in a separate thread with asyncio.run()
        - If no event loop / closed loop: asyncio.run()
        - Otherwise: loop.run_until_complete()
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if loop.is_closed():
            return asyncio.run(coro)

        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)

    async def _create_p2p(self) -> P2P:
        return await P2P.create()

    def _dht_get_compat(self, key: str, *, latest: bool = False, return_metadata: bool = False):
        """
        hivemind 버전에 따라 DHT.get 시그니처가 다를 수 있어 TypeError 방어.
        - latest/return_metadata 지원하면 사용
        - 지원 안하면 인자 없이 get(key)로 fallback
        """
        if return_metadata:
            try:
                return self.dht.get(key, return_metadata=True)  # type: ignore[arg-type]
            except TypeError:
                return self.dht.get(key)
        if latest:
            try:
                return self.dht.get(key, latest=True)  # type: ignore[arg-type]
            except TypeError:
                return self.dht.get(key)
        return self.dht.get(key)

    def _extract_dht_value(self, res: Any) -> Any:
        """
        hivemind DHT.get 반환 형태를 통일해서 value를 뽑아준다.
        가능한 케이스:
        - res.value 존재 (DHTValue)
        - res 자체가 dict
        - res가 (value, meta) 형태의 튜플/리스트
        """
        if res is None:
            return None
        if hasattr(res, "value"):
            return res.value
        if isinstance(res, dict):
            return res
        if isinstance(res, (list, tuple)) and len(res) > 0:
            first = res[0]
            if hasattr(first, "value"):
                return first.value
            return first
        return None

    def _filter_maddrs_for_connect(self, maddrs: List[str]) -> List["Multiaddr"]:
        """
        Convert list[str] multiaddrs to list[Multiaddr] suitable for hivemind connect.
        Strips /p2p/<peer_id> suffix if present.
        """
        try:
            from multiaddr import Multiaddr
        except Exception:
            return []

        filtered: List["Multiaddr"] = []
        for m in maddrs or []:
            try:
                base = m.split("/p2p/")[0]  # keep /ip4/.../tcp/...
                ma = Multiaddr(base)
                protos = [p.name for p in ma.protocols()]
                if any(p in ("ip4", "ip6") for p in protos) and any(p in ("tcp", "quic") for p in protos):
                    filtered.append(ma)
            except Exception:
                continue
        return filtered

    async def _connect_peer_if_possible(self, stage_key: str, peer_id: PeerID, maddrs: List[str]) -> None:
        """
        ✅ Full LB 핵심:
        remote_info가 이미 있어도(connect 안 한 상태) RPC가 나가면
        'failed to find any peer in table'가 뜬다.
        그래서 info가 있든 없든 maddrs가 있으면 항상 connect를 시도한다.
        """
        if self.p2p is None or not maddrs:
            return
        try:
            filtered = self._filter_maddrs_for_connect(maddrs)
            if filtered:
                await self.p2p._client.connect(peer_id, filtered)  # NOTE: internal client
                logger.info(f"Connected to {stage_key} via maddrs: {maddrs}")
        except Exception as e:
            logger.warning(f"Could not connect to {stage_key} via maddrs {maddrs}: {e}")

    # ----------------------------
    # Peer discovery / readiness
    # ----------------------------

    async def _discover_peer(
        self,
        stage_key: str,
        max_retries: int = 10,
        retry_delay: float = 1.0,
        exclude_peer_ids: Optional[Set[str]] = None,
    ) -> Tuple[PeerID, List[str]]:
        """
        Find server peer_id via DHT for a given key (stage key or module key).
        - 여러 후보(subkey들)를 가져와서 exclude_peer_ids 제외 후 선택
        """
        if exclude_peer_ids is None:
            exclude_peer_ids = set()

        loop = asyncio.get_running_loop()

        def _get_candidates_sync():
            candidates: List[Tuple[str, List[str], float]] = []
            try:
                res = self._dht_get_compat(stage_key, return_metadata=False)
            except Exception as e:
                logger.warning(f"{stage_key}: DHT.get failed: {e}")
                return []

            value = self._extract_dht_value(res)
            if value is None:
                logger.warning(f"{stage_key}: DHT.get returned None/empty")
                return []

            if not isinstance(value, dict):
                logger.warning(f"{stage_key}: Unexpected DHT value type {type(value).__name__}")
                return []

            excluded = 0
            for subk, v in value.items():
                entry = v.value if hasattr(v, "value") else v
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    entry = entry[0]
                if isinstance(entry, dict) and "value" in entry and isinstance(entry["value"], dict):
                    entry = entry["value"]

                if not isinstance(entry, dict):
                    continue

                peer_id_str = entry.get("peer_id") or str(subk)
                if not peer_id_str:
                    continue
                if peer_id_str in exclude_peer_ids:
                    excluded += 1
                    continue

                maddrs = entry.get("p2p_maddrs") or []
                ts = float(entry.get("timestamp", 0) or 0)
                candidates.append((peer_id_str, maddrs, ts))

            logger.info(
                f"{stage_key}: DHT discovery - total_entries={len(value)}, candidates={len(candidates)}, excluded={excluded}"
            )
            return candidates

        for attempt in range(max_retries):
            candidates = []
            try:
                candidates = await loop.run_in_executor(None, _get_candidates_sync)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to find peer for {stage_key} failed: {e}")

            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                top = candidates[: min(5, len(candidates))]
                peer_id_str, maddrs, ts = random.choice(top)
                logger.info(
                    f"{stage_key}: Selected peer - peer_id={peer_id_str[:8]}..., timestamp={ts}, maddrs_count={len(maddrs)}"
                )
                return PeerID.from_base58(peer_id_str), maddrs

            logger.warning(
                f"{stage_key}: no candidates found (excluded={len(exclude_peer_ids)}), retrying... "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        raise RuntimeError(f"Could not find peer via DHT key '{stage_key}' (excluded {len(exclude_peer_ids)} failed peers)")

    async def _ensure_ready(self, stage_key: str, force_refresh: bool = False):
        if self.stage != 0:
            raise RuntimeError("RpcTransport client helpers should only be used on stage0")

        if self.p2p is None:
            self.p2p = await self._create_p2p()
            self.peer_id = self.p2p.peer_id

        if force_refresh:
            self.remote_info.pop(stage_key, None)

        info = self.remote_info.get(stage_key)

        # 1) info가 없으면 DHT에서 찾는다
        if info is None:
            exclude = self.failed_peers.get(stage_key, set())
            peer_id, maddrs = await self._discover_peer(stage_key, exclude_peer_ids=exclude)
            info = {"peer_id": peer_id, "maddrs": maddrs}
            self.remote_info[stage_key] = info

        # 2) Full LB(module routing)에서 핵심: info가 있어도 connect를 반드시 시도
        try:
            peer_id = info["peer_id"]
            maddrs = info.get("maddrs") or []
            await self._connect_peer_if_possible(stage_key, peer_id, maddrs)
        except Exception as e:
            logger.warning(f"_ensure_ready connect step failed for {stage_key}: {e}")

        # optional small wait
        try:
            await asyncio.wait_for(self.p2p.wait_for_at_least_n_peers(1), timeout=2)
        except Exception:
            pass

    # ----------------------------
    # Routing (stage / module)
    # ----------------------------

    async def _compute_module_route(self, session_id: str) -> List[Tuple[str, bool]]:
        """
        petals:module 기반으로 0..total_blocks를 덮는 route를 만든다.
        greedy: 현재 block을 커버하는 후보 중 end_block이 가장 큰 서버 선택 (동률이면 throughput)
        """
        if self.total_blocks is None:
            raise ValueError("total_blocks is required when routing='module'")

        cur = int(self.start_block)
        route: List[Tuple[str, bool]] = []
        hops = 0

        while cur < self.total_blocks:
            mk = get_module_key(cur, self.model_name)

            res = self._dht_get_compat(mk, latest=True)
            value = self._extract_dht_value(res)

            if value is None or not isinstance(value, dict):
                raise RuntimeError(f"[module routing] No candidates for {mk} (block={cur})")

            # candidates: (end_block, throughput, peer_id_str, maddrs, final_stage)
            candidates: List[Tuple[int, float, str, List[str], bool]] = []
            for subk, raw in value.items():
                entry = raw.value if hasattr(raw, "value") else raw
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    entry = entry[0]
                if isinstance(entry, dict) and "value" in entry and isinstance(entry["value"], dict):
                    entry = entry["value"]

                if not isinstance(entry, dict):
                    continue

                pid = entry.get("peer_id") or str(subk)
                if not pid:
                    continue

                st = entry.get("start_block")
                ed = entry.get("end_block")
                if st is None or ed is None:
                    continue

                st_i = int(st)
                ed_i = int(ed)
                if not (st_i <= cur < ed_i):
                    continue

                thr = float(entry.get("throughput") or 0.0)
                maddrs = entry.get("p2p_maddrs") or []
                fin = bool(entry.get("final_stage", False))
                candidates.append((ed_i, thr, pid, maddrs, fin))

            if not candidates:
                raise RuntimeError(f"[module routing] No server covers block={cur} (key={mk})")

            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            end_block, _thr, pid, maddrs, _fin = candidates[0]

            # 이 hop을 "block_cur 키"로 호출하되, peer를 고정(pin)
            key = mk
            self.remote_info[key] = {"peer_id": PeerID.from_base58(pid), "maddrs": maddrs}

            is_last = end_block >= self.total_blocks
            route.append((key, not is_last))

            cur = end_block
            hops += 1
            if hops > self.total_blocks + 5:
                raise RuntimeError("[module routing] Route seems stuck; check start/end in DHT entries.")

        # 마지막 hop은 final_stage 서버인지 확인
        last_key, _ = route[-1]
        last_peer_obj = self.remote_info[last_key]["peer_id"]
        last_peer = last_peer_obj.to_base58()

        last_res = self._dht_get_compat(last_key, latest=True)
        last_value = self._extract_dht_value(last_res)

        ok = False
        if isinstance(last_value, dict):
            for subk, raw in last_value.items():
                entry = raw.value if hasattr(raw, "value") else raw
                if isinstance(entry, (list, tuple)) and len(entry) > 0:
                    entry = entry[0]
                if isinstance(entry, dict) and "value" in entry and isinstance(entry["value"], dict):
                    entry = entry["value"]

                if not isinstance(entry, dict):
                    continue
                pid = entry.get("peer_id") or str(subk)
                if pid == last_peer and bool(entry.get("final_stage", False)):
                    ok = True
                    break

        if not ok:
            raise RuntimeError(
                "[module routing] Last hop server is not marked as final_stage; need StageLast span at the end."
            )

        logger.info(f"[module routing] Built route for session {session_id[:8]}: {[k for k, _ in route]}")
        return route

    async def _get_route(self, session_id: str) -> List[Tuple[str, bool]]:
        if self.routing == "stage":
            return [(k, i < len(self.stage_keys) - 1) for i, k in enumerate(self.stage_keys)]

        if session_id not in self.session_routes:
            self.session_routes[session_id] = await self._compute_module_route(session_id)
        return self.session_routes[session_id]

    # ----------------------------
    # RPC calls
    # ----------------------------

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
        await self._ensure_ready(stage_key)
        peer_id = self.remote_info[stage_key]["peer_id"]
        request = runtime_pb2.ExpertRequest(uid=stage_key, tensors=serialized_tensors, metadata=metadata)
        response = await asyncio.wait_for(
            self.p2p.call_protobuf_handler(  # type: ignore[union-attr]
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
        await self._ensure_ready(stage_key)
        peer_id = self.remote_info[stage_key]["peer_id"]

        parts = (
            runtime_pb2.ExpertRequest(uid=stage_key, tensors=[part], metadata=metadata)
            for tensor in serialized_tensors
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        )

        outputs = self.p2p.iterate_protobuf_handler(  # type: ignore[union-attr]
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
        max_recovery_attempts = 3

        for attempt in range(max_recovery_attempts):
            try:
                if serialized_tensors:
                    first_tensor = serialized_tensors[0]
                    if hasattr(first_tensor, "ByteSize"):
                        size = first_tensor.ByteSize()
                    elif isinstance(first_tensor, bytes):
                        size = len(first_tensor)
                    else:
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
                    logger.error(f"Replay failed for {stage_key}: {e}")
                    raise

                logger.warning(f"Stage {stage_key} failed (attempt {attempt + 1}/{max_recovery_attempts}): {e}")
                self.failed_stages.add(stage_key)

                # mark failed peer
                if stage_key in self.remote_info:
                    pid_obj = self.remote_info[stage_key].get("peer_id", None)
                    failed_peer_id = ""
                    if pid_obj is not None:
                        try:
                            failed_peer_id = pid_obj.to_base58()
                        except Exception:
                            failed_peer_id = str(pid_obj)
                    if failed_peer_id:
                        self.failed_peers.setdefault(stage_key, set()).add(failed_peer_id)
                        logger.info(f"Marked peer {failed_peer_id[:8]}... as failed for {stage_key}")

                try:
                    exclude_peers = self.failed_peers.get(stage_key, set())
                    new_peer_id, new_maddrs = await self._discover_peer(
                        stage_key, max_retries=5, retry_delay=0.5, exclude_peer_ids=exclude_peers
                    )

                    self.remote_info[stage_key] = {"peer_id": new_peer_id, "maddrs": new_maddrs}
                    await self._connect_peer_if_possible(stage_key, new_peer_id, new_maddrs)

                    if (
                        stage_key in self.client_cache
                        and session_id is not None
                        and session_id in self.client_cache[stage_key]
                        and len(self.client_cache[stage_key][session_id]) > 0
                    ):
                        await self._replay_past_inputs(stage_key, session_id, metadata)

                    self.failed_stages.discard(stage_key)
                    await asyncio.sleep(0.2)

                    if attempt < max_recovery_attempts - 1:
                        continue
                    raise RuntimeError(f"Failed to recover {stage_key} after {max_recovery_attempts} attempts") from e

                except Exception as recovery_e:
                    logger.error(f"Recovery failed for {stage_key}: {recovery_e}")
                    if attempt < max_recovery_attempts - 1:
                        await asyncio.sleep(1.0)
                        continue
                    raise RuntimeError(f"Failed to recover {stage_key}: {recovery_e}") from recovery_e

    async def _replay_past_inputs(self, stage_key: str, session_id: str, base_metadata: bytes):
        if (
            stage_key not in self.client_cache
            or session_id not in self.client_cache[stage_key]
            or len(self.client_cache[stage_key][session_id]) == 0
        ):
            return

        past_inputs = self.client_cache[stage_key][session_id]
        base_metadata_dict = MSGPackSerializer.loads(base_metadata) if base_metadata else {}

        cumulative_len = 0
        for idx, past_input in enumerate(past_inputs):
            replay_metadata_dict = dict(base_metadata_dict)
            seq_len = int(past_input.shape[1])

            if idx == 0:
                replay_metadata_dict["is_prefill"] = True
                cumulative_len = seq_len
            else:
                replay_metadata_dict["is_prefill"] = False
                cumulative_len += seq_len

            replay_metadata_dict["seq_len"] = seq_len
            replay_metadata_dict["cur_len"] = cumulative_len
            replay_metadata_dict["session_id"] = session_id
            replay_metadata_dict["is_replay"] = True

            replay_metadata = MSGPackSerializer.dumps(replay_metadata_dict)
            serialized = serialize_torch_tensor(past_input)

            if hasattr(serialized, "ByteSize"):
                size = serialized.ByteSize()
            elif isinstance(serialized, bytes):
                size = len(serialized)
            else:
                try:
                    size = len(serialized.SerializeToString())
                except AttributeError:
                    size = 0

            forward_fn = self._call_stage_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else self._call_stage_unary
            await forward_fn(stage_key, [serialized], replay_metadata, self.timeout, expect_hidden=True)

    # ----------------------------
    # Public API: prefill / decode
    # ----------------------------

    def send_prefill(self, L: int, hidden: torch.Tensor, session_id: str, max_length: int):
        if self.stage != 0:
            raise RuntimeError("send_prefill should only be called by stage0")

        async def _send():
            start_all = time.perf_counter()
            hidden_cpu = hidden.cpu().detach()
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": int(L),
                    "cur_len": int(L),
                    "is_prefill": True,
                    "max_length": int(max_length),
                    **self.sampling,
                }
            )

            cur = hidden_cpu
            stage_times: List[Tuple[str, float]] = []
            route = await self._get_route(session_id)

            for stage_key, expect_hidden in route:
                self.client_cache.setdefault(stage_key, {}).setdefault(session_id, []).append(cur.clone().cpu())

                stage_start = time.perf_counter()
                serialized = serialize_torch_tensor(cur)

                result = await self._call_stage_with_recovery(
                    stage_key,
                    [serialized],
                    metadata,
                    self.timeout,
                    expect_hidden=expect_hidden,
                    is_replay=False,
                    session_id=session_id,
                )

                info = self.remote_info.get(stage_key, {})
                maddrs = info.get("maddrs") or []
                logger.info(f"Prefill pass: key={stage_key}, maddrs={maddrs}")

                stage_times.append((stage_key, time.perf_counter() - stage_start))
                if expect_hidden:
                    cur = result
                else:
                    self.last_prefill_stage_times = stage_times
                    self.last_prefill_total = time.perf_counter() - start_all
                    return result

            self.last_prefill_stage_times = stage_times
            self.last_prefill_total = time.perf_counter() - start_all
            raise RuntimeError("No final stage returned a token")

        self._last_token = self._run_async(_send())

    def send_decode_step(
        self,
        cur_len: int,
        hidden: torch.Tensor,
        session_id: str,
        max_length: int,
        generated_tokens: Optional[List[int]] = None,
    ):
        if self.stage != 0:
            raise RuntimeError("send_decode_step should only be called by stage0")

        async def _send():
            start_all = time.perf_counter()
            hidden_cpu = hidden.cpu().detach()
            metadata = MSGPackSerializer.dumps(
                {
                    "session_id": session_id,
                    "seq_len": 1,
                    "cur_len": int(cur_len),
                    "is_prefill": False,
                    "max_length": int(max_length),
                    "generated_tokens": (generated_tokens[-50:] if generated_tokens else []),
                    **self.sampling,
                }
            )

            cur = hidden_cpu
            stage_times: List[Tuple[str, float]] = []
            route = await self._get_route(session_id)

            for stage_key, expect_hidden in route:
                self.client_cache.setdefault(stage_key, {}).setdefault(session_id, []).append(cur.clone().cpu())

                stage_start = time.perf_counter()
                serialized = serialize_torch_tensor(cur)

                result = await self._call_stage_with_recovery(
                    stage_key,
                    [serialized],
                    metadata,
                    self.timeout,
                    expect_hidden=expect_hidden,
                    is_replay=False,
                    session_id=session_id,
                )

                info = self.remote_info.get(stage_key, {})
                maddrs = info.get("maddrs") or []
                logger.info(f"Decode pass: key={stage_key}, maddrs={maddrs}")

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

            total = time.perf_counter() - start_all
            self.last_decode_stage_times = stage_times
            self.last_decode_total = total
            self.decode_stage_history.append(stage_times)
            self.decode_total_times.append(total)
            raise RuntimeError("No final stage returned a token")

        self._last_token = self._run_async(_send())

    def recv_token(self) -> int:
        if self.stage != 0:
            raise RuntimeError("recv_token should only be called by stage0")
        if self._last_token is None:
            raise RuntimeError("No token received. Call send_prefill or send_decode_step first.")
        token_id = self._last_token
        self._last_token = None
        return int(token_id)

    def shutdown(self):
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
