"""
DHT를 통한 서버 정보 저장/조회 유틸리티
Load Balancing을 위해 서버 정보를 DHT에 저장하고 조회

[Full LB 핵심 변경]
- petals:module:<model>:block_i 키는 subkey=peer_id 로 여러 서버 공존 가능
- module entry에 routing에 필요한 필드(start/end/throughput/p2p_maddrs/final_stage) 포함
- TTL 짧게 두고 서버가 heartbeat로 계속 갱신하도록 설계
"""

from typing import List, Optional, Dict, Any
from hivemind import DHT, PeerID, get_dht_time
from hivemind.utils.logging import get_logger

from .load_balancing import RemoteModuleInfo, ServerInfo, ServerState

logger = get_logger(__name__)

# DHT 키 접두사
MODULE_KEY_PREFIX = "petals:module:"
SERVER_KEY_PREFIX = "petals:server:"


def get_module_key(block_idx: int, model_name: str = "default") -> str:
    """모듈(블록) DHT 키 생성"""
    return f"{MODULE_KEY_PREFIX}{model_name}:block_{block_idx}"


def get_server_key(peer_id: PeerID, model_name: str = "default") -> str:
    """서버 DHT 키 생성"""
    return f"{SERVER_KEY_PREFIX}{model_name}:{peer_id}"


def register_server_on_dht(
    dht: DHT,
    peer_id: PeerID,
    start_block: int,
    end_block: int,
    throughput: float,
    model_name: str = "default",
    server_address: Optional[str] = None,
    p2p_maddrs: Optional[List[str]] = None,
    final_stage: bool = False,
    state: ServerState = ServerState.ONLINE,
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 서버 정보 등록 (petals:server:*)

    NOTE:
    - server_key는 peer_id를 포함하므로 subkey 없이 단일 값으로 저장해도 충분
    - 하지만 module_key는 반드시 subkey로 여러 서버 공존이 필요함
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 90  # 짧게. heartbeat에서 TTL/3마다 갱신 권장

    server_info = {
        "peer_id": str(peer_id),
        "timestamp": get_dht_time(),
        "start_block": int(start_block),
        "end_block": int(end_block),
        "throughput": float(throughput),
        "state": state.value,
        "server_address": server_address,
        "p2p_maddrs": p2p_maddrs or [],
        "final_stage": bool(final_stage),
    }

    server_key = get_server_key(peer_id, model_name)
    try:
        dht.store(key=server_key, value=server_info, expiration_time=expiration_time)
        logger.info(
            f"Registered server {peer_id} on DHT: blocks [{start_block}:{end_block}], "
            f"throughput={throughput:.2f} rps, final_stage={final_stage}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to register server on DHT: {e}")
        return False


def register_blocks_on_dht(
    dht: DHT,
    peer_id: PeerID,
    block_indices: List[int],
    model_name: str = "default",
    p2p_maddrs: Optional[List[str]] = None,
    start_block: Optional[int] = None,
    end_block: Optional[int] = None,
    throughput: Optional[float] = None,
    final_stage: bool = False,
    state: ServerState = ServerState.ONLINE,
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 블록-서버 매핑 등록 (petals:module:*)

    Full LB 핵심:
    - 동일 block 키에 여러 서버가 공존해야 하므로 subkey=str(peer_id) 사용
    - entry에 routing에 필요한 정보 포함
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 90  # 짧게. heartbeat로 갱신

    try:
        for block_idx in block_indices:
            module_key = get_module_key(block_idx, model_name)

            module_info = {
                "peer_id": str(peer_id),  # P2P peer_id
                "timestamp": get_dht_time(),
                "block_idx": int(block_idx),
                "start_block": int(start_block) if start_block is not None else None,
                "end_block": int(end_block) if end_block is not None else None,
                "throughput": float(throughput) if throughput is not None else None,
                "state": state.value,
                "p2p_maddrs": p2p_maddrs or [],
                "final_stage": bool(final_stage),
            }

            # ✅ 핵심: subkey로 peer_id 사용 => 여러 서버 공존
            dht.store(
                key=module_key,
                subkey=str(peer_id),
                value=module_info,
                expiration_time=expiration_time,
            )

        logger.info(f"Registered {len(block_indices)} blocks for server {peer_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to register blocks on DHT: {e}")
        return False


def _unwrap_value(v: Any) -> Any:
    """hivemind 버전별 반환 형태(ValueWithExpiration, tuple 등) 정리"""
    if hasattr(v, "value"):
        return v.value
    if isinstance(v, tuple) and len(v) > 0:
        return v[0]
    if isinstance(v, dict) and "value" in v:
        return v.get("value")
    return v


def get_remote_module_infos(
    dht: DHT,
    model_name: str = "default",
    total_blocks: Optional[int] = None,
) -> List[RemoteModuleInfo]:
    """
    DHT에서 모든 원격 모듈 정보 조회

    Full LB:
    - module_key는 dict(subkey->entry) 형태로 저장되어 있으므로 모든 subkey를 순회
    """
    module_infos: List[RemoteModuleInfo] = []

    if total_blocks is None:
        total_blocks = 64

    # peer_id(str) -> ServerInfo cache (module entry만으로도 계산 가능)
    server_infos_cache: Dict[str, ServerInfo] = {}

    for block_idx in range(total_blocks):
        module_key = get_module_key(block_idx, model_name)

        try:
            result = dht.get(module_key, latest=True)
            if result is None or result.value is None:
                continue

            value = result.value
            if not isinstance(value, dict):
                # 구버전/단일 entry 호환: dict가 아닌 경우는 무시
                continue

            for subk, raw in value.items():
                entry = _unwrap_value(raw)
                if not isinstance(entry, dict):
                    continue

                peer_id_str = entry.get("peer_id") or str(subk)
                if not peer_id_str:
                    continue

                # routing 필수 필드
                start_block = entry.get("start_block")
                end_block = entry.get("end_block")
                if start_block is None or end_block is None:
                    # start/end가 없으면 spans 계산이 불가능하므로 스킵
                    continue

                if peer_id_str not in server_infos_cache:
                    try:
                        pid = PeerID.from_base58(peer_id_str)
                    except Exception:
                        continue

                    state_str = entry.get("state", "online")
                    try:
                        state = ServerState(state_str)
                    except Exception:
                        state = ServerState.ONLINE

                    server_infos_cache[peer_id_str] = ServerInfo(
                        peer_id=pid,
                        state=state,
                        throughput=float(entry.get("throughput") or 0.0),
                        start_block=int(start_block),
                        end_block=int(end_block),
                        server_address=None,
                    )

                module_infos.append(
                    RemoteModuleInfo(uid=f"block_{block_idx}", server_info=server_infos_cache[peer_id_str])
                )

        except Exception as e:
            logger.debug(f"Failed to get module info for block {block_idx}: {e}")
            continue

    logger.info(f"Retrieved {len(module_infos)} module infos from DHT (total_blocks={total_blocks})")

    # 요약 로그
    if module_infos:
        block_servers: Dict[int, List[str]] = {}
        for info in module_infos:
            try:
                b = int(info.uid.split("_")[-1])
                sid = str(info.server_info.peer_id)[:16] if info.server_info else "none"
                block_servers.setdefault(b, []).append(sid)
            except Exception:
                pass
        if block_servers:
            logger.info(
                f"Block coverage: {min(block_servers.keys())} to {max(block_servers.keys())} "
                f"({len(block_servers)} blocks found)"
            )

    return module_infos


def update_server_throughput_on_dht(
    dht: DHT,
    peer_id: PeerID,
    new_throughput: float,
    model_name: str = "default",
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 저장된 서버 처리량 업데이트 (petals:server:*)

    NOTE:
    - Full LB에서는 module entry에도 throughput이 들어가므로
      가장 안전한 방식은 "heartbeat에서 register_blocks_on_dht를 최신 throughput으로 계속 덮어쓰기"임.
    - 이 함수는 server_key 값만 업데이트한다.
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 90

    server_key = get_server_key(peer_id, model_name)

    try:
        result = dht.get(server_key, latest=True)
        if result is None or result.value is None:
            logger.warning(f"Server {peer_id} not found on DHT, cannot update throughput")
            return False

        server_data = result.value.copy()
        server_data["throughput"] = float(new_throughput)
        server_data["timestamp"] = get_dht_time()

        dht.store(server_key, server_data, expiration_time=expiration_time)
        logger.debug(f"Updated throughput for server {peer_id}: {new_throughput:.2f} rps")
        return True

    except Exception as e:
        logger.error(f"Failed to update server throughput on DHT: {e}")
        return False
