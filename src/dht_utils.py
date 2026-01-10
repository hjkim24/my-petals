"""
DHT를 통한 서버 정보 저장/조회 유틸리티
Load Balancing을 위해 서버 정보를 DHT에 저장하고 조회
"""

from typing import List, Optional, Dict
from hivemind import DHT, PeerID, get_dht_time
from hivemind.utils.logging import get_logger
from .load_balancing import (
    RemoteModuleInfo, ServerInfo, ServerState,
    compute_spans
)

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
    state: ServerState = ServerState.ONLINE,
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 서버 정보 등록
    
    Args:
        dht: DHT 인스턴스
        peer_id: 서버 PeerID
        start_block: 담당 블록 시작 인덱스
        end_block: 담당 블록 끝 인덱스
        throughput: 처리량
        model_name: 모델 이름
        server_address: 서버 주소 (선택적)
        state: 서버 상태
        expiration_time: 만료 시간 (초), None이면 기본값 사용
    
    Returns:
        성공 여부
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 300  # 기본 5분
    
    server_info = {
        "peer_id": str(peer_id),
        "start_block": start_block,
        "end_block": end_block,
        "throughput": throughput,
        "state": state.value,
        "server_address": server_address,
        "updated_at": get_dht_time(),
    }
    
    # 서버 정보 저장
    server_key = get_server_key(peer_id, model_name)
    try:
        dht.store(server_key, server_info, expiration_time=expiration_time)
        logger.info(f"Registered server {peer_id} on DHT: blocks [{start_block}:{end_block}], "
                   f"throughput={throughput:.2f} rps")
        return True
    except Exception as e:
        logger.error(f"Failed to register server on DHT: {e}")
        return False


def register_blocks_on_dht(
    dht: DHT,
    peer_id: PeerID,
    block_indices: List[int],
    model_name: str = "default",
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 블록-서버 매핑 등록
    
    Args:
        dht: DHT 인스턴스
        peer_id: 서버 PeerID
        block_indices: 담당하는 블록 인덱스 리스트
        model_name: 모델 이름
        expiration_time: 만료 시간
    
    Returns:
        성공 여부
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 300  # 기본 5분
    
    try:
        for block_idx in block_indices:
            module_key = get_module_key(block_idx, model_name)
            module_info = {
                "block_idx": block_idx,
                "server_peer_id": str(peer_id),
                "updated_at": get_dht_time(),
            }
            dht.store(module_key, module_info, expiration_time=expiration_time)
        
        logger.info(f"Registered {len(block_indices)} blocks for server {peer_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to register blocks on DHT: {e}")
        return False


def get_remote_module_infos(
    dht: DHT,
    model_name: str = "default",
    total_blocks: Optional[int] = None,
) -> List[RemoteModuleInfo]:
    """
    DHT에서 모든 원격 모듈 정보 조회
    
    Args:
        dht: DHT 인스턴스
        model_name: 모델 이름
        total_blocks: 전체 블록 개수 (None이면 자동 탐지)
    
    Returns:
        원격 모듈 정보 리스트
    """
    module_infos: List[RemoteModuleInfo] = []
    
    # 전체 블록 개수 추정
    if total_blocks is None:
        # DHT에서 최대 블록 인덱스 탐색 (간단한 휴리스틱)
        total_blocks = 64  # 기본값, 실제로는 동적으로 탐지해야 함
    
    # 서버 정보를 먼저 조회
    server_infos_cache: Dict[str, ServerInfo] = {}
    
    # 각 블록에 대해 조회
    for block_idx in range(total_blocks):
        module_key = get_module_key(block_idx, model_name)
        
        try:
            result = dht.get(module_key, latest=True)
            if result is None or result.value is None:
                continue
            
            module_data = result.value
            server_peer_id_str = module_data.get("server_peer_id")
            
            if server_peer_id_str is None:
                continue
            
            # 서버 정보 조회 (캐시 활용)
            if server_peer_id_str not in server_infos_cache:
                server_key = get_server_key(PeerID.from_base58(server_peer_id_str), model_name)
                server_result = dht.get(server_key, latest=True)
                
                if server_result is None or server_result.value is None:
                    continue
                
                server_data = server_result.value
                server_info = ServerInfo(
                    peer_id=PeerID.from_base58(server_peer_id_str),
                    state=ServerState(server_data.get("state", "online")),
                    throughput=float(server_data.get("throughput", 0.0)),
                    start_block=int(server_data.get("start_block", 0)),
                    end_block=int(server_data.get("end_block", 0)),
                    server_address=server_data.get("server_address"),
                )
                server_infos_cache[server_peer_id_str] = server_info
            
            # 모듈 정보 생성
            module_info = RemoteModuleInfo(
                uid=f"block_{block_idx}",
                server_info=server_infos_cache[server_peer_id_str],
            )
            module_infos.append(module_info)
            
        except Exception as e:
            logger.debug(f"Failed to get module info for block {block_idx}: {e}")
            continue
    
    logger.debug(f"Retrieved {len(module_infos)} module infos from DHT")
    return module_infos


def update_server_throughput_on_dht(
    dht: DHT,
    peer_id: PeerID,
    new_throughput: float,
    model_name: str = "default",
    expiration_time: Optional[float] = None,
) -> bool:
    """
    DHT에 저장된 서버 처리량 업데이트
    
    Args:
        dht: DHT 인스턴스
        peer_id: 서버 PeerID
        new_throughput: 새로운 처리량
        model_name: 모델 이름
        expiration_time: 만료 시간
    
    Returns:
        성공 여부
    """
    if expiration_time is None:
        expiration_time = get_dht_time() + 300
    
    server_key = get_server_key(peer_id, model_name)
    
    try:
        # 기존 정보 조회
        result = dht.get(server_key, latest=True)
        if result is None or result.value is None:
            logger.warning(f"Server {peer_id} not found on DHT, cannot update throughput")
            return False
        
        server_data = result.value.copy()
        server_data["throughput"] = new_throughput
        server_data["updated_at"] = get_dht_time()
        
        dht.store(server_key, server_data, expiration_time=expiration_time)
        logger.debug(f"Updated throughput for server {peer_id}: {new_throughput:.2f} rps")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update server throughput on DHT: {e}")
        return False

