"""
논문식 Full Load Balancing 구현
Paper: "Distributed Inference and Fine-tuning of Large Language Models Over The Internet"
Section E & Appendix D
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from hivemind import DHT, PeerID, get_logger

logger = get_logger(__name__)


class ServerState(Enum):
    """서버 상태 (논문 Appendix D)"""
    JOINING = "joining"  # 조인 중 (블록 선택 중)
    ONLINE = "online"    # 온라인 및 서빙 중
    OFFLINE = "offline"  # 오프라인


@dataclass
class RemoteModuleInfo:
    """원격 모듈(블록) 정보"""
    uid: str  # 모듈 고유 ID (예: "block_0", "block_1")
    server_info: Optional["ServerInfo"] = None  # 해당 블록을 서빙하는 서버 정보


@dataclass
class ServerInfo:
    """서버 정보"""
    peer_id: PeerID
    state: ServerState
    throughput: float  # 초당 처리량 (requests per second)
    start_block: int  # 담당하는 블록 시작 인덱스
    end_block: int    # 담당하는 블록 끝 인덱스 (exclusive)
    server_address: Optional[str] = None  # 서버 주소 (선택적)

    @property
    def num_blocks(self) -> int:
        """담당하는 블록 개수"""
        return self.end_block - self.start_block


@dataclass
class RemoteSpanInfo:
    """원격 스팬 정보 (논문 Appendix D)"""
    peer_id: PeerID
    start: int  # 블록 시작 인덱스
    end: int    # 블록 끝 인덱스 (exclusive)
    length: int  # 블록 개수
    throughput: float  # 처리량

    def __post_init__(self):
        if self.length != self.end - self.start:
            self.length = self.end - self.start


def compute_spans(
    module_infos: List[RemoteModuleInfo],
    min_state: ServerState = ServerState.JOINING
) -> Dict[PeerID, RemoteSpanInfo]:
    """
    모듈 정보들로부터 스팬 정보를 계산 (논문 Appendix D)

    Args:
        module_infos: 원격 모듈 정보 리스트
        min_state: 최소 서버 상태 (이 이상인 서버만 포함)

    Returns:
        PeerID를 키로 하는 스팬 정보 딕셔너리
    """
    spans: Dict[PeerID, RemoteSpanInfo] = {}

    # 서버별로 그룹화
    server_blocks: Dict[PeerID, List[Tuple[int, float]]] = {}  # {peer_id: [(block_idx, throughput), ...]}

    # 상태 우선순위 정의
    state_priority = {ServerState.JOINING: 0, ServerState.ONLINE: 1, ServerState.OFFLINE: 2}
    min_priority = state_priority[min_state]

    for module_info in module_infos:
        if module_info.server_info is None:
            continue

        server = module_info.server_info

        # 상태 필터링
        server_priority = state_priority.get(server.state, 999)
        if server_priority < min_priority:
            continue

        if server.peer_id not in server_blocks:
            server_blocks[server.peer_id] = []

        # 블록 인덱스 추출 (uid에서, 예: "block_0" -> 0)
        try:
            block_idx = int(module_info.uid.split("_")[-1])
            server_blocks[server.peer_id].append((block_idx, server.throughput))
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse block index from uid: {module_info.uid}")
            continue

    # 연속된 블록들로 스팬 생성
    for peer_id, blocks in server_blocks.items():
        if not blocks:
            continue

        blocks.sort(key=lambda x: x[0])  # 블록 인덱스로 정렬

        # 연속된 블록들을 그룹화
        current_start = blocks[0][0]
        current_end = blocks[0][0] + 1
        current_throughput = blocks[0][1]

        for i in range(1, len(blocks)):
            block_idx, throughput = blocks[i]

            if block_idx == current_end:  # 연속된 블록
                current_end = block_idx + 1
                # 처리량은 병목(min)을 사용
                current_throughput = min(current_throughput, throughput)
            else:  # 불연속 - 새 스팬 시작
                span = RemoteSpanInfo(
                    peer_id=peer_id,
                    start=current_start,
                    end=current_end,
                    length=current_end - current_start,
                    throughput=current_throughput
                )
                spans[peer_id] = span

                current_start = block_idx
                current_end = block_idx + 1
                current_throughput = throughput

        span = RemoteSpanInfo(
            peer_id=peer_id,
            start=current_start,
            end=current_end,
            length=current_end - current_start,
            throughput=current_throughput
        )
        spans[peer_id] = span

    return spans


def compute_throughputs(
    spans: Dict[PeerID, RemoteSpanInfo],
    total_blocks: int
) -> np.ndarray:
    """
    각 블록별 누적 처리량 계산 (논문 Appendix D)

    여러 서버가 같은 블록을 담당할 경우 처리량 합산

    Args:
        spans: 스팬 정보 딕셔너리
        total_blocks: 전체 블록 개수

    Returns:
        각 블록별 처리량 배열 (shape: [total_blocks])
    """
    throughputs = np.zeros(total_blocks)

    for span in sorted(spans.values(), key=lambda span: str(span.peer_id)):
        throughputs[span.start : span.end] += span.throughput

    return throughputs


def _choose_best_start(
    throughputs: np.ndarray,
    num_blocks: int,
    min_block: int = 0,
) -> int:
    """
    시작 블록 인덱스 선택.

    기존 min-max(=이미 throughput이 큰 구간 선호) 대신,
    "가장 부족한(throughput이 낮은)" 구간을 우선적으로 커버하도록 선택한다.

    - 우선순위: (구간 내 최소 throughput, 구간 평균 throughput, 시작 인덱스) 를 오름차순
      => 병목이 가장 심한 구간을 먼저 메꾼다.
    - min_block: 이 인덱스 미만으로는 선택하지 않음 (예: Stage0가 0..7을 로컬 처리하면 min_block=8)

    Args:
        throughputs: 각 블록별 처리량 배열
        num_blocks: 선택할 연속 블록 개수
        min_block: 선택 가능한 최소 시작 블록

    Returns:
        선택된 시작 블록 인덱스
    """
    if len(throughputs) < num_blocks:
        return max(0, int(min_block))

    max_i = len(throughputs) - num_blocks
    min_block = int(max(0, min(min_block, max_i)))

    options: List[Tuple[float, float, int]] = []
    for i in range(min_block, max_i + 1):
        segment = throughputs[i : i + num_blocks]
        options.append((float(np.min(segment)), float(np.mean(segment)), i))

    return min(options, key=lambda x: (x[0], x[1], x[2]))[2]


def choose_best_blocks(
    num_blocks: int,
    module_infos: List[RemoteModuleInfo],
    total_blocks: Optional[int] = None,
    min_block: int = 0,
) -> List[int]:
    """
    새 서버 조인 시 최적 블록 선택 (논문 Appendix D 규칙 1)

    Args:
        num_blocks: 서버가 담당할 블록 개수
        module_infos: 현재 시스템의 모든 모듈 정보
        total_blocks: 전체 블록 개수 (None이면 module_infos에서 추론)
        min_block: 선택 가능한 최소 시작 블록 (Stage0가 앞 블록을 로컬 처리하면 여기로 막아야 함)

    Returns:
        선택된 블록 인덱스 리스트
    """
    if total_blocks is None:
        max_block = 0
        for info in module_infos:
            try:
                block_idx = int(info.uid.split("_")[-1])
                max_block = max(max_block, block_idx)
            except (ValueError, IndexError):
                pass
        total_blocks = max_block + 1 if max_block > 0 else num_blocks

    spans = compute_spans(module_infos, min_state=ServerState.JOINING)
    throughputs = compute_throughputs(spans, total_blocks=total_blocks)

    start = _choose_best_start(throughputs, num_blocks, min_block=min_block)
    return list(range(start, start + num_blocks))


def _move_span(span: RemoteSpanInfo, new_start: int):
    """스팬 위치 이동"""
    span.start = new_start
    span.end = new_start + span.length


def should_choose_other_blocks(
    local_peer_id: PeerID,
    module_infos: List[RemoteModuleInfo],
    balance_quality: float = 0.75,
    total_blocks: Optional[int] = None,
    min_block: int = 0,
) -> bool:
    """
    동적 재조정 판단 (논문 Appendix D 규칙 2)

    주기적으로 호출하여 블록 재조정 필요 여부 판단

    Args:
        local_peer_id: 현재 서버의 PeerID
        module_infos: 현재 시스템의 모든 모듈 정보
        balance_quality: 품질 임계값 (0.75 = 25% 이상 개선 가능 시 재조정)
        total_blocks: 전체 블록 개수
        min_block: 선택 가능한 최소 시작 블록 (Stage0 로컬 구간 보호)

    Returns:
        True이면 블록 재조정 필요
    """
    if balance_quality > 1.0:
        return True  # 강제 재조정 (디버깅용)

    if total_blocks is None:
        max_block = 0
        for info in module_infos:
            try:
                block_idx = int(info.uid.split("_")[-1])
                max_block = max(max_block, block_idx)
            except (ValueError, IndexError):
                pass
        total_blocks = max_block + 1 if max_block > 0 else 32  # 기본값

    spans = compute_spans(module_infos, min_state=ServerState.JOINING)

    if len(spans) > 0:
        span_peer_ids = [str(pid)[:16] for pid in spans.keys()]
        logger.debug(f"Found {len(spans)} spans with peer_ids: {span_peer_ids}")
    else:
        logger.debug(f"No spans found in module_infos (total: {len(module_infos)})")

    logger.debug(f"Looking for local peer_id: {local_peer_id}")

    throughputs = compute_throughputs(spans, total_blocks=total_blocks)
    initial_throughput = float(np.min(throughputs)) if len(throughputs) > 0 else 0.0
    eps = 1e-3

    local_peer_id_str = str(local_peer_id)
    matching_peer_id = None
    for span_peer_id in spans.keys():
        if str(span_peer_id) == local_peer_id_str:
            matching_peer_id = span_peer_id
            break

    if matching_peer_id is None:
        logger.warning(f"Local peer {local_peer_id_str[:16]}... not found in spans (have {len(spans)} spans)")
        if len(spans) > 0:
            logger.warning(f"Available peer_ids in spans: {[str(pid)[:16] for pid in spans.keys()]}")
        return False

    local_span = spans[matching_peer_id]
    logger.debug(f"Found local span: start={local_span.start}, end={local_span.end}, throughput={local_span.throughput}")

    local_start = max(0, min(local_span.start, len(throughputs) - 1))
    local_end = min(local_span.end, len(throughputs))
    if local_end > local_start:
        throughputs[local_start : local_end] -= local_span.throughput * (1 + eps)

    if initial_throughput > eps and np.min(throughputs) <= 0:
        return False

    new_start = _choose_best_start(throughputs, local_span.length, min_block=min_block)

    if local_span.start == new_start:
        return False

    throughputs[local_span.start : local_span.end] += local_span.throughput * eps
    _move_span(local_span, new_start)
    throughputs[local_span.start : local_span.end] += local_span.throughput

    moved = True
    iteration = 0
    max_iterations = 10

    while moved and iteration < max_iterations:
        iteration += 1
        server_list = list(spans.keys())
        np.random.shuffle(server_list)

        moved = False
        for peer_id in server_list:
            span = spans[peer_id]
            throughputs[span.start : span.end] -= span.throughput * (1 + eps)

            new_start_candidate = _choose_best_start(throughputs, span.length, min_block=min_block)

            throughputs[span.start : span.end] += span.throughput * eps
            if span.start != new_start_candidate:
                _move_span(span, new_start_candidate)
                moved = True
            throughputs[span.start : span.end] += span.throughput

    new_throughput = float(np.min(throughputs))

    if new_throughput < initial_throughput or new_throughput < eps:
        return False

    actual_quality = initial_throughput / new_throughput
    logger.info(f"Swarm balance quality: {actual_quality * 100:.1f}% "
                f"(initial={initial_throughput:.2f}, new={new_throughput:.2f})")

    return actual_quality < balance_quality - eps
