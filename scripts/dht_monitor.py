#!/usr/bin/env python3
"""
DHT Network Monitor
모델을 로드하지 않고 DHT 네트워크에 등록된 노드들을 실시간으로 모니터링합니다.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hivemind import DHT, get_dht_time
from hivemind.utils.logging import get_logger
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


def _format_initial_peers(dht_initial_peers: str) -> list:
    """Format DHT initial peers from comma-separated string."""
    if not dht_initial_peers:
        return []
    peers = []
    for p in dht_initial_peers.split(","):
        p = p.strip()
        if not p:
            continue
        # Require full multiaddr with peer ID to avoid invalid p2p multiaddr errors
        if "/p2p/" in p:
            peers.append(p)
        elif ":" in p:
            logger.warning(
                f"Initial peer '{p}' is missing '/p2p/<peer_id>'. "
                "Use the full multiaddr printed by Stage1 (e.g., /ip4/127.0.0.1/tcp/8000/p2p/<peer_id>)."
            )
            # Try to use it anyway, but warn
            peers.append(p)
        else:
            peers.append(p)
    return peers


def _get_local_ip() -> str:
    """Get local IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def query_dht_nodes(dht: DHT, stage_key: str) -> List[Dict[str, Any]]:
    """
    Query DHT for all nodes registered under a stage key.
    Returns a list of node information dictionaries.
    """
    nodes = []
    
    try:
        # hivemind 버전에 따라 return_metadata 지원 여부가 다름
        try:
            res = dht.get(stage_key)
        except TypeError:
            try:
                res = dht.get(stage_key, return_metadata=True)
            except TypeError:
                res = dht.get(stage_key)
        
        if res is None:
            return nodes
        
        # res가 다양한 형태일 수 있음
        value = None
        if hasattr(res, 'value'):
            value = res.value
        elif isinstance(res, dict):
            value = res
        elif isinstance(res, (list, tuple)) and len(res) > 0:
            value = res[0] if isinstance(res[0], dict) else res
        
        if value is None:
            return nodes
        
        # hivemind 버전에 따라 value가 dict(subkey->entry)일 수 있음
        if isinstance(value, dict):
            # subkey가 있는 경우 (여러 서버)
            for subk, v in value.items():
                entry = v
                # ValueWithExpiration 객체 처리
                if hasattr(v, 'value'):
                    entry = v.value
                elif isinstance(v, tuple) and len(v) > 0:
                    entry = v[0]
                elif isinstance(v, dict) and "value" in v:
                    entry = v.get("value", v)
                
                if not isinstance(entry, dict):
                    continue
                
                peer_id_str = entry.get("peer_id") or str(subk)
                if not peer_id_str:
                    continue
                
                node_info = {
                    "subkey": str(subk),
                    "peer_id": peer_id_str,
                    "stage": entry.get("stage", "unknown"),
                    "timestamp": entry.get("timestamp", 0),
                    "throughput": entry.get("throughput", 0.0),
                    "benchmark_time": entry.get("benchmark_time", 0),
                    "p2p_maddrs": entry.get("p2p_maddrs", []),
                }
                nodes.append(node_info)
        else:
            # 단일 entry 형태 (subkey 없음)
            if isinstance(value, dict):
                peer_id_str = value.get("peer_id")
                if peer_id_str:
                    node_info = {
                        "subkey": "N/A",
                        "peer_id": peer_id_str,
                        "stage": value.get("stage", "unknown"),
                        "timestamp": value.get("timestamp", 0),
                        "throughput": value.get("throughput", 0.0),
                        "benchmark_time": value.get("benchmark_time", 0),
                        "p2p_maddrs": value.get("p2p_maddrs", []),
                    }
                    nodes.append(node_info)
    
    except Exception as e:
        logger.error(f"Error querying {stage_key}: {e}")
    
    return nodes


def format_timestamp(ts: int) -> str:
    """Format DHT timestamp to readable string."""
    if ts == 0:
        return "N/A"
    try:
        # DHT time is typically in seconds since epoch
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def print_node_table(stage_key: str, nodes: List[Dict[str, Any]]):
    """Print a formatted table of nodes for a stage."""
    if not nodes:
        print(f"  {stage_key}: No nodes registered")
        return
    
    print(f"\n{'='*100}")
    print(f"{stage_key}: {len(nodes)} node(s) registered")
    print(f"{'='*100}")
    print(f"{'Peer ID':<20} {'Stage':<8} {'Throughput':<15} {'Timestamp':<20} {'Maddrs Count':<15}")
    print(f"{'-'*100}")
    
    for node in nodes:
        peer_id_short = node["peer_id"][:16] + "..." if len(node["peer_id"]) > 16 else node["peer_id"]
        throughput_str = f"{node['throughput']:.1f} tok/s" if node['throughput'] > 0 else "N/A"
        ts_str = format_timestamp(node["timestamp"])
        maddrs_count = len(node["p2p_maddrs"])
        
        print(f"{peer_id_short:<20} {node['stage']:<8} {throughput_str:<15} {ts_str:<20} {maddrs_count:<15}")
        
        # Print maddrs if available
        if node["p2p_maddrs"]:
            for maddr in node["p2p_maddrs"][:2]:  # Show first 2 maddrs
                print(f"  └─ {maddr}")
            if len(node["p2p_maddrs"]) > 2:
                print(f"  └─ ... and {len(node['p2p_maddrs']) - 2} more")


async def monitor_dht(
    dht_initial_peers: str,
    dht_port: int,
    refresh_interval: float = 5.0,
):
    """
    Monitor DHT network for registered nodes.
    
    Args:
        dht_initial_peers: Comma-separated list of initial DHT peers
        dht_port: Port for DHT
        refresh_interval: How often to refresh the node list (seconds)
    """
    # Format initial peers
    initial_peers_list = _format_initial_peers(dht_initial_peers)
    local_ip = _get_local_ip()
    
    # Check if port is available
    logger.info(f"Checking if port {dht_port} is available on {local_ip}...")
    if not _is_port_available(local_ip, dht_port):
        logger.warning(f"Port {dht_port} is not available. This might be due to:")
        logger.warning(f"  1. Another process is using the port")
        logger.warning(f"  2. Previous DHT instance didn't fully release the port (TIME_WAIT state)")
        logger.warning(f"  3. Port is in TIME_WAIT state (usually clears in 30-60 seconds)")
        logger.warning(f"")
        logger.warning(f"Solutions:")
        logger.warning(f"  - Wait 30-60 seconds and try again")
        logger.warning(f"  - Use a different port with --dht_port")
        logger.warning(f"  - Check for zombie processes: lsof -i :{dht_port} or netstat -an | grep {dht_port}")
        logger.warning(f"")
        logger.warning(f"Attempting to bind anyway (may fail)...")
    
    # Initialize DHT (read-only, no model loading)
    logger.info(f"Connecting to DHT network on {local_ip}:{dht_port}")
    if initial_peers_list:
        logger.info(f"Initial peers ({len(initial_peers_list)}):")
        for i, peer in enumerate(initial_peers_list, 1):
            logger.info(f"  {i}. {peer}")
    else:
        logger.warning("No initial peers provided. DHT may not connect to existing network.")
    
    try:
        logger.info("Initializing DHT...")
        dht = DHT(
            start=True,
            initial_peers=initial_peers_list if initial_peers_list else None,
            host_maddrs=[f"/ip4/{local_ip}/tcp/{dht_port}"],
        )
        logger.info(f"DHT initialized. Peer ID: {dht.peer_id}")
        
        # Wait a bit for connections to establish
        logger.info("Waiting for DHT connections to establish...")
        await asyncio.sleep(2.0)
        
        # Check if we have any peers
        try:
            visible_maddrs = dht.get_visible_maddrs()
            logger.info(f"DHT visible maddrs: {visible_maddrs}")
        except Exception as e:
            logger.warning(f"Could not get visible maddrs: {e}")
        
    except Exception as e:
        logger.error(f"Failed to initialize DHT: {e}")
        logger.error("This might be due to:")
        logger.error("  1. Invalid initial_peers format (should be full multiaddr with /p2p/<peer_id>)")
        logger.error("  2. Network connectivity issues")
        logger.error("  3. Bootstrap peers are not reachable")
        logger.error("  4. Port conflicts")
        if initial_peers_list:
            logger.error(f"  Provided initial_peers: {initial_peers_list}")
        raise
    logger.info(f"Monitoring stage keys: mini_petals:stage1, mini_petals:stage2, mini_petals:stage3")
    logger.info(f"Refresh interval: {refresh_interval}s")
    logger.info("Press Ctrl+C to stop\n")
    
    stage_keys = ["mini_petals:stage1", "mini_petals:stage2", "mini_petals:stage3"]
    
    try:
        iteration = 0
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n{'#'*100}")
            print(f"# Iteration {iteration} - {current_time}")
            print(f"{'#'*100}")
            
            all_nodes = {}
            for stage_key in stage_keys:
                nodes = query_dht_nodes(dht, stage_key)
                all_nodes[stage_key] = nodes
                print_node_table(stage_key, nodes)
            
            # Summary
            total_nodes = sum(len(nodes) for nodes in all_nodes.values())
            print(f"\n{'='*100}")
            print(f"SUMMARY: Total {total_nodes} node(s) across all stages")
            for stage_key in stage_keys:
                count = len(all_nodes[stage_key])
                if count > 0:
                    avg_throughput = sum(n["throughput"] for n in all_nodes[stage_key] if n["throughput"] > 0) / count
                    print(f"  {stage_key}: {count} node(s), avg throughput: {avg_throughput:.1f} tok/s" if avg_throughput > 0 else f"  {stage_key}: {count} node(s)")
            print(f"{'='*100}\n")
            
            await asyncio.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}", exc_info=True)
    finally:
        try:
            logger.info("Shutting down DHT...")
            dht.shutdown()
            logger.info("DHT disconnected")
            # Give the port some time to be released
            logger.info("Waiting for port to be released...")
            await asyncio.sleep(1.0)
        except Exception as e:
            logger.warning(f"Error shutting down DHT: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="DHT Network Monitor - Monitor registered nodes without loading models"
    )
    parser.add_argument(
        "--dht_initial_peers",
        type=str,
        default="",
        help="Comma-separated list of initial DHT peers (e.g., full multiaddrs /ip4/host/tcp/port/p2p/PeerID)"
    )
    parser.add_argument(
        "--dht_port",
        type=int,
        default=8000,
        help="Port for DHT (default: 8000)"
    )
    parser.add_argument(
        "--refresh_interval",
        type=float,
        default=5.0,
        help="How often to refresh the node list in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Run async monitor
    asyncio.run(monitor_dht(
        dht_initial_peers=args.dht_initial_peers,
        dht_port=args.dht_port,
        refresh_interval=args.refresh_interval,
    ))


if __name__ == "__main__":
    main()