"""
Activation cache for fault tolerance (C-2: Dual Cache).
Stores past inputs (activations) sent to each stage for server-side KV cache restoration.
"""
from typing import Dict, List, Optional, Tuple
import torch
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class ActivationCache:
    """
    Client-side cache for storing past inputs (activations) sent to each stage.
    Used for restoring server-side KV cache when a server fails.
    
    논문 Algorithm 1 line 18: cache[server].append(inputs)
    """
    
    def __init__(
        self,
        max_len: int = 2048,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            max_len: Maximum number of activations to store per stage (to prevent memory explosion)
            dtype: Optional dtype for stored activations (default: keep original dtype)
            device: Device to store activations on (default: CPU to save GPU memory)
        """
        self.max_len = max_len
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        
        # cache[(session_id, stage_key)] -> List[Tensor]
        # Each tensor is the hidden_states input sent to that stage
        self._cache: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    
    def append(self, session_id: str, stage_key: str, activation: torch.Tensor):
        """
        Append an activation (hidden_states input) to the cache for a specific stage.
        
        Args:
            session_id: Session identifier
            stage_key: Stage key (e.g., "mini_petals:stage1")
            activation: Hidden states tensor [batch, seq_len, hidden_size]
        """
        key = (session_id, stage_key)
        
        if key not in self._cache:
            self._cache[key] = []
        
        # Move to CPU and optionally convert dtype to save memory
        activation_cpu = activation.cpu().detach()
        if self.dtype is not None and activation_cpu.dtype != self.dtype:
            activation_cpu = activation_cpu.to(dtype=self.dtype)
        
        self._cache[key].append(activation_cpu)
        
        # Enforce max_len: remove oldest if exceeded
        if len(self._cache[key]) > self.max_len:
            self._cache[key] = self._cache[key][-self.max_len:]
    
    def get(self, session_id: str, stage_key: str) -> List[torch.Tensor]:
        """
        Get all cached activations for a session and stage.
        
        Args:
            session_id: Session identifier
            stage_key: Stage key
            
        Returns:
            List of activation tensors in chronological order
        """
        key = (session_id, stage_key)
        return self._cache.get(key, [])
    
    def clear(self, session_id: Optional[str] = None, stage_key: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            session_id: If provided, only clear entries for this session
            stage_key: If provided, only clear entries for this stage
        """
        if session_id is None and stage_key is None:
            self._cache.clear()
        else:
            keys_to_remove = []
            for key in self._cache.keys():
                sess_id, stg_key = key
                if session_id is not None and sess_id != session_id:
                    continue
                if stage_key is not None and stg_key != stage_key:
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
    
    def get_all_stages(self, session_id: str) -> Dict[str, List[torch.Tensor]]:
        """
        Get cached activations for all stages in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict mapping stage_key -> List[Tensor]
        """
        result = {}
        for (sess_id, stg_key), activations in self._cache.items():
            if sess_id == session_id:
                result[stg_key] = activations
        return result


# Global singleton instance (can be accessed from RpcTransport)
_global_cache: Optional[ActivationCache] = None


def get_cache() -> ActivationCache:
    """Get or create the global ActivationCache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ActivationCache()
    return _global_cache


def set_cache(cache: ActivationCache):
    """Set the global ActivationCache instance."""
    global _global_cache
    _global_cache = cache

