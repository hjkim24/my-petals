import torch
from typing import Optional, Tuple, Iterable


def extract_kv_tuple(output: Iterable) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a transformer layer output, return (key, value) tuple if present.
    Expected LLaMA-style outputs:
      - (hidden_states, past_key_value)
      - (hidden_states, attentions, past_key_value) when output_attentions=True
    """
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return None
    candidate = output[-1] if len(output) > 2 else output[1]
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        if all(isinstance(t, torch.Tensor) for t in candidate):
            return candidate  # (key, value)
    return None


def default_position_ids(layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]], seq_len: int, device) -> torch.Tensor:
    """
    Build position_ids using past KV length if available; otherwise start at 0.
    """
    past_len = 0
    if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
        if layer_past[0] is not None and layer_past[0].dim() >= 3:
            past_len = layer_past[0].shape[2]
    return torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long).unsqueeze(0)
