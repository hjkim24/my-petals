import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, LlamaDecoderLayer
try:
    from transformers.cache_utils import Cache  # type: ignore
except Exception:
    Cache = None
from typing import Optional, Tuple

from .utils import extract_kv_tuple, default_position_ids
from block import WrappedLlamaBlock


class Stage0(nn.Module):
    """LLaMA-only Stage0 using tuple KV cache."""

    def __init__(self, full, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in Stage0.")

        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            self.embed_tokens = full.model.embed_tokens
            raw_layers = full.model.layers[:end]
        elif hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, "wpe", None)
            raw_layers = full.transformer.h[:end]
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = nn.ModuleList(raw_layers)
        self.config = full.config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = self.embed_tokens(input_ids)
        new_cache = []
        cache_obj: Optional[Cache] = None

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            if isinstance(layer, WrappedLlamaBlock):
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=None,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            else:
                layer_pos = position_ids if position_ids is not None else default_position_ids(layer_past, x.shape[1], x.device)
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=False,
                )
            x = out[0]
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                new_cache.append(kv)
                if Cache is not None and isinstance(out[-1] if len(out) > 1 else None, Cache):
                    cache_obj = out[-1]

        if not use_cache:
            return x, None
        if cache_obj is not None:
            return x, cache_obj
        return x, tuple(new_cache)


class StageSegment(nn.Module):
    """LLaMA-only middle segment using tuple KV cache."""

    def __init__(self, full, start: int, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageSegment.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers[start:end]
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h[start:end]
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = nn.ModuleList(raw_layers)
        self.config = full.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        new_cache = []
        cache_obj: Optional[Cache] = None

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            if isinstance(layer, WrappedLlamaBlock):
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=None,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            else:
                layer_pos = position_ids if position_ids is not None else default_position_ids(layer_past, x.shape[1], x.device)
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=False,
                )
            x = out[0]
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                new_cache.append(kv)
                if Cache is not None and isinstance(out[-1] if len(out) > 1 else None, Cache):
                    cache_obj = out[-1]

        if not use_cache:
            return x, None
        if cache_obj is not None:
            return x, cache_obj
        return x, tuple(new_cache)


class StageLast(nn.Module):
    """LLaMA-only last stage using tuple KV cache."""

    def __init__(self, full, start: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageLast.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers[start:]
            if hasattr(full.model, "norm"):
                self.norm = full.model.norm
            elif hasattr(full.model, "final_layer_norm"):
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h[start:]
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = nn.ModuleList(raw_layers)

        self.lm_head = full.lm_head
        self.config = full.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        new_cache = []
        cache_obj: Optional[Cache] = None

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            if isinstance(layer, WrappedLlamaBlock):
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=None,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            else:
                layer_pos = position_ids if position_ids is not None else default_position_ids(layer_past, x.shape[1], x.device)
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    output_attentions=False,
                )
            x = out[0]
            if use_cache:
                kv = extract_kv_tuple(out, layer_idx=i)
                new_cache.append(kv)
                if Cache is not None and isinstance(out[-1] if len(out) > 1 else None, Cache):
                    cache_obj = out[-1]

        x = self.norm(x)
        logits = self.lm_head(x)
        if not use_cache:
            return logits, None
        if cache_obj is not None:
            return logits, cache_obj
        return logits, tuple(new_cache)


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    dtype=torch.float16,
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    """
    full = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    try:
        full.config.use_cache = True
    except Exception:
        pass
    full.eval()

    def _wrap_layers(module_list: nn.ModuleList):
        new_layers = []
        for idx, layer in enumerate(module_list):
            if isinstance(layer, LlamaDecoderLayer):
                wrapped = WrappedLlamaBlock(full.config)
                wrapped.load_state_dict(layer.state_dict())
                new_layers.append(wrapped)
            else:
                new_layers.append(layer)
        return nn.ModuleList(new_layers)

    def _prune_layers(obj, start_idx, end_idx):
        if hasattr(obj, "model") and hasattr(obj.model, "layers"):
            obj.model.layers = _wrap_layers(nn.ModuleList(obj.model.layers[start_idx:end_idx]))
        elif hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
            obj.transformer.h = _wrap_layers(nn.ModuleList(obj.transformer.h[start_idx:end_idx]))
        else:
            raise ValueError(f"Unsupported model architecture for pruning: {type(obj)}")

    if role == "stage0":
        _prune_layers(full, 0, end)
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "segment":
        _prune_layers(full, start, end)
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "last":
        _prune_layers(full, start, None)
        # keep norm/head
    else:
        raise ValueError(f"Unknown role: {role}")

    full = full.to(device)
    return full
