import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .utils import default_position_ids

logger = logging.getLogger(__name__)

# Prefer petals optimized block (uses rotary_emb cache) when available
try:
    from petals.llama.block import OptimizedLlamaDecoderLayer  # type: ignore
    OPT_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    OptimizedLlamaDecoderLayer = None
    OPT_AVAILABLE = False
    logger.warning(f"OptimizedLlamaDecoderLayer not available ({e}), using vanilla LlamaDecoderLayer.")

try:
    from transformers.cache_utils import Cache, DynamicCache  # type: ignore
except Exception:
    Cache, DynamicCache = None, None


def _convert_layers(raw_layers: nn.ModuleList, config) -> nn.ModuleList:
    """
    Convert HF layers to OptimizedLlamaDecoderLayer if available.
    Otherwise keep as-is to stay close to HF reference.
    """
    converted = []
    for idx, layer in enumerate(raw_layers):
        if OPT_AVAILABLE:
            if isinstance(layer, OptimizedLlamaDecoderLayer):
                converted.append(layer)
                continue
            if isinstance(layer, LlamaDecoderLayer):
                opt_layer = OptimizedLlamaDecoderLayer(config)
                missing, unexpected = opt_layer.load_state_dict(layer.state_dict(), strict=False)
                if missing or unexpected:
                    logger.warning(
                        f"Layer {idx}: optimized load missing={len(missing)}, unexpected={len(unexpected)}"
                    )
                converted.append(opt_layer)
                continue
        converted.append(layer)
    return nn.ModuleList(converted)


def _to_cache(past):
    """Convert legacy tuple to DynamicCache if available, else return as-is."""
    if past is None:
        return None
    if Cache is not None and isinstance(past, Cache):
        return past
    if DynamicCache is not None and isinstance(past, (tuple, list)):
        try:
            return DynamicCache.from_legacy_cache(past)
        except Exception:
            return past
    return past


def _from_cache(present):
    """Convert Cache to legacy tuple if needed."""
    if Cache is not None and isinstance(present, Cache):
        try:
            return present.to_legacy_cache()
        except Exception:
            return present
    return present


class Stage0(nn.Module):
    """LLaMA-only Stage0; keep Cache end-to-end (no manual recompute)."""

    def __init__(self, full, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in Stage0.")

        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            self.embed_tokens = full.model.embed_tokens
            raw_layers = full.model.layers  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, "wpe", None)
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        logger.info(f"Stage0 initialized with {len(self.layers)} layers (end={end})")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = self.embed_tokens(input_ids)
        cache_obj = None
        tuple_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            out = layer(
                x,
                attention_mask=None,
                position_ids=layer_pos,
                past_key_value=_to_cache(layer_past),
                use_cache=use_cache,
                output_attentions=False,
            )
            x = out[0]
            if use_cache:
                present = out[-1] if len(out) > 1 else None
                present = _from_cache(present)
                # if present is None:
                #     logger.warning(f"Stage0: layer {i} returned no KV cache")
                # else:
                #     cache_len = present[0].shape[-2] if isinstance(present, tuple) else "cache_obj"
                #     logger.info(f"Stage0 layer {i} present cache_len={cache_len}")
                tuple_cache.append(present)

        if not use_cache:
            return x, None
        return x, tuple(tuple_cache)


class StageSegment(nn.Module):
    """LLaMA-only middle segment; keep Cache end-to-end."""

    def __init__(self, full, start: int, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageSegment.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        if len(self.layers) == 0:
            logger.warning(f"StageSegment initialized with 0 layers (start={start}, end={end})")
        else:
            logger.info(f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end})")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        tuple_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            out = layer(
                x,
                attention_mask=None,
                position_ids=layer_pos,
                past_key_value=_to_cache(layer_past),
                use_cache=use_cache,
                output_attentions=False,
            )
            x = out[0]
            if use_cache:
                present = out[-1] if len(out) > 1 else None
                present = _from_cache(present)
                if present is None:
                    logger.warning(f"StageSegment: layer {i} returned no KV cache")
                # else:
                    # cache_len = present[0].shape[-2] if isinstance(present, tuple) else "cache_obj"
                    # logger.info(f"StageSegment layer {i} present cache_len={cache_len}")
                tuple_cache.append(present)

        if not use_cache:
            return x, None
        return x, tuple(tuple_cache)


class StageLast(nn.Module):
    """LLaMA-only last stage; keep Cache end-to-end."""

    def __init__(self, full, start: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        if "llama" not in model_type and "mistral" not in model_type and "mixtral" not in model_type:
            raise ValueError("Only LLaMA-style models are supported in StageLast.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers  # already pruned in load_stage_model
            if hasattr(full.model, "norm"):
                self.norm = full.model.norm
            elif hasattr(full.model, "final_layer_norm"):
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h  # already pruned in load_stage_model
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.lm_head = full.lm_head
        self.config = full.config
        logger.info(f"StageLast initialized with {len(self.layers)} layers (start={start})")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
    ):
        x = hidden_states
        tuple_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
            out = layer(
                x,
                attention_mask=None,
                position_ids=layer_pos,
                past_key_value=_to_cache(layer_past),
                use_cache=use_cache,
                output_attentions=False,
            )
            x = out[0]
            if use_cache:
                present = out[-1] if len(out) > 1 else None
                present = _from_cache(present)
                if present is None:
                    logger.warning(f"StageLast: layer {i} returned no KV cache")
                # else:
                #     cache_len = present[0].shape[-2] if isinstance(present, tuple) else "cache_obj"
                #     logger.info(f"StageLast layer {i} present cache_len={cache_len}")
                tuple_cache.append(present)

        x = self.norm(x)
        # Ensure x dtype matches lm_head weight dtype to avoid dtype mismatch
        if hasattr(self.lm_head, 'weight') and self.lm_head.weight is not None:
            x = x.to(dtype=self.lm_head.weight.dtype)
        logits = self.lm_head(x)
        if not use_cache:
            return logits, None
        return logits, tuple(tuple_cache)


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    dtype=torch.float16,
    quantization_config=None,
    load_in_4bit=False,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=None,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type=None,
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    quantization_config: Optional BitsAndBytesConfig for int4/int8 quantization (newer transformers)
    load_in_4bit/load_in_8bit: Direct parameters for quantization (compatible with transformers 4.43+)
    """
    # Determine quantization mode
    use_quantization = load_in_4bit or load_in_8bit
    
    if use_quantization:
        # Build kwargs for quantization using direct parameters (compatible with transformers 4.43+)
        # Direct parameters are more reliable than BitsAndBytesConfig for transformers 4.43.1
        quant_kwargs = {"low_cpu_mem_usage": True}
        
        if load_in_4bit:
            quant_kwargs["load_in_4bit"] = True
            if bnb_4bit_compute_dtype is not None:
                quant_kwargs["bnb_4bit_compute_dtype"] = bnb_4bit_compute_dtype
            if bnb_4bit_use_double_quant:
                quant_kwargs["bnb_4bit_use_double_quant"] = True
            if bnb_4bit_quant_type is not None:
                quant_kwargs["bnb_4bit_quant_type"] = bnb_4bit_quant_type
        elif load_in_8bit:
            quant_kwargs["load_in_8bit"] = True
        
        # For quantization, device_map="auto" is typically needed
        # But transformers 4.43 may have issues, so we try both ways
        try:
            full = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                **quant_kwargs,
            )
        except (TypeError, AttributeError) as e:
            # Fallback: try without device_map (will handle device manually later)
            logger.warning(f"Failed to load quantized model with device_map='auto': {e}, trying without device_map")
            full = AutoModelForCausalLM.from_pretrained(
                model_name,
                **quant_kwargs,
            )
    else:
        # Normal mode: use torch_dtype
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

    def _prune_layers(obj, start_idx, end_idx):
        if hasattr(obj, "model") and hasattr(obj.model, "layers"):
            obj.model.layers = nn.ModuleList(obj.model.layers[start_idx:end_idx])
        elif hasattr(obj, "transformer") and hasattr(obj.transformer, "h"):
            obj.transformer.h = nn.ModuleList(obj.transformer.h[start_idx:end_idx])
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

    # Log resulting layer counts to catch empty segments early
    if hasattr(full, "model") and hasattr(full.model, "layers"):
        num_layers = len(full.model.layers)
    elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
        num_layers = len(full.transformer.h)
    else:
        num_layers = -1
    quantization_status = "enabled" if (quantization_config is not None or load_in_4bit or load_in_8bit) else "disabled"
    logger.info(f"load_stage_model: role={role}, layers={num_layers}, start={start}, end={end}, quantization={quantization_status}")
    if num_layers == 0:
        raise ValueError(f"Pruned model has 0 layers for role={role} (start={start}, end={end}). Check --splits.")

    # Move model to device
    # For quantized models with device_map="auto", they're already on device, but we may need to handle manually
    use_quantization = quantization_config is not None or load_in_4bit or load_in_8bit
    if use_quantization:
        # Quantized models with device_map="auto" are already on device
        # Only move if device_map was not used or if we need to change device
        if not hasattr(full, 'hf_device_map') or full.hf_device_map is None:
            # device_map was not used, try to move manually
            try:
                full = full.to(device)
            except Exception as e:
                logger.warning(f"Failed to move quantized model to {device}: {e}. Model may already be on correct device.")
                # Try to verify device placement
                if hasattr(full, 'device') and str(full.device) != str(device):
                    logger.warning(f"Quantized model device mismatch: expected {device}, but model is on {full.device}")
        else:
            logger.info(f"Quantized model loaded with device_map, device placement: {full.hf_device_map}")
    else:
        full = full.to(device)
    
    return full
