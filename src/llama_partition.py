import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from huggingface_hub import hf_hub_download, HfApi

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    # Don't warn here, will warn in function if needed

from .utils import default_position_ids

logger = logging.getLogger(__name__)


class QuantType(Enum):
    NONE = 0
    INT8 = 1  # 8-bit as in the LLM.int8() paper
    NF4 = 2  # 4-bit as in the QLoRA paper


def quantize_module(model: nn.Module, *, quant_type: QuantType, compute_dtype: Optional[torch.dtype] = None) -> nn.Module:
    """
    Quantize a model module by replacing Linear layers with quantized versions.
    This is based on the original Petals implementation.
    
    Args:
        model: The model module to quantize
        quant_type: Type of quantization (INT8 or NF4)
        compute_dtype: Compute dtype for NF4 quantization (e.g., torch.float16).
                      If None, uses default (torch.float32). Setting to float16
                      matches input dtype and avoids warnings.
    
    Returns:
        The quantized model (modified in-place)
    """
    # Import bitsandbytes only when necessary
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install it with: pip install bitsandbytes"
        )

    quantized_in_this_call = []
    
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            quantize_module(module, quant_type=quant_type, compute_dtype=compute_dtype)

        # Skip critical projection layers used for KV cache correctness
        # q_proj / k_proj / v_proj / o_proj must stay in higher precision
        # TODO: Temporarily allowing attention quantization for testing - can revert if issues occur
        skip_names = {"lm_head", "score"}  # , "q_proj", "k_proj", "v_proj", "o_proj"}  # Commented out to enable attention quantization

        if isinstance(module, torch.nn.Linear) and n not in skip_names:
            # Check if weight is on meta device (from device_map="auto")
            if hasattr(module, 'weight') and hasattr(module.weight, 'device'):
                if module.weight.device.type == "meta":
                    logger.warning(f"Linear layer '{n}' is on meta device, skipping quantization (will be loaded later)")
                    continue
                elif module.weight.device.type != "cpu":
                    # If somehow on GPU, move to CPU first
                    logger.warning(
                        f"Linear layer '{n}' is on {module.weight.device}, moving to CPU for quantization"
                    )
                    # Move the actual module in the model
                    model._modules[n] = module.cpu()
                    module = model._modules[n]
            
            if quant_type == QuantType.INT8:
                model._modules[n] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,  # Default from the LLM.int8() paper
                )
                model._modules[n].weight = bnb.nn.Int8Params(
                    module.weight.data, requires_grad=False, has_fp16_weights=False
                ).to(module.weight.dtype)
            elif quant_type == QuantType.NF4:
                compress_statistics = True
                # compute_dtype이 지정되지 않으면 기본값 사용 (기본값은 float32)
                # float16으로 설정하면 입력 dtype과 일치하여 경고 방지 및 성능 향상
                # LinearNF4는 위치 인자로 input_features, output_features를 받음
                if compute_dtype is not None:
                    model._modules[n] = bnb.nn.LinearNF4(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compress_statistics=compress_statistics,
                        compute_dtype=compute_dtype,
                    )
                else:
                    model._modules[n] = bnb.nn.LinearNF4(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compress_statistics=compress_statistics,
                    )
                model._modules[n].weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                    blocksize=64,
                    compress_statistics=compress_statistics,
                ).to(module.weight.dtype)
            else:
                raise ValueError(f"Unsupported quant_type='{quant_type}'")
            model._modules[n].bias = module.bias
            quantized_in_this_call.append(n)
    
    # Log quantization summary (only for top-level calls to avoid duplicates)
    if quantized_in_this_call and len(list(model.named_children())) > 5:  # Heuristic: top-level if many children
        logger.info(
            f"Quantization applied: {len(quantized_in_this_call)} Linear layers quantized to {quant_type.name}"
        )
        logger.debug(f"Quantized layer names: {quantized_in_this_call[:10]}{'...' if len(quantized_in_this_call) > 10 else ''}")
    
    return model

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


def _has_quantized_layers(layer: nn.Module) -> bool:
    """
    Check if a layer contains quantized Linear layers (bitsandbytes).
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    
    for module in layer.modules():
        if isinstance(module, (bnb.nn.LinearNF4, bnb.nn.Linear8bitLt)):
            return True
    return False


def _count_quantized_modules(layer: nn.Module) -> dict:
    """Count quantized modules in a layer."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return {"int8": 0, "nf4": 0, "total": 0}
    
    int8_count = 0
    nf4_count = 0
    
    for module in layer.modules():
        if isinstance(module, bnb.nn.Linear8bitLt):
            int8_count += 1
        elif isinstance(module, bnb.nn.LinearNF4):
            nf4_count += 1
    
    return {"int8": int8_count, "nf4": nf4_count, "total": int8_count + nf4_count}


def _convert_layers(raw_layers: nn.ModuleList, config) -> nn.ModuleList:
    """
    Convert HF layers to OptimizedLlamaDecoderLayer if available.
    Otherwise keep as-is to stay close to HF reference.
    
    For quantized layers, copy modules directly instead of using load_state_dict
    to avoid shape mismatch issues with quantized weight formats.
    """
    converted = []
    quantized_converted = 0
    non_quantized_converted = 0
    already_optimized = 0
    
    for idx, layer in enumerate(raw_layers):
        if OPT_AVAILABLE:
            if isinstance(layer, OptimizedLlamaDecoderLayer):
                converted.append(layer)
                already_optimized += 1
                continue
            
            if isinstance(layer, LlamaDecoderLayer):
                if _has_quantized_layers(layer):
                    # For quantized layers, create OptimizedLlamaDecoderLayer and copy modules directly
                    # to avoid shape mismatch from load_state_dict
                    try:
                        opt_layer = OptimizedLlamaDecoderLayer(config)
                        orig_attn = layer.self_attn
                        opt_attn = opt_layer.self_attn
                        
                        # Copy attention projection layers (q_proj, k_proj, v_proj, o_proj)
                        # These are not quantized (excluded from quantization), so safe to copy
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                            if hasattr(orig_attn, proj_name) and hasattr(opt_attn, proj_name):
                                orig_proj = getattr(orig_attn, proj_name)
                                setattr(opt_attn, proj_name, orig_proj)
                        
                        # Copy rotary embedding (if exists)
                        if hasattr(orig_attn, 'rotary_emb') and hasattr(opt_attn, 'rotary_emb'):
                            opt_attn.rotary_emb = orig_attn.rotary_emb
                        
                        # Copy MLP (may contain quantized layers)
                        opt_layer.mlp = layer.mlp
                        
                        # Copy layernorms (not quantized, safe to copy)
                        opt_layer.input_layernorm = layer.input_layernorm
                        opt_layer.post_attention_layernorm = layer.post_attention_layernorm
                        
                        # Log quantization status
                        quant_stats = _count_quantized_modules(opt_layer)
                        logger.info(
                            f"Layer {idx}: converted quantized layer to OptimizedLlamaDecoderLayer "
                            f"(quantized modules: {quant_stats['total']} total, "
                            f"{quant_stats['int8']} INT8, {quant_stats['nf4']} NF4)"
                        )
                        converted.append(opt_layer)
                        quantized_converted += 1
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Layer {idx}: failed to convert quantized layer to OptimizedLlamaDecoderLayer: {e}. "
                            "Keeping original layer."
                        )
                        converted.append(layer)
                        continue
                else:
                    # Non-quantized: use standard conversion with load_state_dict
                    opt_layer = OptimizedLlamaDecoderLayer(config)
                    missing, unexpected = opt_layer.load_state_dict(layer.state_dict(), strict=False)
                    if missing or unexpected:
                        logger.warning(
                            f"Layer {idx}: optimized load missing={len(missing)}, unexpected={len(unexpected)}"
                        )
                    converted.append(opt_layer)
                    non_quantized_converted += 1
                    continue
        converted.append(layer)
    
    # Log conversion summary
    if quantized_converted > 0 or non_quantized_converted > 0:
        logger.info(
            f"Layer conversion summary: {quantized_converted} quantized layers converted, "
            f"{non_quantized_converted} non-quantized layers converted, "
            f"{already_optimized} already optimized"
        )
    
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
        # Support LLaMA, Mistral, Mixtral, and BLOOM models
        is_supported = ("llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "bloom" in model_type)
        if not is_supported:
            raise ValueError(f"Unsupported model type '{model_type}' in Stage0. Supported: LLaMA, Mistral, Mixtral, BLOOM.")
        
        # Store model type for forward pass
        self.is_bloom = "bloom" in model_type
        
        # For BLOOM models, store build_alibi_tensor function and num_heads
        if self.is_bloom:
            try:
                from transformers.models.bloom.modeling_bloom import build_alibi_tensor
                self.build_alibi_tensor = build_alibi_tensor
                self.num_heads = getattr(full.config, "num_attention_heads", None)
                if self.num_heads is None:
                    logger.warning("BLOOM model config missing num_attention_heads, ALiBi generation may fail")
            except ImportError:
                logger.warning("Could not import build_alibi_tensor, ALiBi generation will be disabled")
                self.build_alibi_tensor = None
                self.num_heads = None

        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            # LLaMA-style
            self.embed_tokens = full.model.embed_tokens
            raw_layers = full.model.layers  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            # GPT-2 style (with wte)
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, "wpe", None)
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "word_embeddings"):
            # BLOOM style (with word_embeddings)
            self.embed_tokens = full.transformer.word_embeddings
            self.pos_embed = None  # BLOOM doesn't have positional embeddings
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported architecture: {type(full)}.")

        self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        
        # Log layer status
        quantized_layers = sum(1 for layer in self.layers if _has_quantized_layers(layer))
        if OPT_AVAILABLE and OptimizedLlamaDecoderLayer is not None:
            optimized_layers = sum(1 for layer in self.layers if isinstance(layer, OptimizedLlamaDecoderLayer))
            logger.info(
                f"Stage0 initialized with {len(self.layers)} layers (end={end}): "
                f"{quantized_layers} quantized, {optimized_layers} OptimizedLlamaDecoderLayer"
            )
        else:
            logger.info(
                f"Stage0 initialized with {len(self.layers)} layers (end={end}): "
                f"{quantized_layers} quantized"
            )

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
        
        # For BLOOM models, attention_mask is required - create default if None
        if self.is_bloom and attention_mask is None:
            batch_size, seq_length = x.shape[:2]
            # Calculate total sequence length including past
            total_seq_length = seq_length
            if past_key_values is not None and len(past_key_values) > 0:
                # Check first layer's past_key_value to get past length
                first_past = past_key_values[0] if isinstance(past_key_values[0], (tuple, list)) else None
                if first_past is not None and len(first_past) >= 1:
                    past_key = first_past[0]
                    if past_key is not None and past_key.ndim >= 3:
                        total_seq_length += past_key.shape[2]
            # Create default attention mask (all ones = no masking)
            attention_mask = torch.ones(
                (batch_size, total_seq_length),
                device=x.device,
                dtype=torch.bool
            )

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM models don't accept position_ids or past_key_value parameters
            # BLOOM uses layer_past instead and requires alibi for ALiBi (Attention with Linear Biases)
            if self.is_bloom:
                # BLOOM: pass attention_mask, alibi, layer_past, and use_cache
                # Generate ALiBi tensor for BLOOM
                alibi = None
                if self.build_alibi_tensor is not None and self.num_heads is not None:
                    try:
                        batch_size, seq_length = x.shape[:2]
                        
                        # Calculate total key length (current sequence + past if applicable)
                        total_seq_length = seq_length
                        if layer_past is not None:
                            # layer_past is tuple of (key, value), key shape is [batch, num_heads, past_seq_len, head_dim]
                            if isinstance(layer_past, (tuple, list)) and len(layer_past) >= 1:
                                past_key = layer_past[0]
                                if past_key is not None and past_key.ndim >= 3:
                                    past_seq_len = past_key.shape[2]
                                    total_seq_length += past_seq_len
                        
                        # Create or use attention_mask for ALiBi generation
                        if attention_mask is not None:
                            # attention_mask might be 2D (batch_size, seq_length) or 4D (batch_size, 1, 1, seq_length)
                            # For ALiBi, we need 2D shape (batch_size, seq_length)
                            if attention_mask.ndim == 4:
                                alibi_attention_mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
                            elif attention_mask.ndim == 2:
                                alibi_attention_mask = attention_mask
                            else:
                                # Fallback: create a dummy mask
                                alibi_attention_mask = torch.ones(
                                    (batch_size, total_seq_length),
                                    device=x.device,
                                    dtype=torch.bool
                                )
                            
                            # If attention_mask is shorter than total_seq_length, pad it
                            if alibi_attention_mask.shape[1] < total_seq_length:
                                # Pad with ones (assuming past tokens are valid)
                                padding = torch.ones(
                                    (batch_size, total_seq_length - alibi_attention_mask.shape[1]),
                                    device=x.device,
                                    dtype=alibi_attention_mask.dtype
                                )
                                alibi_attention_mask = torch.cat([padding, alibi_attention_mask], dim=1)
                            elif alibi_attention_mask.shape[1] > total_seq_length:
                                # Truncate to total_seq_length (shouldn't happen, but handle it)
                                alibi_attention_mask = alibi_attention_mask[:, -total_seq_length:]
                        else:
                            # No attention_mask provided, create a dummy one
                            alibi_attention_mask = torch.ones(
                                (batch_size, total_seq_length),
                                device=x.device,
                                dtype=torch.bool
                            )
                        
                        # build_alibi_tensor(attention_mask, num_heads, dtype)
                        alibi = self.build_alibi_tensor(
                            alibi_attention_mask,
                            self.num_heads,
                            dtype=x.dtype,
                        )
                        # Move alibi to correct device
                        alibi = alibi.to(x.device)
                        
                    except Exception as e:
                        logger.warning(f"Failed to build ALiBi tensor: {e}, using None")
                        import traceback
                        logger.debug(f"ALiBi generation traceback: {traceback.format_exc()}")
                        alibi = None
                
                # Build kwargs dict
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "alibi": alibi,
                    "use_cache": use_cache,
                    "output_attentions": False,
                }
                # Add layer_past only if it's not None
                if layer_past is not None:
                    layer_kwargs["layer_past"] = layer_past
                
                # Filter out None values from kwargs
                layer_kwargs = {k: v for k, v in layer_kwargs.items() if v is not None}
                
                out = layer(x, **layer_kwargs)
            else:
                # LLaMA/Mistral/Mixtral: use standard forward with position_ids
                layer_pos = position_ids if position_ids is not None else default_position_ids(
                    layer_past, x.shape[1], x.device
                )
                # Use standard layer forward
                # OptimizedLlamaDecoderLayer (including quantized ones) properly returns KV cache
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=_to_cache(layer_past),
                    use_cache=use_cache,
                    output_attentions=False,
                )
            
            # Validate output structure
            if not isinstance(out, (tuple, list)) or len(out) == 0:
                raise RuntimeError(f"Stage0: layer {i} returned invalid output: {type(out)}")
            
            x = out[0]
            if use_cache:
                if len(out) < 2:
                    logger.error(
                        f"Stage0: layer {i} output too short for use_cache=True "
                        f"(out_len={len(out)}, expected >= 2, layer_type={type(layer).__name__})"
                    )
                    present = None
                else:
                    present = out[-1]  # Last element should be past_key_value
                    present = _from_cache(present)
            
            if use_cache:
                # Check if layer returned KV cache
                if present is None:
                    logger.warning(
                        f"Stage0: layer {i} returned no KV cache "
                        f"(layer_type={type(layer).__name__}, quantized={_has_quantized_layers(layer)})"
                    )
                elif isinstance(present, (tuple, list)) and len(present) == 2:
                    if present[0] is None or present[1] is None:
                        logger.warning(
                            f"Stage0: layer {i} KV cache contains None "
                            f"(key={present[0] is not None}, value={present[1] is not None})"
                        )
                else:
                    logger.debug(
                        f"Stage0: layer {i} KV cache format: {type(present)}, "
                        f"len={len(present) if isinstance(present, (tuple, list)) else 'N/A'}"
                    )
                tuple_cache.append(present)

        if not use_cache:
            return x, None
        return x, tuple(tuple_cache)


class StageSegment(nn.Module):
    """LLaMA-only middle segment; keep Cache end-to-end."""

    def __init__(self, full, start: int, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        # Support LLaMA, Mistral, Mixtral, and BLOOM models
        is_supported = ("llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "bloom" in model_type)
        if not is_supported:
            raise ValueError(f"Unsupported model type '{model_type}' in StageSegment. Supported: LLaMA, Mistral, Mixtral, BLOOM.")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported LLaMA architecture: {type(full)}.")

        self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        self.is_bloom = "bloom" in model_type
        
        # For BLOOM models, we need to prepare ALiBi generation
        if self.is_bloom:
            try:
                from transformers.models.bloom.modeling_bloom import build_alibi_tensor
                self.build_alibi_tensor = build_alibi_tensor
                # Get number of heads from config
                self.num_heads = getattr(full.config, "num_attention_heads", None)
                if self.num_heads is None:
                    logger.warning("Could not determine num_attention_heads for BLOOM model, ALiBi generation may fail")
            except ImportError:
                logger.warning("Could not import build_alibi_tensor from transformers, ALiBi generation may fail")
                self.build_alibi_tensor = None
                self.num_heads = None
        
        if len(self.layers) == 0:
            logger.warning(f"StageSegment initialized with 0 layers (start={start}, end={end})")
        else:
            # Log layer status
            quantized_layers = sum(1 for layer in self.layers if _has_quantized_layers(layer))
            if OPT_AVAILABLE and OptimizedLlamaDecoderLayer is not None:
                optimized_layers = sum(1 for layer in self.layers if isinstance(layer, OptimizedLlamaDecoderLayer))
                logger.info(
                    f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}): "
                    f"{quantized_layers} quantized, {optimized_layers} OptimizedLlamaDecoderLayer, "
                    f"model_type={model_type}"
                )
            else:
                logger.info(
                    f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}): "
                    f"{quantized_layers} quantized, model_type={model_type}"
                )

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
        
        # For BLOOM models, attention_mask is required - create default if None
        if self.is_bloom and attention_mask is None:
            batch_size, seq_length = x.shape[:2]
            # Calculate total sequence length including past
            total_seq_length = seq_length
            if past_key_values is not None and len(past_key_values) > 0:
                # Check first layer's past_key_value to get past length
                first_past = past_key_values[0] if isinstance(past_key_values[0], (tuple, list)) else None
                if first_past is not None and len(first_past) >= 1:
                    past_key = first_past[0]
                    if past_key is not None and past_key.ndim >= 3:
                        total_seq_length += past_key.shape[2]
            # Create default attention mask (all ones = no masking)
            attention_mask = torch.ones(
                (batch_size, total_seq_length),
                device=x.device,
                dtype=torch.bool
            )

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM models don't accept position_ids or past_key_value parameters
            # BLOOM uses layer_past instead and requires alibi for ALiBi (Attention with Linear Biases)
            if self.is_bloom:
                # BLOOM: pass attention_mask, alibi, layer_past, and use_cache
                # Generate ALiBi tensor for BLOOM
                # build_alibi_tensor signature: build_alibi_tensor(attention_mask, num_heads, dtype)
                # attention_mask must be a tensor of shape (batch_size, seq_length)
                alibi = None
                if self.build_alibi_tensor is not None and self.num_heads is not None:
                    try:
                        batch_size, seq_length = x.shape[:2]
                        
                        # Calculate total key length (current sequence + past if applicable)
                        total_seq_length = seq_length
                        if layer_past is not None:
                            # layer_past is tuple of (key, value), key shape is [batch, num_heads, past_seq_len, head_dim]
                            if isinstance(layer_past, (tuple, list)) and len(layer_past) >= 1:
                                past_key = layer_past[0]
                                if past_key is not None and past_key.ndim >= 3:
                                    past_seq_len = past_key.shape[2]
                                    total_seq_length += past_seq_len
                        
                        # Create or use attention_mask for ALiBi generation
                        # If attention_mask is provided, use it (should already have correct shape)
                        # Otherwise, create a dummy one with shape (batch_size, total_seq_length)
                        if attention_mask is not None:
                            # attention_mask might be 2D (batch_size, seq_length) or 4D (batch_size, 1, 1, seq_length)
                            # For ALiBi, we need 2D shape (batch_size, seq_length)
                            if attention_mask.ndim == 4:
                                # Extract the last dimension
                                alibi_attention_mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
                            elif attention_mask.ndim == 2:
                                alibi_attention_mask = attention_mask
                            else:
                                # Fallback: create a dummy mask
                                alibi_attention_mask = torch.ones(
                                    (batch_size, total_seq_length),
                                    device=x.device,
                                    dtype=torch.bool
                                )
                            
                            # If attention_mask is shorter than total_seq_length, pad it
                            if alibi_attention_mask.shape[1] < total_seq_length:
                                # Pad with ones (assuming past tokens are valid)
                                padding = torch.ones(
                                    (batch_size, total_seq_length - alibi_attention_mask.shape[1]),
                                    device=x.device,
                                    dtype=alibi_attention_mask.dtype
                                )
                                alibi_attention_mask = torch.cat([padding, alibi_attention_mask], dim=1)
                            elif alibi_attention_mask.shape[1] > total_seq_length:
                                # Truncate to total_seq_length (shouldn't happen, but handle it)
                                alibi_attention_mask = alibi_attention_mask[:, -total_seq_length:]
                        else:
                            # No attention_mask provided, create a dummy one
                            alibi_attention_mask = torch.ones(
                                (batch_size, total_seq_length),
                                device=x.device,
                                dtype=torch.bool
                            )
                        
                        # build_alibi_tensor(attention_mask, num_heads, dtype)
                        # attention_mask: (batch_size, seq_length) tensor
                        # num_heads: int
                        # dtype: torch.dtype
                        alibi = self.build_alibi_tensor(
                            alibi_attention_mask,
                            self.num_heads,
                            dtype=x.dtype,
                        )
                        
                    except Exception as e:
                        logger.warning(f"Failed to build ALiBi tensor: {e}, using None")
                        import traceback
                        logger.debug(f"ALiBi generation traceback: {traceback.format_exc()}")
                        alibi = None
                
                # Build kwargs dict
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "alibi": alibi,
                    "use_cache": use_cache,
                    "output_attentions": False,
                }
                # Add layer_past only if it's not None
                if layer_past is not None:
                    layer_kwargs["layer_past"] = layer_past
                out = layer(x, **layer_kwargs)
            else:
                # LLaMA/Mistral/Mixtral: pass position_ids
                layer_pos = position_ids if position_ids is not None else default_position_ids(
                    layer_past, x.shape[1], x.device
                )
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    position_ids=layer_pos,
                    past_key_value=_to_cache(layer_past),
                    use_cache=use_cache,
                    output_attentions=False,
                )
            
            # Validate output structure
            if not isinstance(out, (tuple, list)) or len(out) == 0:
                raise RuntimeError(f"StageSegment: layer {i} returned invalid output: {type(out)}")
            
            x = out[0]
            if use_cache:
                if len(out) < 2:
                    logger.error(
                        f"StageSegment: layer {i} output too short for use_cache=True "
                        f"(out_len={len(out)}, expected >= 2, layer_type={type(layer).__name__})"
                    )
                    present = None
                else:
                    present = out[-1]  # Last element should be past_key_value
                    present = _from_cache(present)
                
                # Check if layer returned KV cache
                if present is None:
                    logger.warning(
                        f"StageSegment: layer {i} returned no KV cache "
                        f"(layer_type={type(layer).__name__})"
                    )
                elif isinstance(present, (tuple, list)) and len(present) == 2:
                    if present[0] is None or present[1] is None:
                        logger.warning(
                            f"StageSegment: layer {i} KV cache contains None "
                            f"(key={present[0] is not None}, value={present[1] is not None})"
                        )
                tuple_cache.append(present)

        if not use_cache:
            return x, None
        return x, tuple(tuple_cache)


class StageLast(nn.Module):
    """LLaMA-only last stage; keep Cache end-to-end."""

    def __init__(self, full, start: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        # Support LLaMA, Mistral, Mixtral, and BLOOM models
        is_supported = ("llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "bloom" in model_type)
        if not is_supported:
            raise ValueError(f"Unsupported model type '{model_type}' in StageLast. Supported: LLaMA, Mistral, Mixtral, BLOOM.")
        
        # Store model type for forward pass
        self.is_bloom = "bloom" in model_type
        
        # For BLOOM models, store build_alibi_tensor function and num_heads
        if self.is_bloom:
            try:
                from transformers.models.bloom.modeling_bloom import build_alibi_tensor
                self.build_alibi_tensor = build_alibi_tensor
                self.num_heads = getattr(full.config, "num_attention_heads", None)
                if self.num_heads is None:
                    logger.warning("BLOOM model config missing num_attention_heads, ALiBi generation may fail")
            except ImportError:
                logger.warning("Could not import build_alibi_tensor, ALiBi generation will be disabled")
                self.build_alibi_tensor = None
                self.num_heads = None

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
        
        # Log layer status
        quantized_layers = sum(1 for layer in self.layers if _has_quantized_layers(layer))
        if OPT_AVAILABLE and OptimizedLlamaDecoderLayer is not None:
            optimized_layers = sum(1 for layer in self.layers if isinstance(layer, OptimizedLlamaDecoderLayer))
            logger.info(
                f"StageLast initialized with {len(self.layers)} layers (start={start}): "
                f"{quantized_layers} quantized, {optimized_layers} OptimizedLlamaDecoderLayer"
            )
        else:
            logger.info(
                f"StageLast initialized with {len(self.layers)} layers (start={start}): "
                f"{quantized_layers} quantized"
            )

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
        
        # For BLOOM models, attention_mask is required - create default if None
        if self.is_bloom and attention_mask is None:
            batch_size, seq_length = x.shape[:2]
            # Calculate total sequence length including past
            total_seq_length = seq_length
            if past_key_values is not None and len(past_key_values) > 0:
                # Check first layer's past_key_value to get past length
                first_past = past_key_values[0] if isinstance(past_key_values[0], (tuple, list)) else None
                if first_past is not None and len(first_past) >= 1:
                    past_key = first_past[0]
                    if past_key is not None and past_key.ndim >= 3:
                        total_seq_length += past_key.shape[2]
            # Create default attention mask (all ones = no masking)
            attention_mask = torch.ones(
                (batch_size, total_seq_length),
                device=x.device,
                dtype=torch.bool
            )

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM models don't accept position_ids or past_key_value parameters
            # BLOOM uses layer_past instead and requires alibi for ALiBi (Attention with Linear Biases)
            if self.is_bloom:
                # BLOOM: pass attention_mask, alibi, layer_past, and use_cache
                # Generate ALiBi tensor for BLOOM
                # build_alibi_tensor signature: build_alibi_tensor(attention_mask, num_heads, dtype)
                # attention_mask must be a tensor of shape (batch_size, seq_length)
                alibi = None
                if self.build_alibi_tensor is not None and self.num_heads is not None:
                    try:
                        batch_size, seq_length = x.shape[:2]
                        
                        # Calculate total key length (current sequence + past if applicable)
                        total_seq_length = seq_length
                        if layer_past is not None:
                            # layer_past is tuple of (key, value), key shape is [batch, num_heads, past_seq_len, head_dim]
                            if isinstance(layer_past, (tuple, list)) and len(layer_past) >= 1:
                                past_key = layer_past[0]
                                if past_key is not None and past_key.ndim >= 3:
                                    past_seq_len = past_key.shape[2]
                                    total_seq_length += past_seq_len
                        
                        # Create or use attention_mask for ALiBi generation
                        # If attention_mask is provided, use it (should already have correct shape)
                        # Otherwise, create a dummy one with shape (batch_size, total_seq_length)
                        if attention_mask is not None:
                            # attention_mask might be 2D (batch_size, seq_length) or 4D (batch_size, 1, 1, seq_length)
                            # For ALiBi, we need 2D shape (batch_size, seq_length)
                            if attention_mask.ndim == 4:
                                # Extract the last dimension
                                alibi_attention_mask = attention_mask.squeeze(1).squeeze(1)  # (batch_size, seq_length)
                            elif attention_mask.ndim == 2:
                                alibi_attention_mask = attention_mask
                            else:
                                # Fallback: create a dummy mask
                                alibi_attention_mask = torch.ones(
                                    (batch_size, total_seq_length),
                                    device=x.device,
                                    dtype=torch.bool
                                )
                            
                            # If attention_mask is shorter than total_seq_length, pad it
                            if alibi_attention_mask.shape[1] < total_seq_length:
                                # Pad with ones (assuming past tokens are valid)
                                padding = torch.ones(
                                    (batch_size, total_seq_length - alibi_attention_mask.shape[1]),
                                    device=x.device,
                                    dtype=alibi_attention_mask.dtype
                                )
                                alibi_attention_mask = torch.cat([padding, alibi_attention_mask], dim=1)
                            elif alibi_attention_mask.shape[1] > total_seq_length:
                                # Truncate to total_seq_length (shouldn't happen, but handle it)
                                alibi_attention_mask = alibi_attention_mask[:, -total_seq_length:]
                        else:
                            # No attention_mask provided, create a dummy one
                            alibi_attention_mask = torch.ones(
                                (batch_size, total_seq_length),
                                device=x.device,
                                dtype=torch.bool
                            )
                        
                        # build_alibi_tensor(attention_mask, num_heads, dtype)
                        # attention_mask: (batch_size, seq_length) tensor
                        # num_heads: int
                        # dtype: torch.dtype
                        alibi = self.build_alibi_tensor(
                            alibi_attention_mask,
                            self.num_heads,
                            dtype=x.dtype,
                        )
                        # Move alibi to correct device
                        alibi = alibi.to(x.device)
                        
                    except Exception as e:
                        logger.warning(f"Failed to build ALiBi tensor: {e}, using None")
                        import traceback
                        logger.debug(f"ALiBi generation traceback: {traceback.format_exc()}")
                        alibi = None
                
                # Build kwargs dict
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "alibi": alibi,
                    "use_cache": use_cache,
                    "output_attentions": False,
                }
                # Add layer_past only if it's not None
                if layer_past is not None:
                    layer_kwargs["layer_past"] = layer_past
                
                # Filter out None values from kwargs
                layer_kwargs = {k: v for k, v in layer_kwargs.items() if v is not None}
                
                out = layer(x, **layer_kwargs)
            else:
                # LLaMA/Mistral/Mixtral: use standard forward with position_ids
                layer_pos = position_ids if position_ids is not None else default_position_ids(
                    layer_past, x.shape[1], x.device
                )
                # Use standard layer forward
                # OptimizedLlamaDecoderLayer (including quantized ones) properly returns KV cache
                out = layer(
                    x,
                    attention_mask=None,
                    position_ids=layer_pos,
                    past_key_value=_to_cache(layer_past),
                    use_cache=use_cache,
                    output_attentions=False,
                )
            
            # Validate output structure
            if not isinstance(out, (tuple, list)) or len(out) == 0:
                raise RuntimeError(f"StageLast: layer {i} returned invalid output: {type(out)}")
            
            x = out[0]
            if use_cache:
                if len(out) < 2:
                    logger.error(
                        f"StageLast: layer {i} output too short for use_cache=True "
                        f"(out_len={len(out)}, expected >= 2, layer_type={type(layer).__name__})"
                    )
                    present = None
                else:
                    present = out[-1]  # Last element should be past_key_value
                    present = _from_cache(present)
            
            if use_cache:
                # Check if layer returned KV cache
                if present is None:
                    logger.warning(
                        f"StageLast: layer {i} returned no KV cache "
                        f"(layer_type={type(layer).__name__}, quantized={_has_quantized_layers(layer)})"
                    )
                elif isinstance(present, (tuple, list)) and len(present) == 2:
                    if present[0] is None or present[1] is None:
                        logger.warning(
                            f"StageLast: layer {i} KV cache contains None "
                            f"(key={present[0] is not None}, value={present[1] is not None})"
                        )
                tuple_cache.append(present)

        x = self.norm(x)
        # Ensure x dtype matches lm_head weight dtype to avoid dtype mismatch
        if hasattr(self.lm_head, 'weight') and self.lm_head.weight is not None:
            x = x.to(dtype=self.lm_head.weight.dtype)
        logits = self.lm_head(x)
        if not use_cache:
            return logits, None
        return logits, tuple(tuple_cache)


def _create_stage_config(config: PretrainedConfig, role: str, start: int, end: Optional[int]) -> PretrainedConfig:
    """
    Create a modified config with only the required number of layers.
    This reduces memory usage when creating model structure.
    
    Args:
        config: Original model config
        role: Stage role ("stage0", "segment", "last")
        start: Start layer index
        end: End layer index (None for "last")
    
    Returns:
        Modified config with reduced num_hidden_layers
    """
    # Config 복사
    stage_config = config.__class__(**config.to_dict())
    
    # 필요한 레이어 수 계산
    if role == "stage0":
        num_layers_needed = end
    elif role == "segment":
        num_layers_needed = end - start
    elif role == "last":
        num_layers_needed = config.num_hidden_layers - start
    else:
        num_layers_needed = config.num_hidden_layers
    
    # num_hidden_layers 수정 (메모리 절약을 위해 작은 구조 생성)
    stage_config.num_hidden_layers = num_layers_needed
    
    logger.info(f"Created stage config: role={role}, original_layers={config.num_hidden_layers}, "
                f"stage_layers={num_layers_needed}, start={start}, end={end}")
    
    return stage_config


def _remap_state_dict_keys(state_dict: Dict[str, torch.Tensor], role: str, start: int, end: Optional[int]) -> Dict[str, torch.Tensor]:
    """
    Remap state_dict keys to match the smaller model structure.
    
    For segment/last roles, layer indices need to be remapped:
    - Original: model.layers.10.* -> Remapped: model.layers.0.*
    - Original: model.layers.11.* -> Remapped: model.layers.1.*
    etc.
    
    Args:
        state_dict: Original state_dict with full layer indices
        role: Stage role ("stage0", "segment", "last")
        start: Start layer index in original model
        end: End layer index in original model
    
    Returns:
        Remapped state_dict with layer indices starting from 0
    """
    remapped = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # BLOOM 모델 키 처리: h.{layer_idx}.* 형식
        if key.startswith("h."):
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    if role == "stage0":
                        # stage0는 0부터 시작하므로 prefix만 추가
                        new_key = f"transformer.h.{layer_idx}." + ".".join(parts[2:])
                    elif role == "segment" or role == "last":
                        # segment/last는 인덱스를 0부터 시작하도록 재매핑
                        if layer_idx >= start:
                            new_layer_idx = layer_idx - start
                            new_key = f"transformer.h.{new_layer_idx}." + ".".join(parts[2:])
                        else:
                            continue  # Skip keys before start
                except ValueError:
                    pass
        elif key.startswith("word_embeddings"):
            # BLOOM embedding
            if role == "stage0":
                new_key = "transformer." + key
        elif key.startswith("model.layers."):
            # LLaMA 모델 키 처리
            if role == "segment" or role == "last":
                parts = key.split(".")
                if len(parts) >= 3:
                    try:
                        layer_idx = int(parts[2])
                        if layer_idx >= start:
                            new_layer_idx = layer_idx - start
                            new_key = f"model.layers.{new_layer_idx}." + ".".join(parts[3:])
                    except ValueError:
                        pass
        
        remapped[new_key] = value
    
    return remapped


def _get_required_tensor_keys(role: str, start: int, end: Optional[int], config) -> Set[str]:
    """
    Get the set of tensor key prefixes required for a given stage.
    Returns a set of tensor key prefixes that need to be loaded.
    Supports both LLaMA-style (model.layers) and BLOOM-style (transformer.h) architectures.
    """
    required_prefixes = set()
    num_layers = config.num_hidden_layers
    
    # Detect model architecture from config
    model_type = getattr(config, "model_type", "").lower()
    is_bloom = "bloom" in model_type
    is_llama = "llama" in model_type or "mistral" in model_type or "mixtral" in model_type
    
    # Determine layer prefix based on architecture
    if is_bloom:
        # BLOOM model uses "h.0." format (without "transformer." prefix in weight_map)
        layer_prefix = "h."
        embed_prefix = "word_embeddings"
        norm_prefix = "ln_f"
        head_prefix = "lm_head"
    elif is_llama:
        layer_prefix = "model.layers."
        embed_prefix = "model.embed_tokens"
        norm_prefix = "model.norm"
        head_prefix = "lm_head"
    else:
        # Default to LLaMA-style (backward compatibility)
        layer_prefix = "model.layers."
        embed_prefix = "model.embed_tokens"
        norm_prefix = "model.norm"
        head_prefix = "lm_head"
        logger.warning(f"Unknown model type '{model_type}', assuming LLaMA-style architecture")
    
    # Embedding layers
    if role == "stage0":
        required_prefixes.add(embed_prefix)
        # Layers from 0 to end
        for i in range(end):
            required_prefixes.add(f"{layer_prefix}{i}.")
    elif role == "segment":
        # Layers from start to end
        for i in range(start, end):
            required_prefixes.add(f"{layer_prefix}{i}.")
    elif role == "last":
        # Layers from start to end
        for i in range(start, num_layers):
            required_prefixes.add(f"{layer_prefix}{i}.")
        required_prefixes.add(norm_prefix)
        required_prefixes.add(head_prefix)
    
    logger.info(f"Detected model type: {model_type}, using layer prefix: {layer_prefix}")
    return required_prefixes


def _load_selective_weights(
    model_name: str,
    required_keys: Set[str],
    cache_dir: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load only the required weights from sharded safetensors files.
    Returns a state_dict with only the required tensors.
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for selective loading")
    
    try:
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    except ImportError:
        SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
        SAFE_WEIGHTS_NAME = "model.safetensors"
    
    api = HfApi()
    
    # Try to get index file
    try:
        index_path = hf_hub_download(
            repo_id=model_name,
            filename=SAFE_WEIGHTS_INDEX_NAME,
            cache_dir=cache_dir,
        )
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        state_dict = {}
        
        # Debug: Log a few example keys from weight_map
        sample_keys = list(weight_map.keys())[:5]
        logger.info(f"Sample tensor keys from weight_map: {sample_keys}")
        logger.info(f"Required prefixes: {sorted(list(required_keys))[:10]}...")
        
        # Find which shard files contain our required keys
        shard_files = set()
        matched_keys = []
        for tensor_key, shard_file in weight_map.items():
            # Match keys that start with any required prefix
            for req_prefix in required_keys:
                if tensor_key.startswith(req_prefix):
                    shard_files.add(shard_file)
                    if len(matched_keys) < 10:  # Log first 10 matches for debugging
                        matched_keys.append((tensor_key, shard_file))
                    break
        
        if matched_keys:
            logger.info(f"Found {len(shard_files)} shard files with matching keys")
            logger.info(f"Sample matched keys: {matched_keys[:5]}")
        else:
            logger.warning(f"No matching keys found! This may indicate a key format mismatch.")
            logger.warning(f"First few weight_map keys: {list(weight_map.keys())[:10]}")
        
        shard_files = sorted(shard_files)  # 정렬하여 일관된 순서 보장
        total_shards = len(shard_files)
        logger.info(f"Selective loading: {total_shards} shard files needed for {len(required_keys)} required prefixes")
        logger.info(f"Selective loading: Starting download of {total_shards} shard files...")
        
        # Download and load only required shard files
        for idx, shard_file in enumerate(shard_files, 1):
            logger.info(f"Selective loading: Downloading shard {idx}/{total_shards}: {shard_file}")
            shard_path = hf_hub_download(
                repo_id=model_name,
                filename=shard_file,
                cache_dir=cache_dir,
            )
            logger.info(f"Selective loading: Shard {idx}/{total_shards} downloaded/loaded from cache: {shard_path}")
            
            logger.info(f"Selective loading: Loading tensors from shard {idx}/{total_shards}...")
            tensors_loaded_from_shard = 0
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_key in f.keys():
                    # Check if this tensor is needed (matches any required prefix)
                    if any(tensor_key.startswith(req_prefix) for req_prefix in required_keys):
                        state_dict[tensor_key] = f.get_tensor(tensor_key)
                        tensors_loaded_from_shard += 1
            logger.info(f"Selective loading: Loaded {tensors_loaded_from_shard} tensors from shard {idx}/{total_shards}")
        
        logger.info(f"Selective loading: Completed! Loaded {len(state_dict)} total tensors from {total_shards} shard files")
        return state_dict
        
    except Exception as e:
        logger.warning(f"Selective loading failed: {e}, falling back to full download")
        # Fallback: return empty dict to trigger full download
        return {}


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    dtype=torch.float16,
    quant_type: QuantType = QuantType.NONE,
    use_selective_loading: bool = True,  # Enabled: uses modified config to prevent OOM
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    quant_type: Quantization type (QuantType.NONE, QuantType.INT8, or QuantType.NF4)
                If quantization is enabled, model will be loaded on CPU, quantized, then moved to device
    use_selective_loading: If True, try to download only required layers (requires safetensors)
    """
    # Load config first (always needed)
    config = AutoConfig.from_pretrained(model_name)
    
    # Try selective loading if enabled and safetensors is available
    selective_state_dict = None
    if use_selective_loading and SAFETENSORS_AVAILABLE:
        try:
            required_keys = _get_required_tensor_keys(role, start, end, config)
            logger.info(f"Attempting selective loading for {len(required_keys)} required keys")
            selective_state_dict = _load_selective_weights(model_name, required_keys)
            
            if not selective_state_dict:
                logger.info("Selective loading returned empty dict, falling back to full download")
                selective_state_dict = None
        except Exception as e:
            logger.warning(f"Selective loading failed: {e}, falling back to full download")
            selective_state_dict = None
    
    # If selective loading failed or is disabled, use full download
    if selective_state_dict is None:
        logger.info("Loading full model (will prune after loading)")
        # Always load model on CPU first (required for quantization, and safe for normal loading)
        # For large models like BLOOM-176B, use memory limits and disk offloading
        import psutil
        import tempfile
        import shutil
        
        available_memory = psutil.virtual_memory().available
        # BLOOM-176B는 매우 크므로 더 보수적으로 사용 가능한 메모리의 25%만 사용
        # 전체 모델 로딩 중에는 더 많은 메모리가 필요하므로 더 낮게 설정
        # 사용 가능한 메모리의 25%를 사용하되, 최소 80GB는 보장
        min_required_mb = 80 * 1024  # 최소 80GB (더 보수적으로)
        calculated_mb = int(available_memory * 0.25 / (1024 * 1024))
        max_memory_mb = max(min_required_mb, calculated_mb)
        max_memory = {"cpu": f"{max_memory_mb}MiB"}
        logger.info(f"Available memory: {available_memory / (1024**3):.1f}GB, limiting to {max_memory_mb / 1024:.1f}GB (25% of available)")
        
        # 디스크 오프로딩을 위한 임시 디렉토리 생성
        offload_folder = tempfile.mkdtemp(prefix="model_offload_")
        logger.info(f"Using disk offloading folder: {offload_folder}")
        
        try:
            # Try device_map="cpu" first, fallback to manual CPU loading if not supported
            try:
                # Use accelerate's device_map="auto" with max_memory for better memory management
                # This allows automatic offloading to disk when memory is constrained
                # Use device_map="cpu" directly to avoid meta tensor issues
                # device_map="auto" can leave some tensors on meta device which causes issues with quantization
                full = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="cpu",  # Explicitly load on CPU for quantization compatibility
                    max_memory=max_memory,  # 메모리 사용량 제한
                    offload_folder=offload_folder,  # 디스크 오프로딩
                    cache_dir=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None,
                    use_safetensors=True,  # safetensors 사용 (더 안전하고 메모리 효율적)
                )
            except TypeError:
                # Fallback for older transformers versions that don't support device_map
                logger.warning("device_map not supported, loading model and moving to CPU manually")
                full = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    max_memory=max_memory,  # 메모리 사용량 제한
                    offload_folder=offload_folder,  # 디스크 오프로딩
                    cache_dir=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None,
                    use_safetensors=True,
                )
                full = full.cpu()
        except Exception as e:
            # 오프로딩 폴더 정리
            try:
                shutil.rmtree(offload_folder)
            except:
                pass
            raise e
    else:
        # Build model structure with selective weights
        logger.info("Building model structure with selectively loaded weights")
        
        # Create a modified config with only required layers (OOM 방지)
        logger.info("Step 1/4: Creating stage config...")
        stage_config = _create_stage_config(config, role, start, end)
        logger.info("Step 1/4: Stage config created successfully")
        
        # Create model structure with smaller config (메모리 절약)
        # Use init_empty_weights to create structure without allocating weights (more memory efficient)
        logger.info("Step 2/4: Creating model structure from config (this may take a while)...")
        try:
            from accelerate import init_empty_weights
            with init_empty_weights():
                full = AutoModelForCausalLM.from_config(stage_config)
            logger.info("Step 2/4: Model structure created successfully (empty weights)")
        except ImportError:
            # Fallback if accelerate is not available
            logger.warning("accelerate not available, using regular from_config (may use more memory)")
            full = AutoModelForCausalLM.from_config(stage_config)
            logger.info("Step 2/4: Model structure created successfully")
        
        # Remap state_dict keys to match the smaller model structure
        # (segment/last의 경우 layer indices를 0부터 시작하도록 재매핑)
        logger.info("Step 3/4: Remapping state_dict keys to match model structure...")
        remapped_state_dict = _remap_state_dict_keys(selective_state_dict, role, start, end)
        logger.info(f"Step 3/4: Remapped {len(remapped_state_dict)} tensor keys")
        
        # Load only the required weights
        # Use assign=True when loading into meta device tensors (from init_empty_weights)
        logger.info("Step 4/4: Loading weights into model structure (this may take a while)...")
        try:
            # Try with assign=True first (for meta device tensors from init_empty_weights)
            # assign=True materializes meta tensors and assigns values directly
            missing_keys, unexpected_keys = full.load_state_dict(remapped_state_dict, strict=False, assign=True)
            logger.info("Step 4/4: Weights loaded successfully (using assign=True for meta tensors)")
            
            # After assign=True, verify that all tensors are materialized
            # If still on meta, try manual materialization
            try:
                sample_param = next(full.parameters())
                if sample_param.device.type == "meta":
                    logger.warning("Tensors still on meta device after assign=True, attempting manual materialization...")
                    # assign=True should have worked, but if not, we need to manually materialize
                    # This should not happen, but handle it gracefully
            except StopIteration:
                pass
        except TypeError:
            # Fallback if assign parameter is not supported (older PyTorch versions)
            logger.warning("assign=True not supported, trying without assign parameter")
            missing_keys, unexpected_keys = full.load_state_dict(remapped_state_dict, strict=False)
            logger.info("Step 4/4: Weights loaded successfully")
        
        if missing_keys:
            logger.warning(f"Missing keys in selective loading: {len(missing_keys)} keys")
            logger.debug(f"First 10 missing keys: {missing_keys[:10]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in selective loading: {len(unexpected_keys)} keys")
        
        # Move to CPU and set dtype
        # After assign=True, check if ANY tensors are still on meta device
        # We need to check ALL parameters and buffers, not just the first one
        has_meta = False
        try:
            # Check all parameters for meta device
            for param in full.parameters():
                if param.device.type == "meta":
                    has_meta = True
                    break
            # Also check buffers
            if not has_meta:
                for buffer in full.buffers():
                    if buffer.device.type == "meta":
                        has_meta = True
                        break
        except StopIteration:
            pass
        
        if has_meta:
            # Still on meta device, need to materialize manually using accelerate
            logger.warning("Model still has meta tensors after load_state_dict, materializing to CPU...")
            from accelerate.utils import set_module_tensor_to_device
            
            # Recursively materialize all parameters and buffers
            def materialize_module(module, prefix=""):
                materialized_count = 0
                # Materialize parameters
                for name, param in module.named_parameters(recurse=False):
                    full_name = f"{prefix}.{name}" if prefix else name
                    if param.device.type == "meta":
                        # Find corresponding tensor in state_dict
                        if full_name in remapped_state_dict:
                            tensor = remapped_state_dict[full_name]
                            set_module_tensor_to_device(module, name, "cpu", value=tensor)
                            materialized_count += 1
                        else:
                            # If not in state_dict, create empty tensor on CPU with same shape/dtype
                            try:
                                # Get shape and dtype from meta tensor
                                shape = param.shape
                                dtype = param.dtype
                                empty_tensor = torch.zeros(shape, dtype=dtype, device="cpu")
                                set_module_tensor_to_device(module, name, "cpu", value=empty_tensor)
                                materialized_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to create empty tensor for {full_name}: {e}")
                
                # Materialize buffers
                for name, buffer in module.named_buffers(recurse=False):
                    full_name = f"{prefix}.{name}" if prefix else name
                    if buffer.device.type == "meta":
                        # Find corresponding tensor in state_dict
                        if full_name in remapped_state_dict:
                            tensor = remapped_state_dict[full_name]
                            set_module_tensor_to_device(module, name, "cpu", value=tensor)
                            materialized_count += 1
                        else:
                            # If not in state_dict, create empty tensor on CPU with same shape/dtype
                            try:
                                shape = buffer.shape
                                dtype = buffer.dtype
                                empty_tensor = torch.zeros(shape, dtype=dtype, device="cpu")
                                set_module_tensor_to_device(module, name, "cpu", value=empty_tensor)
                                materialized_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to create empty buffer for {full_name}: {e}")
                
                # Recursively process child modules
                for child_name, child_module in module.named_children():
                    child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                    materialized_count += materialize_module(child_module, child_prefix)
                
                return materialized_count
            
            materialized_count = materialize_module(full)
            logger.info(f"Materialized {materialized_count} tensors from meta to CPU")
            
            # Verify all tensors are now on CPU (not meta) - check ALL parameters and buffers
            has_meta_after = False
            try:
                for param in full.parameters():
                    if param.device.type == "meta":
                        has_meta_after = True
                        logger.warning(f"Parameter still on meta after materialization: {list(param.shape)}")
                        break
                if not has_meta_after:
                    for buffer in full.buffers():
                        if buffer.device.type == "meta":
                            has_meta_after = True
                            logger.warning(f"Buffer still on meta after materialization: {list(buffer.shape)}")
                            break
            except StopIteration:
                pass
            
            if has_meta_after:
                logger.error("Some tensors are still on meta device after materialization! Cannot proceed.")
                raise RuntimeError("Failed to materialize all meta tensors to CPU. Some tensors remain on meta device.")
            else:
                logger.info("All tensors successfully materialized to CPU")
                # Now safe to move to CPU (though they should already be on CPU)
                full = full.cpu()
        else:
            # Normal case: no meta tensors, move to CPU
            full = full.cpu()
        
        if dtype is not None:
            full = full.to(dtype)
    
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

    # Pruning: 선택적 로딩을 사용한 경우 이미 작은 구조이므로 인덱스 조정 필요
    if selective_state_dict is not None:
        # 선택적 로딩 사용: 이미 작은 구조이므로 모든 레이어 유지 (pruning 불필요)
        # 하지만 role에 따라 불필요한 부분 제거는 여전히 필요
        if role == "stage0":
            # Stage0: 모든 레이어 유지 (이미 작은 구조)
            pass  # layers[0:end]는 이미 작은 구조에 포함됨
            if hasattr(full, "lm_head"):
                full.lm_head = None
            if hasattr(full, "model") and hasattr(full.model, "norm"):
                full.model.norm = None
        elif role == "segment":
            # Segment: 모든 레이어 유지 (이미 작은 구조, layers[0:end-start])
            pass  # layers[0:end-start]는 이미 작은 구조에 포함됨
            if hasattr(full, "lm_head"):
                full.lm_head = None
            if hasattr(full, "model") and hasattr(full.model, "norm"):
                full.model.norm = None
        elif role == "last":
            # Last: 모든 레이어 유지 (이미 작은 구조, layers[0:num_layers-start])
            pass  # layers[0:num_layers-start]는 이미 작은 구조에 포함됨
            # norm과 lm_head는 유지
        else:
            raise ValueError(f"Unknown role: {role}")
    else:
        # 전체 다운로드 사용: 기존 pruning 로직 사용
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
    logger.info(f"load_stage_model: role={role}, layers={num_layers}, start={start}, end={end}, quant_type={quant_type.name}")
    if num_layers == 0:
        raise ValueError(f"Pruned model has 0 layers for role={role} (start={start}, end={end}). Check --splits.")

    # Apply quantization if requested (must be done on CPU before moving to device)
    if quant_type != QuantType.NONE:
        logger.info(f"Quantizing model with {quant_type.name}...")
        # Ensure model is on CPU for quantization
        # Handle meta device tensors from device_map="auto"
        try:
            first_param = next(full.parameters())
            if first_param.device.type == "meta":
                logger.warning("Model has meta device tensors from device_map='auto', will skip meta tensors during quantization")
            elif first_param.device.type != "cpu":
                logger.warning("Moving model to CPU for quantization...")
                full = full.cpu()
        except StopIteration:
            logger.warning("Model has no parameters, skipping quantization")
        
        # Apply quantization (will skip meta device tensors)
        full = quantize_module(full, quant_type=quant_type)
        
        # Verify quantization was applied
        try:
            import bitsandbytes as bnb
            quantized_modules = []
            linear_modules = []
            for name, m in full.named_modules():
                if isinstance(m, torch.nn.Linear):
                    linear_modules.append(name)
                if isinstance(m, (bnb.nn.LinearNF4, bnb.nn.Linear8bitLt)):
                    quantized_modules.append(name)
            
            total_quantized = len(quantized_modules)
            total_linear = len(linear_modules)
            quant_ratio = (total_quantized / total_linear * 100) if total_linear > 0 else 0
            
            logger.info(
                f"Quantization verification: {total_quantized} quantized Linear layers "
                f"out of {total_linear} total Linear layers ({quant_ratio:.1f}%)"
            )
            if quantized_modules:
                logger.debug(
                    f"Quantized modules (first 10): {quantized_modules[:10]}"
                    f"{'...' if len(quantized_modules) > 10 else ''}"
                )
            if linear_modules:
                non_quantized = [name for name in linear_modules if name not in quantized_modules]
                logger.debug(
                    f"Non-quantized Linear modules (first 10): {non_quantized[:10]}"
                    f"{'...' if len(non_quantized) > 10 else ''}"
                )
        except ImportError:
            pass
        
        logger.info(f"Quantization with {quant_type.name} completed")

    # Move model to target device
    # For quantized models, bitsandbytes handles device placement automatically during forward pass
    # but we still need to move non-quantized parts (embeddings, norm, lm_head) to device
    if quant_type != QuantType.NONE:
        # For quantized models, move to device carefully
        # Quantized Linear layers will handle device placement during forward pass
        # But we need to move embeddings and other non-quantized components
        try:
            # Verify no meta tensors before moving
            has_meta = False
            try:
                for param in full.parameters():
                    if param.device.type == "meta":
                        has_meta = True
                        break
            except:
                pass
            
            if has_meta:
                logger.error("Model still has meta tensors! Cannot move to device. This should not happen after materialization.")
                raise RuntimeError("Model has meta tensors that were not materialized")
            else:
                # Move the entire model structure, bitsandbytes will handle quantized layers
                full = full.to(device)
        except Exception as e:
            logger.warning(f"Failed to move quantized model to {device}: {e}. "
                         "Quantized layers may handle device placement automatically during forward pass.")
    else:
        # Normal model: verify no meta tensors before moving
        try:
            has_meta = False
            try:
                for param in full.parameters():
                    if param.device.type == "meta":
                        has_meta = True
                        break
            except:
                pass
            
            if has_meta:
                logger.error("Model still has meta tensors! Cannot move to device. This should not happen after materialization.")
                raise RuntimeError("Model has meta tensors that were not materialized")
            else:
                full = full.to(device)
        except StopIteration:
            logger.warning("Model has no parameters")
        except Exception as e:
            logger.warning(f"Failed to move model to {device}: {e}")
    
    return full
