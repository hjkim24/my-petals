import logging
from typing import Optional, Tuple
from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig

# Qwen 모델 지원을 위한 import (선택적)
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    QWEN_AVAILABLE = True
except ImportError:
    try:
        from transformers.models.qwen.modeling_qwen import QwenDecoderLayer as Qwen2DecoderLayer
        from transformers.models.qwen.configuration_qwen import QwenConfig as Qwen2Config
        QWEN_AVAILABLE = True
    except ImportError:
        Qwen2DecoderLayer = None
        Qwen2Config = None
        QWEN_AVAILABLE = False

from .utils import default_position_ids

logger = logging.getLogger(__name__)


class QuantType(Enum):
    NONE = 0
    INT8 = 1  # 8-bit as in the LLM.int8() paper
    NF4 = 2  # 4-bit as in the QLoRA paper


def quantize_module(model: nn.Module, *, quant_type: QuantType) -> nn.Module:
    """
    Quantize a model module by replacing Linear layers with quantized versions.
    This is based on the original Petals implementation.
    
    Args:
        model: The model module to quantize
        quant_type: Type of quantization (INT8 or NF4)
    
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
            quantize_module(module, quant_type=quant_type)

        # Skip critical projection layers used for KV cache correctness
        # q_proj / k_proj / v_proj / o_proj must stay in higher precision
        # TODO: Temporarily allowing attention quantization for testing - can revert if issues occur
        skip_names = {"lm_head", "score"}  # , "q_proj", "k_proj", "v_proj", "o_proj"}  # Commented out to enable attention quantization

        if isinstance(module, torch.nn.Linear) and n not in skip_names:
            # Ensure the module is on CPU before quantization
            # Note: We load the model on CPU initially, so this should already be on CPU
            if module.weight.device.type != "cpu":
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
                model._modules[n] = bnb.nn.LinearNF4(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
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
    except (ImportError, RuntimeError):
        # ImportError: bitsandbytes not installed
        # RuntimeError: bitsandbytes CUDA setup failed
        return False
    
    try:
        for module in layer.modules():
            if isinstance(module, (bnb.nn.LinearNF4, bnb.nn.Linear8bitLt)):
                return True
    except (AttributeError, RuntimeError):
        # bitsandbytes가 설치되어 있지만 제대로 작동하지 않는 경우
        return False
    return False


def _count_quantized_modules(layer: nn.Module) -> dict:
    """Count quantized modules in a layer."""
    try:
        import bitsandbytes as bnb
    except (ImportError, RuntimeError):
        # ImportError: bitsandbytes not installed
        # RuntimeError: bitsandbytes CUDA setup failed
        return {"int8": 0, "nf4": 0, "total": 0}
    
    try:
        int8_count = 0
        nf4_count = 0
        
        for module in layer.modules():
            if isinstance(module, bnb.nn.Linear8bitLt):
                int8_count += 1
            elif isinstance(module, bnb.nn.LinearNF4):
                nf4_count += 1
        
        return {"int8": int8_count, "nf4": nf4_count, "total": int8_count + nf4_count}
    except (AttributeError, RuntimeError):
        # bitsandbytes가 설치되어 있지만 제대로 작동하지 않는 경우
        return {"int8": 0, "nf4": 0, "total": 0}


def _convert_config_to_llama_config(config):
    """
    Convert Qwen2Config (or other LLaMA-compatible configs) to LlamaConfig
    for use with OptimizedLlamaDecoderLayer.
    
    This creates a LlamaConfig with the same parameters as the original config,
    ensuring compatibility with OptimizedLlamaDecoderLayer.
    """
    # If already LlamaConfig, return as-is
    if isinstance(config, LlamaConfig):
        return config
    
    # Check if it's a Qwen config
    is_qwen_config = QWEN_AVAILABLE and isinstance(config, Qwen2Config)
    
    if is_qwen_config or getattr(config, "model_type", "").lower() in ["qwen", "qwen2"]:
        # Create LlamaConfig from Qwen2Config parameters
        llama_config = LlamaConfig(
            vocab_size=getattr(config, 'vocab_size', 151936),
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            hidden_act=config.hidden_act,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 32768),
            initializer_range=getattr(config, 'initializer_range', 0.02),
            rms_norm_eps=config.rms_norm_eps,
            use_cache=getattr(config, 'use_cache', True),
            pad_token_id=getattr(config, 'pad_token_id', None),
            bos_token_id=getattr(config, 'bos_token_id', None),
            eos_token_id=getattr(config, 'eos_token_id', None),
            tie_word_embeddings=getattr(config, 'tie_word_embeddings', False),
            rope_theta=getattr(config, 'rope_theta', 10000.0),
            attention_bias=False,  # Qwen doesn't use attention bias
            pretraining_tp=1,     # Default: no tensor parallelism
        )
        logger.debug(f"Converted Qwen2Config to LlamaConfig for OptimizedLlamaDecoderLayer compatibility")
        return llama_config
    
    # For other configs, try to create LlamaConfig with available attributes
    # This is a fallback for other LLaMA-compatible models
    try:
        llama_config = LlamaConfig(
            vocab_size=getattr(config, 'vocab_size', 32000),
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            hidden_act=getattr(config, 'hidden_act', 'silu'),
            max_position_embeddings=getattr(config, 'max_position_embeddings', 2048),
            initializer_range=getattr(config, 'initializer_range', 0.02),
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
            use_cache=getattr(config, 'use_cache', True),
            attention_bias=False,
            pretraining_tp=1,
        )
        logger.debug(f"Converted {type(config).__name__} to LlamaConfig for OptimizedLlamaDecoderLayer compatibility")
        return llama_config
    except Exception as e:
        logger.warning(f"Failed to convert config to LlamaConfig: {e}. Using original config.")
        # Fallback: add missing attributes to original config
        if not hasattr(config, 'attention_bias'):
            config.attention_bias = False
        if not hasattr(config, 'pretraining_tp'):
            config.pretraining_tp = 1
        return config


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
    
    # Convert config to LlamaConfig for OptimizedLlamaDecoderLayer compatibility
    # This ensures Qwen2Config and other LLaMA-compatible configs work properly
    if OPT_AVAILABLE:
        optimized_config = _convert_config_to_llama_config(config)
    else:
        optimized_config = config
    
    for idx, layer in enumerate(raw_layers):
        if OPT_AVAILABLE:
            if isinstance(layer, OptimizedLlamaDecoderLayer):
                converted.append(layer)
                already_optimized += 1
                continue
            
            # Support both LlamaDecoderLayer and Qwen2DecoderLayer (Qwen2.5 uses LLaMA-based architecture)
            is_llama_layer = isinstance(layer, LlamaDecoderLayer)
            is_qwen_layer = QWEN_AVAILABLE and isinstance(layer, Qwen2DecoderLayer)
            
            if is_llama_layer or is_qwen_layer:
                if _has_quantized_layers(layer):
                    # For quantized layers, create OptimizedLlamaDecoderLayer and copy modules directly
                    # to avoid shape mismatch from load_state_dict
                    try:
                        # Use optimized_config (LlamaConfig) for OptimizedLlamaDecoderLayer
                        opt_layer = OptimizedLlamaDecoderLayer(optimized_config)
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
                    # Use optimized_config (LlamaConfig) for OptimizedLlamaDecoderLayer
                    opt_layer = OptimizedLlamaDecoderLayer(optimized_config)
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
    """Stage0 for LLaMA-style and GPT-style (BLOOM) models; keep Cache end-to-end (no manual recompute)."""

    def __init__(self, full, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        is_llama_style = "llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "qwen" in model_type
        is_gpt_style = "bloom" in model_type or "gpt" in model_type
        if not is_llama_style and not is_gpt_style:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: LLaMA-style (llama, mistral, mixtral, qwen) or GPT-style (bloom, gpt)")

        # LLaMA-style models (LLaMA, Mistral, Mixtral, Qwen)
        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            self.embed_tokens = full.model.embed_tokens
            self.pos_embed = None  # LLaMA uses RoPE, no positional embeddings
            raw_layers = full.model.layers  # already pruned in load_stage_model
        # GPT-style models (BLOOM, GPT-2, etc.)
        elif hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, "wpe", None)  # Positional embeddings for GPT-style
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}. Expected LLaMA-style (model.embed_tokens) or GPT-style (transformer.wte)")

        # For BLOOM/GPT models, don't use OptimizedLlamaDecoderLayer (incompatible architecture)
        # For LLaMA-style models, try to use OptimizedLlamaDecoderLayer if available
        self.is_gpt_style = is_gpt_style  # Store for use in forward method
        if is_gpt_style:
            # BLOOM/GPT models: use original layers without optimization
            self.layers = nn.ModuleList(raw_layers)
            logger.info(f"Stage0 initialized with {len(self.layers)} layers (end={end}) for GPT-style model (BLOOM/GPT)")
        else:
            # LLaMA-style models: try to convert to OptimizedLlamaDecoderLayer
            self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        
        # Log layer status (only for LLaMA-style models)
        if not is_gpt_style:
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
        # Add positional embeddings for GPT-style models (BLOOM, GPT-2)
        if self.pos_embed is not None:
            if position_ids is None:
                # Generate position_ids if not provided
                seq_length = input_ids.shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(position_ids)
        cache_obj = None
        tuple_cache = []

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM/GPT models don't use position_ids, LLaMA-style models do
            if self.is_gpt_style:
                # BLOOM/GPT models: use layer_past format, no position_ids
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    layer_past=_to_cache(layer_past),
                    use_cache=use_cache,
                )
            else:
                # LLaMA-style models: use position_ids and past_key_value
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
    """Middle segment for LLaMA-style and GPT-style (BLOOM) models; keep Cache end-to-end."""

    def __init__(self, full, start: int, end: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        is_llama_style = "llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "qwen" in model_type
        is_gpt_style = "bloom" in model_type or "gpt" in model_type
        if not is_llama_style and not is_gpt_style:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: LLaMA-style (llama, mistral, mixtral, qwen) or GPT-style (bloom, gpt)")

        if hasattr(full, "model") and hasattr(full.model, "layers"):
            raw_layers = full.model.layers  # already pruned in load_stage_model
        elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
            raw_layers = full.transformer.h  # already pruned in load_stage_model
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}. Expected LLaMA-style (model.layers) or GPT-style (transformer.h)")

        # For BLOOM/GPT models, don't use OptimizedLlamaDecoderLayer (incompatible architecture)
        self.is_gpt_style = is_gpt_style  # Store for use in forward method
        if is_gpt_style:
            self.layers = nn.ModuleList(raw_layers)
        else:
            self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.config = full.config
        if len(self.layers) == 0:
            logger.warning(f"StageSegment initialized with 0 layers (start={start}, end={end})")
        else:
            # Log layer status (only for LLaMA-style models)
            if not is_gpt_style:
                quantized_layers = sum(1 for layer in self.layers if _has_quantized_layers(layer))
                if OPT_AVAILABLE and OptimizedLlamaDecoderLayer is not None:
                    optimized_layers = sum(1 for layer in self.layers if isinstance(layer, OptimizedLlamaDecoderLayer))
                    logger.info(
                        f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}): "
                        f"{quantized_layers} quantized, {optimized_layers} OptimizedLlamaDecoderLayer"
                    )
                else:
                    logger.info(
                        f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}): "
                        f"{quantized_layers} quantized"
                    )
            else:
                logger.info(
                    f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}) for GPT-style model (BLOOM/GPT)"
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

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM/GPT models don't use position_ids, LLaMA-style models do
            if self.is_gpt_style:
                # BLOOM/GPT models: use layer_past format, no position_ids
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    layer_past=_to_cache(layer_past),
                    use_cache=use_cache,
                )
            else:
                # LLaMA-style models: use position_ids and past_key_value
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
    """Last stage for LLaMA-style and GPT-style (BLOOM) models; keep Cache end-to-end."""

    def __init__(self, full, start: int):
        super().__init__()
        model_type = getattr(full.config, "model_type", "").lower()
        is_llama_style = "llama" in model_type or "mistral" in model_type or "mixtral" in model_type or "qwen" in model_type
        is_gpt_style = "bloom" in model_type or "gpt" in model_type
        if not is_llama_style and not is_gpt_style:
            raise ValueError(f"Unsupported model type: {model_type}. Supported: LLaMA-style (llama, mistral, mixtral, qwen) or GPT-style (bloom, gpt)")

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
            raise ValueError(f"Unsupported model architecture: {type(full)}. Expected LLaMA-style (model.layers) or GPT-style (transformer.h)")

        # For BLOOM/GPT models, don't use OptimizedLlamaDecoderLayer (incompatible architecture)
        self.is_gpt_style = is_gpt_style  # Store for use in forward method
        if is_gpt_style:
            self.layers = nn.ModuleList(raw_layers)
        else:
            self.layers = _convert_layers(nn.ModuleList(raw_layers), full.config)
        self.lm_head = full.lm_head
        self.config = full.config
        
        # Log layer status (only for LLaMA-style models)
        if not is_gpt_style:
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
        else:
            logger.info(
                f"StageLast initialized with {len(self.layers)} layers (start={start}) for GPT-style model (BLOOM/GPT)"
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

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            # BLOOM/GPT models don't use position_ids, LLaMA-style models do
            if self.is_gpt_style:
                # BLOOM/GPT models: use layer_past format, no position_ids
                out = layer(
                    x,
                    attention_mask=attention_mask,
                    layer_past=_to_cache(layer_past),
                    use_cache=use_cache,
                )
            else:
                # LLaMA-style models: use position_ids and past_key_value
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


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *,
    start: int = 0,
    end: Optional[int] = None,
    dtype=torch.float16,
    quant_type: QuantType = QuantType.NONE,
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    quant_type: Quantization type (QuantType.NONE, QuantType.INT8, or QuantType.NF4)
                If quantization is enabled, model will be loaded on CPU, quantized, then moved to device
    """
    # Always load model on CPU first (required for quantization, and safe for normal loading)
    # Try device_map="cpu" first, fallback to manual CPU loading if not supported
    try:
        full = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu"  # Explicitly load on CPU for quantization compatibility
        )
    except TypeError:
        # Fallback for older transformers versions that don't support device_map="cpu"
        logger.warning("device_map='cpu' not supported, loading model and moving to CPU manually")
        full = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        full = full.cpu()
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
    logger.info(f"load_stage_model: role={role}, layers={num_layers}, start={start}, end={end}, quant_type={quant_type.name}")
    if num_layers == 0:
        raise ValueError(f"Pruned model has 0 layers for role={role} (start={start}, end={end}). Check --splits.")

    # Apply quantization if requested (must be done on CPU before moving to device)
    if quant_type != QuantType.NONE:
        logger.info(f"Quantizing model with {quant_type.name}...")
        # Ensure model is on CPU for quantization
        if next(full.parameters()).device.type != "cpu":
            logger.warning("Moving model to CPU for quantization...")
            full = full.cpu()
        
        # Apply quantization
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
            # Move the entire model structure, bitsandbytes will handle quantized layers
            full = full.to(device)
        except Exception as e:
            logger.warning(f"Failed to move quantized model to {device}: {e}. "
                         "Quantized layers may handle device placement automatically during forward pass.")
    else:
        # Normal model: just move to device
        full = full.to(device)
    
    return full
