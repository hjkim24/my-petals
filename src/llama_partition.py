import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .utils import extract_kv_tuple, default_position_ids

logger = logging.getLogger(__name__)

# Prefer optimized block implementation if available
try:
    from petals.llama.block import OptimizedLlamaDecoderLayer

    OPTIMIZED_LAYER_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    OptimizedLlamaDecoderLayer = None
    OPTIMIZED_LAYER_AVAILABLE = False
    logger.warning(f"OptimizedLlamaDecoderLayer not available ({e}), using vanilla LlamaDecoderLayer.")


def _verify_layer_conversion(original_layer: LlamaDecoderLayer, optimized_layer: OptimizedLlamaDecoderLayer, 
                             config, layer_idx: int) -> bool:
    """Verify that the converted layer produces the same output as the original."""
    try:
        # 입력 텐서 생성 (원본 레이어의 device와 dtype 사용)
        device = original_layer.input_layernorm.weight.device
        dtype = original_layer.input_layernorm.weight.dtype
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(
            batch_size, seq_len, config.hidden_size,
            device=device,
            dtype=dtype
        )
        
        # position_ids 생성
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        with torch.no_grad():
            # 원본 레이어 출력
            original_output = original_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
            
            # 최적화 버전 출력
            optimized_output = optimized_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=False,
            )
        
        # 결과 비교 (hidden_states만 비교)
        original_hidden = original_output[0]
        optimized_hidden = optimized_output[0]
        
        if not torch.allclose(original_hidden, optimized_hidden, atol=1e-5, rtol=1e-5):
            max_diff = (original_hidden - optimized_hidden).abs().max().item()
            mean_diff = (original_hidden - optimized_hidden).abs().mean().item()
            logger.error(
                f"Layer {layer_idx}: Output mismatch! max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )
            return False
        
        logger.info(f"Layer {layer_idx}: ✓ Conversion verified successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Layer {layer_idx}: Verification failed with error: {e}")
        return False


def _convert_layers(raw_layers: nn.ModuleList, config, device=None, dtype=None, verify: bool = True) -> nn.ModuleList:
    """Convert HF LlamaDecoderLayer to optimized version if available.
    
    Args:
        raw_layers: Original layers to convert
        config: Model configuration
        device: Target device (if None, uses original layer's device)
        dtype: Target dtype (if None, uses original layer's dtype)
        verify: Whether to verify conversion by running forward pass
    """
    if not OPTIMIZED_LAYER_AVAILABLE:
        return nn.ModuleList(raw_layers)

    optimized = []
    for idx, layer in enumerate(raw_layers):
        if isinstance(layer, OptimizedLlamaDecoderLayer):
            optimized.append(layer)
        elif isinstance(layer, LlamaDecoderLayer):
            # 1. 원본 레이어에서 state_dict 추출
            original_state_dict = layer.state_dict()
            
            # 2. OptimizedLlamaDecoderLayer 생성
            opt_layer = OptimizedLlamaDecoderLayer(config)
            
            # 3. 가중치 로드 (strict=True로 시도, 실패하면 strict=False)
            try:
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=True)
                if missing:
                    logger.warning(f"Layer {idx}: missing keys: {list(missing)[:5]}")
                if unexpected:
                    logger.warning(f"Layer {idx}: unexpected keys: {list(unexpected)[:5]}")
            except RuntimeError as e:
                logger.warning(f"Layer {idx}: strict=True failed, trying strict=False: {e}")
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(
                        f"Layer {idx}: optimized load had missing={len(missing)}, unexpected={len(unexpected)}"
                    )
            
            # 4. 디바이스 이동 (원본 레이어와 동일한 device로)
            if device is not None:
                opt_layer = opt_layer.to(device)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                # 원본 레이어의 device 확인
                original_device = layer.self_attn.q_proj.weight.device
                opt_layer = opt_layer.to(original_device)
            
            # 5. dtype 변환 (원본 레이어와 동일한 dtype으로)
            if dtype is not None:
                opt_layer = opt_layer.to(dtype)
            elif hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                # 원본 레이어의 dtype 확인
                original_dtype = layer.self_attn.q_proj.weight.dtype
                opt_layer = opt_layer.to(original_dtype)
            
            # 6. 가중치 검증 (선택적)
            opt_state_dict = opt_layer.state_dict()
            key_checks = [
                'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'
            ]
            for key in key_checks:
                if key in original_state_dict and key in opt_state_dict:
                    orig_tensor = original_state_dict[key]
                    opt_tensor = opt_state_dict[key]
                    # CPU, float32로 변환하여 비교
                    orig_tensor = orig_tensor.cpu().float()
                    opt_tensor = opt_tensor.cpu().float()
                    if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                        logger.error(f"Layer {idx}: Weight mismatch for {key}!")
                elif key in original_state_dict:
                    logger.error(f"Layer {idx}: Missing key {key} in optimized layer!")
            
            # 7. Forward pass 검증 (선택적)
            if verify:
                verification_passed = _verify_layer_conversion(layer, opt_layer, config, idx)
                if not verification_passed:
                    logger.warning(f"Layer {idx}: Forward pass verification failed, but continuing...")
            
            optimized.append(opt_layer)
        else:
            optimized.append(layer)
    
    return nn.ModuleList(optimized)


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

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
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

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
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
                if kv is None:
                    logger.warning(f"Stage0: layer {i} did not return KV cache")
                new_cache.append(kv)

        if not use_cache:
            return x, None
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

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
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

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
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
                if kv is None:
                    logger.warning(f"StageSegment: layer {i} did not return KV cache")
                new_cache.append(kv)

        if not use_cache:
            return x, None
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

        # device와 dtype 추출
        device = None
        dtype = None
        if len(raw_layers) > 0:
            first_layer = raw_layers[0]
            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
                device = first_layer.self_attn.q_proj.weight.device
                dtype = first_layer.self_attn.q_proj.weight.dtype

        self.layers = _convert_layers(raw_layers, full.config, device=device, dtype=dtype)
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

        for i, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[i]
            layer_pos = position_ids if position_ids is not None else default_position_ids(
                layer_past, x.shape[1], x.device
            )
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
                if kv is None:
                    logger.warning(f"StageLast: layer {i} did not return KV cache")
                new_cache.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        if not use_cache:
            return logits, None
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

    full = full.to(device)
    return full
