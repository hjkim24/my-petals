import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
try:
    from transformers.cache_utils import Cache  # type: ignore
except Exception:
    Cache = None
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from petals.llama.block import OptimizedLlamaDecoderLayer
    # 임시로 OptimizedLlamaDecoderLayer 사용 비활성화 (hidden states 폭발 문제 해결을 위해)
    OPTIMIZED_LAYER_AVAILABLE = False
    logger.warning("OptimizedLlamaDecoderLayer is available but disabled due to hidden states explosion issue. Using default LlamaDecoderLayer.")
except ImportError as e:
    OPTIMIZED_LAYER_AVAILABLE = False
    logger.warning(f"OptimizedLlamaDecoderLayer not available ({e}), using default LlamaDecoderLayer")

from .utils import extract_kv_tuple, default_position_ids


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

        # Convert to OptimizedLlamaDecoderLayer if available
        if OPTIMIZED_LAYER_AVAILABLE:
            optimized_layers = []
            for i, layer in enumerate(raw_layers):
                if isinstance(layer, OptimizedLlamaDecoderLayer):
                    optimized_layers.append(layer)
                elif isinstance(layer, LlamaDecoderLayer):
                    # Create OptimizedLlamaDecoderLayer and copy weights
                    opt_layer = OptimizedLlamaDecoderLayer(full.config)
                    original_state = layer.state_dict()
                    missing_keys, unexpected_keys = opt_layer.load_state_dict(original_state, strict=False)
                    optimized_layers.append(opt_layer)
                    # 디버깅: 가중치 복사 확인 및 검증
                    if missing_keys or unexpected_keys:
                        logger.warning(f"Stage0 layer {i}: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
                        if missing_keys:
                            logger.warning(f"Stage0 layer {i} missing_keys: {list(missing_keys)[:5]}")  # 처음 5개만 출력
                        if unexpected_keys:
                            logger.warning(f"Stage0 layer {i} unexpected_keys: {list(unexpected_keys)[:5]}")
                    else:
                        logger.info(f"Stage0 layer {i}: Successfully converted to OptimizedLlamaDecoderLayer")
                    
                    # 가중치 복사 검증: 중요한 가중치가 제대로 복사되었는지 확인
                    opt_state = opt_layer.state_dict()
                    key_checks = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 
                                 'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
                    for key in key_checks:
                        if key in original_state and key in opt_state:
                            orig_tensor = original_state[key]
                            opt_tensor = opt_state[key]
                            # device와 dtype이 다를 수 있으므로 CPU, float32로 변환하여 비교
                            orig_tensor = orig_tensor.cpu().float()
                            opt_tensor = opt_tensor.cpu().float()
                            if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                                logger.error(f"Stage0 layer {i}: Weight mismatch for {key}!")
                        elif key in original_state:
                            logger.error(f"Stage0 layer {i}: Missing key {key} in optimized layer!")
                else:
                    optimized_layers.append(layer)
            self.layers = nn.ModuleList(optimized_layers)
        else:
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
                # 디버깅: 레이어 출력 구조 확인
                if i == 0:
                    logger.info(f"Stage0 layer {i} output: type={type(out)}, len={len(out)}, "
                               f"out[0] type={type(out[0])}, out[-1] type={type(out[-1]) if len(out) > 0 else 'N/A'}, "
                               f"out[-1] value={out[-1] if len(out) > 0 else 'N/A'}")
                
                kv_candidate = out[-1] if len(out) > 2 else (out[1] if len(out) > 1 else None)
                if Cache is not None and isinstance(kv_candidate, Cache):
                    cache_obj = kv_candidate
                    kv = kv_candidate[i] if hasattr(kv_candidate, "__getitem__") else None
                else:
                    kv = extract_kv_tuple(out, layer_idx=i)
                
                # 기본 LlamaDecoderLayer는 past_key_value를 반환하지 않을 수 있으므로
                # attention 모듈의 forward 결과에서 직접 가져오기 시도
                if kv is None:
                    # LlamaDecoderLayer는 self_attn을 호출하고 그 결과를 사용
                    # self_attn의 forward가 (attn_output, attn_weights, past_key_value)를 반환
                    # 하지만 이것은 내부적으로만 사용되므로 직접 접근 불가
                    # 대신 레이어 출력에서 찾기
                    if len(out) >= 2:
                        # out[1]이 None이면, 실제로는 다른 위치에 있을 수 있음
                        logger.warning(f"Stage0: layer {i} out[1] is None, checking all outputs")
                        for idx, item in enumerate(out):
                            logger.info(f"Stage0: layer {i} out[{idx}] = {item} (type={type(item)})")
                
                new_cache.append(kv)
                if kv is None:
                    logger.warning(f"Stage0: layer {i} returned no KV (kv_candidate type={type(kv_candidate)}, out_len={len(out)})")

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

        # Convert to OptimizedLlamaDecoderLayer if available
        if OPTIMIZED_LAYER_AVAILABLE:
            optimized_layers = []
            for i, layer in enumerate(raw_layers):
                if isinstance(layer, OptimizedLlamaDecoderLayer):
                    optimized_layers.append(layer)
                elif isinstance(layer, LlamaDecoderLayer):
                    # Create OptimizedLlamaDecoderLayer and copy weights
                    opt_layer = OptimizedLlamaDecoderLayer(full.config)
                    original_state = layer.state_dict()
                    missing_keys, unexpected_keys = opt_layer.load_state_dict(original_state, strict=False)
                    optimized_layers.append(opt_layer)
                    # 디버깅: 가중치 복사 확인
                    if missing_keys or unexpected_keys:
                        logger.warning(f"StageSegment layer {i}: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
                        if missing_keys:
                            logger.warning(f"StageSegment layer {i} missing_keys: {list(missing_keys)[:5]}")  # 처음 5개만 출력
                        if unexpected_keys:
                            logger.warning(f"StageSegment layer {i} unexpected_keys: {list(unexpected_keys)[:5]}")
                    else:
                        logger.info(f"StageSegment layer {i}: Successfully converted to OptimizedLlamaDecoderLayer")
                    
                    # 가중치 복사 검증
                    opt_state = opt_layer.state_dict()
                    key_checks = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 
                                 'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
                    for key in key_checks:
                        if key in original_state and key in opt_state:
                            orig_tensor = original_state[key]
                            opt_tensor = opt_state[key]
                            # device와 dtype이 다를 수 있으므로 CPU, float32로 변환하여 비교
                            orig_tensor = orig_tensor.cpu().float()
                            opt_tensor = opt_tensor.cpu().float()
                            if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                                logger.error(f"StageSegment layer {i}: Weight mismatch for {key}!")
                        elif key in original_state:
                            logger.error(f"StageSegment layer {i}: Missing key {key} in optimized layer!")
                else:
                    optimized_layers.append(layer)
            self.layers = nn.ModuleList(optimized_layers)
        else:
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
                kv_candidate = out[-1] if len(out) > 2 else (out[1] if len(out) > 1 else None)
                if Cache is not None and isinstance(kv_candidate, Cache):
                    cache_obj = kv_candidate
                    kv = kv_candidate[i] if hasattr(kv_candidate, "__getitem__") else None
                else:
                    kv = extract_kv_tuple(out, layer_idx=i)
                
                # 기본 LlamaDecoderLayer는 past_key_value를 반환하지 않을 수 있으므로
                # attention 모듈에서 직접 가져오기 시도
                if kv is None and hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'past_key_value'):
                    attn_past = layer.self_attn.past_key_value
                    if attn_past is not None:
                        kv = attn_past
                
                new_cache.append(kv)

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

        # Convert to OptimizedLlamaDecoderLayer if available
        if OPTIMIZED_LAYER_AVAILABLE:
            optimized_layers = []
            for i, layer in enumerate(raw_layers):
                if isinstance(layer, OptimizedLlamaDecoderLayer):
                    optimized_layers.append(layer)
                elif isinstance(layer, LlamaDecoderLayer):
                    # Create OptimizedLlamaDecoderLayer and copy weights
                    opt_layer = OptimizedLlamaDecoderLayer(full.config)
                    original_state = layer.state_dict()
                    missing_keys, unexpected_keys = opt_layer.load_state_dict(original_state, strict=False)
                    optimized_layers.append(opt_layer)
                    # 디버깅: 가중치 복사 확인
                    if missing_keys or unexpected_keys:
                        logger.warning(f"StageLast layer {i}: missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}")
                        if missing_keys:
                            logger.warning(f"StageLast layer {i} missing_keys: {list(missing_keys)[:5]}")  # 처음 5개만 출력
                        if unexpected_keys:
                            logger.warning(f"StageLast layer {i} unexpected_keys: {list(unexpected_keys)[:5]}")
                    else:
                        logger.info(f"StageLast layer {i}: Successfully converted to OptimizedLlamaDecoderLayer")
                    
                    # 가중치 복사 검증
                    opt_state = opt_layer.state_dict()
                    key_checks = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 
                                 'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
                    for key in key_checks:
                        if key in original_state and key in opt_state:
                            orig_tensor = original_state[key]
                            opt_tensor = opt_state[key]
                            # device와 dtype이 다를 수 있으므로 CPU, float32로 변환하여 비교
                            orig_tensor = orig_tensor.cpu().float()
                            opt_tensor = opt_tensor.cpu().float()
                            if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                                logger.error(f"StageLast layer {i}: Weight mismatch for {key}!")
                        elif key in original_state:
                            logger.error(f"StageLast layer {i}: Missing key {key} in optimized layer!")
                else:
                    optimized_layers.append(layer)
            self.layers = nn.ModuleList(optimized_layers)
        else:
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
                kv_candidate = out[-1] if len(out) > 2 else (out[1] if len(out) > 1 else None)
                if Cache is not None and isinstance(kv_candidate, Cache):
                    cache_obj = kv_candidate
                    kv = kv_candidate[i] if hasattr(kv_candidate, "__getitem__") else None
                else:
                    kv = extract_kv_tuple(out, layer_idx=i)
                
                # 기본 LlamaDecoderLayer는 past_key_value를 반환하지 않을 수 있으므로
                # attention 모듈에서 직접 가져오기 시도
                if kv is None and hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'past_key_value'):
                    attn_past = layer.self_attn.past_key_value
                    if attn_past is not None:
                        kv = attn_past
                
                new_cache.append(kv)

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
