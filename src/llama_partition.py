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

    def __init__(self, full, start: int, end: int, 
                 gpu_device: Optional[torch.device] = None,
                 keep_layers_on_gpu: int = 0):
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
        
        # GPU device 설정
        self.gpu_device = gpu_device if gpu_device is not None else torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.keep_layers_on_gpu = keep_layers_on_gpu
        
        # 레이어들의 현재 device 추적
        self._layer_devices = {}  # {layer_idx: device}
        
        # 초기화: 모든 레이어를 CPU에 저장 (이미 CPU에 있으면 그대로 유지)
        for i, layer in enumerate(self.layers):
            # PyTorch 모듈의 device 확인: 첫 번째 파라미터의 device 사용
            try:
                layer_device = next(layer.parameters()).device
            except StopIteration:
                # 파라미터가 없는 경우 (거의 없지만) 기본값으로 CPU 가정
                layer_device = self.cpu_device
            
            if layer_device.type != "cpu":
                self.layers[i] = layer.to(self.cpu_device)
            self._layer_devices[i] = self.cpu_device
        
        if len(self.layers) == 0:
            logger.warning(f"StageSegment initialized with 0 layers (start={start}, end={end})")
        else:
            logger.info(f"StageSegment initialized with {len(self.layers)} layers (start={start}, end={end}), "
                       f"lazy GPU loading enabled (keep {keep_layers_on_gpu} layers on GPU)")
    
    def _move_layer_to_gpu(self, layer_idx: int):
        """레이어를 GPU로 이동"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        
        layer = self.layers[layer_idx]
        if self._layer_devices.get(layer_idx) != self.gpu_device:
            try:
                layer = layer.to(self.gpu_device, non_blocking=True)
                self._layer_devices[layer_idx] = self.gpu_device
            except RuntimeError as e:
                # GPU 메모리 부족 시 CPU에서 실행하도록 fallback
                logger.warning(f"Failed to move layer {layer_idx} to GPU: {e}, keeping on CPU")
                self._layer_devices[layer_idx] = self.cpu_device
    
    def _move_layer_to_cpu(self, layer_idx: int):
        """레이어를 CPU로 이동 (keep_layers_on_gpu 설정 고려)"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        
        # 최근 N개 레이어는 GPU에 유지
        if self.keep_layers_on_gpu > 0:
            total_layers = len(self.layers)
            if layer_idx >= total_layers - self.keep_layers_on_gpu:
                return  # GPU에 유지
        
        layer = self.layers[layer_idx]
        if self._layer_devices.get(layer_idx) != self.cpu_device:
            try:
                layer = layer.to(self.cpu_device, non_blocking=True)
                self._layer_devices[layer_idx] = self.cpu_device
            except Exception as e:
                logger.warning(f"Failed to move layer {layer_idx} to CPU: {e}")

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
            # 이전 레이어를 CPU로 이동 (필요한 경우)
            if i > 0:
                self._move_layer_to_cpu(i - 1)
            
            # 현재 레이어를 GPU로 이동
            self._move_layer_to_gpu(i)
            
            # 레이어의 현재 device 확인
            layer_device = self._layer_devices.get(i, self.cpu_device)
            
            # 입력을 레이어 device에 맞게 이동
            if layer_device == self.gpu_device:
                x = x.to(self.gpu_device, non_blocking=True)
            
            layer_past = None if past_key_values is None else past_key_values[i]
            
            # past_key_values도 레이어와 동일한 device로 이동
            if layer_past is not None and layer_device == self.gpu_device:
                if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                    key, value = layer_past
                    layer_past = (
                        key.to(self.gpu_device, non_blocking=True) if key is not None else None,
                        value.to(self.gpu_device, non_blocking=True) if value is not None else None
                    )
            
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
                # cache도 device 일치 확인 (필요한 경우)
                if present is not None and layer_device == self.gpu_device:
                    if isinstance(present, (tuple, list)) and len(present) == 2:
                        key, value = present
                        # 다음 레이어가 CPU면 CPU로 이동
                        if i + 1 < len(self.layers) and self._layer_devices.get(i + 1, self.cpu_device) == self.cpu_device:
                            present = (
                                key.to(self.cpu_device, non_blocking=True) if key is not None else None,
                                value.to(self.cpu_device, non_blocking=True) if value is not None else None
                            )
                tuple_cache.append(present)
        
        # 마지막 레이어 처리 후, 마지막 N개 레이어는 GPU에 유지
        # 나머지는 CPU로 이동
        for i in range(len(self.layers) - self.keep_layers_on_gpu):
            self._move_layer_to_cpu(i)

        if not use_cache:
            return x, None
        return x, tuple(tuple_cache)


class StageLast(nn.Module):
    """LLaMA-only last stage; keep Cache end-to-end."""

    def __init__(self, full, start: int,
                 gpu_device: Optional[torch.device] = None,
                 keep_layers_on_gpu: int = 0):
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
        
        # GPU device 설정
        self.gpu_device = gpu_device if gpu_device is not None else torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.keep_layers_on_gpu = keep_layers_on_gpu
        
        # 레이어들의 현재 device 추적
        self._layer_devices = {}
        
        # 모든 레이어를 CPU에 저장 (이미 CPU에 있으면 그대로 유지)
        for i, layer in enumerate(self.layers):
            # PyTorch 모듈의 device 확인: 첫 번째 파라미터의 device 사용
            try:
                layer_device = next(layer.parameters()).device
            except StopIteration:
                # 파라미터가 없는 경우 (거의 없지만) 기본값으로 CPU 가정
                layer_device = self.cpu_device
            
            if layer_device.type != "cpu":
                self.layers[i] = layer.to(self.cpu_device)
            self._layer_devices[i] = self.cpu_device
        
        # norm과 lm_head는 항상 GPU에 유지 (작고 자주 사용)
        if hasattr(self, 'norm') and self.norm is not None:
            self.norm = self.norm.to(self.gpu_device)
        if hasattr(self, 'lm_head') and self.lm_head is not None:
            self.lm_head = self.lm_head.to(self.gpu_device)
        
        logger.info(f"StageLast initialized with {len(self.layers)} layers (start={start}), "
                   f"lazy GPU loading enabled (keep {keep_layers_on_gpu} layers on GPU)")
    
    def _move_layer_to_gpu(self, layer_idx: int):
        """레이어를 GPU로 이동"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        
        if self._layer_devices.get(layer_idx) != self.gpu_device:
            try:
                self.layers[layer_idx] = self.layers[layer_idx].to(self.gpu_device, non_blocking=True)
                self._layer_devices[layer_idx] = self.gpu_device
            except RuntimeError as e:
                # GPU 메모리 부족 시 CPU에서 실행하도록 fallback
                logger.warning(f"Failed to move layer {layer_idx} to GPU: {e}, keeping on CPU")
                self._layer_devices[layer_idx] = self.cpu_device
    
    def _move_layer_to_cpu(self, layer_idx: int):
        """레이어를 CPU로 이동 (keep_layers_on_gpu 설정 고려)"""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        
        # 최근 N개 레이어는 GPU에 유지
        if self.keep_layers_on_gpu > 0:
            total_layers = len(self.layers)
            if layer_idx >= total_layers - self.keep_layers_on_gpu:
                return  # GPU에 유지
        
        if self._layer_devices.get(layer_idx) != self.cpu_device:
            try:
                self.layers[layer_idx] = self.layers[layer_idx].to(self.cpu_device, non_blocking=True)
                self._layer_devices[layer_idx] = self.cpu_device
            except Exception as e:
                logger.warning(f"Failed to move layer {layer_idx} to CPU: {e}")

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
            # 이전 레이어를 CPU로 이동 (필요한 경우)
            if i > 0:
                self._move_layer_to_cpu(i - 1)
            
            # 현재 레이어를 GPU로 이동
            self._move_layer_to_gpu(i)
            
            # 레이어의 현재 device 확인
            layer_device = self._layer_devices.get(i, self.cpu_device)
            
            # 입력을 레이어 device에 맞게 이동
            if layer_device == self.gpu_device:
                x = x.to(self.gpu_device, non_blocking=True)
            
            layer_past = None if past_key_values is None else past_key_values[i]
            
            # past_key_values도 레이어와 동일한 device로 이동
            if layer_past is not None and layer_device == self.gpu_device:
                if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                    key, value = layer_past
                    layer_past = (
                        key.to(self.gpu_device, non_blocking=True) if key is not None else None,
                        value.to(self.gpu_device, non_blocking=True) if value is not None else None
                    )
            
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
                # cache도 device 일치 확인 (필요한 경우)
                if present is not None and layer_device == self.gpu_device:
                    if isinstance(present, (tuple, list)) and len(present) == 2:
                        key, value = present
                        # 다음 레이어가 CPU면 CPU로 이동 (마지막 레이어는 norm/lm_head가 GPU이므로 GPU에 유지)
                        if i + 1 < len(self.layers) and self._layer_devices.get(i + 1, self.cpu_device) == self.cpu_device:
                            present = (
                                key.to(self.cpu_device, non_blocking=True) if key is not None else None,
                                value.to(self.cpu_device, non_blocking=True) if value is not None else None
                            )
                tuple_cache.append(present)
        
        # 마지막 레이어 처리 후, 마지막 N개 레이어는 GPU에 유지
        # 나머지는 CPU로 이동
        for i in range(len(self.layers) - self.keep_layers_on_gpu):
            self._move_layer_to_cpu(i)
        
        # norm과 lm_head는 항상 GPU에 있음
        x = x.to(self.gpu_device, non_blocking=True)
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
    use_cpu_offload: bool = False,
):
    """
    Load only the layers needed for a stage to reduce memory (LLaMA-only).
    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head
    use_cpu_offload: If True, load model to CPU instead of device (for lazy GPU loading)
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

    # Log resulting layer counts to catch empty segments early
    if hasattr(full, "model") and hasattr(full.model, "layers"):
        num_layers = len(full.model.layers)
    elif hasattr(full, "transformer") and hasattr(full.transformer, "h"):
        num_layers = len(full.transformer.h)
    else:
        num_layers = -1
    logger.info(f"load_stage_model: role={role}, layers={num_layers}, start={start}, end={end}")
    if num_layers == 0:
        raise ValueError(f"Pruned model has 0 layers for role={role} (start={start}, end={end}). Check --splits.")

    # CPU 오프로딩 모드면 CPU에 로드, 아니면 기존대로 device에 로드
    if use_cpu_offload:
        full = full.to(torch.device("cpu"))
        logger.info(f"load_stage_model: Model loaded to CPU (lazy GPU loading enabled)")
    else:
        full = full.to(device)
    
    return full
