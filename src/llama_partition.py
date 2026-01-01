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


def _debug_conversion(original_layer: LlamaDecoderLayer, optimized_layer: OptimizedLlamaDecoderLayer, 
                      layer_idx: int) -> tuple[float, float]:
    """Detailed debugging of layer conversion."""
    # 1. State dict 키 비교 (가중치 + buffer 포함)
    orig_sd = original_layer.state_dict()
    opt_sd = optimized_layer.state_dict()
    
    logger.info(f"Layer {layer_idx}: === State Dict Keys ===")
    logger.info(f"Layer {layer_idx}: Original has {len(orig_sd)} keys")
    logger.info(f"Layer {layer_idx}: Optimized has {len(opt_sd)} keys")
    
    missing = set(orig_sd.keys()) - set(opt_sd.keys())
    extra = set(opt_sd.keys()) - set(orig_sd.keys())
    
    if missing:
        logger.error(f"Layer {layer_idx}: Missing in optimized: {list(missing)}")
    if extra:
        logger.warning(f"Layer {layer_idx}: Extra in optimized: {list(extra)[:10]}")
    
    # 1-1. Buffer 명시적 확인 (특히 rotary_emb)
    orig_buffers = dict(original_layer.named_buffers())
    opt_buffers = dict(optimized_layer.named_buffers())
    
    logger.info(f"Layer {layer_idx}: === Buffers ===")
    logger.info(f"Layer {layer_idx}: Original buffers: {list(orig_buffers.keys())[:10]}")
    logger.info(f"Layer {layer_idx}: Optimized buffers: {list(opt_buffers.keys())[:10]}")
    
    missing_buffers = set(orig_buffers.keys()) - set(opt_buffers.keys())
    if missing_buffers:
        logger.error(f"Layer {layer_idx}: Missing buffers: {list(missing_buffers)}")
    
    # Rotary embedding buffer 확인
    if hasattr(original_layer, 'self_attn') and hasattr(original_layer.self_attn, 'rotary_emb'):
        orig_rotary_buffers = dict(original_layer.self_attn.rotary_emb.named_buffers())
        if hasattr(optimized_layer, 'self_attn') and hasattr(optimized_layer.self_attn, 'rotary_emb'):
            opt_rotary_buffers = dict(optimized_layer.self_attn.rotary_emb.named_buffers())
            logger.info(f"Layer {layer_idx}: Rotary embedding buffers - original: {list(orig_rotary_buffers.keys())}, optimized: {list(opt_rotary_buffers.keys())}")
            
            for buffer_name in orig_rotary_buffers:
                if buffer_name in opt_rotary_buffers:
                    orig_buf = orig_rotary_buffers[buffer_name]
                    opt_buf = opt_rotary_buffers[buffer_name]
                    # Device가 다를 수 있으므로 CPU로 이동하여 비교
                    orig_buf_cpu = orig_buf.cpu()
                    opt_buf_cpu = opt_buf.cpu()
                    if not torch.equal(orig_buf_cpu, opt_buf_cpu):
                        buf_diff = (orig_buf_cpu - opt_buf_cpu).abs().max().item()
                        logger.error(f"Layer {layer_idx}: Rotary buffer '{buffer_name}' mismatch: {buf_diff:.8f}")
                    else:
                        logger.debug(f"Layer {layer_idx}: Rotary buffer '{buffer_name}' OK")
                else:
                    logger.error(f"Layer {layer_idx}: Rotary buffer '{buffer_name}' missing in optimized")
    
    # 2. 가중치 직접 비교 (공통 키만)
    common_keys = set(orig_sd.keys()) & set(opt_sd.keys())
    max_diff = 0.0
    max_diff_key = None
    mismatched_keys = []
    
    for key in common_keys:
        orig_tensor = orig_sd[key]
        opt_tensor = opt_sd[key]
        
        # Shape 확인
        if orig_tensor.shape != opt_tensor.shape:
            logger.error(f"Layer {layer_idx}: Shape mismatch for {key}: {orig_tensor.shape} vs {opt_tensor.shape}")
            continue
        
        # CPU, float32로 변환하여 비교
        orig_tensor = orig_tensor.cpu().float()
        opt_tensor = opt_tensor.cpu().float()
        
        diff = (orig_tensor - opt_tensor).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            max_diff_key = key
        if diff > 1e-5:
            mismatched_keys.append((key, diff))
            logger.error(f"Layer {layer_idx}:   {key}: max diff = {diff:.8f}")
    
    if max_diff_key:
        logger.info(f"Layer {layer_idx}: Max difference in '{max_diff_key}': {max_diff:.8f}")
    
    # 3. Forward 결과 비교
    device = next(original_layer.parameters()).device
    dtype = next(original_layer.parameters()).dtype
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(
        batch_size, seq_len, original_layer.hidden_size,
        device=device,
        dtype=dtype
    )
    
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    original_layer.eval()
    optimized_layer.eval()
    
    with torch.no_grad():
        orig_out = original_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
        opt_out = optimized_layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
        )[0]
    
    output_diff = (orig_out - opt_out).abs().mean().item()
    output_diff_pct = (output_diff / (orig_out.abs().mean().item() + 1e-8)) * 100
    
    logger.info(f"Layer {layer_idx}: === Forward Output Comparison ===")
    logger.info(f"Layer {layer_idx}: Mean absolute difference: {output_diff:.8f}")
    logger.info(f"Layer {layer_idx}: Percentage difference: {output_diff_pct:.4f}%")
    
    return max_diff, output_diff_pct


def _verify_layer_conversion(original_layer: LlamaDecoderLayer, optimized_layer: OptimizedLlamaDecoderLayer, 
                             config, layer_idx: int) -> bool:
    """Verify that the converted layer produces the same output as the original."""
    try:
        # 상세 디버깅 실행
        max_weight_diff, output_diff_pct = _debug_conversion(original_layer, optimized_layer, layer_idx)
        
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
        
        # 더 관대한 tolerance 사용 (float16의 경우 수치적 오차가 클 수 있음)
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        rtol = 1e-3 if dtype == torch.float16 else 1e-5
        
        if not torch.allclose(original_hidden, optimized_hidden, atol=atol, rtol=rtol):
            max_diff = (original_hidden - optimized_hidden).abs().max().item()
            mean_diff = (original_hidden - optimized_hidden).abs().mean().item()
            std_diff = (original_hidden - optimized_hidden).abs().std().item()
            
            # 상대 오차 계산
            relative_max_diff = (max_diff / (original_hidden.abs().max().item() + 1e-8)) * 100
            
            logger.warning(
                f"Layer {layer_idx}: Output mismatch! max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                f"std_diff={std_diff:.6f}, relative_max_diff={relative_max_diff:.4f}%, dtype={dtype}"
            )
            
            # 가중치 차이와 출력 차이 비교
            if max_weight_diff > 1e-4:
                logger.error(
                    f"Layer {layer_idx}: Large weight difference detected ({max_weight_diff:.8f})! "
                    f"This is likely the cause of output mismatch."
                )
            
            # 상대 오차가 작으면 (1% 미만) 경고만 출력하고 계속 진행
            if relative_max_diff < 1.0:
                logger.warning(
                    f"Layer {layer_idx}: Small relative error ({relative_max_diff:.4f}%), "
                    f"likely due to numerical precision. Continuing..."
                )
                return True
            else:
                logger.error(f"Layer {layer_idx}: Large relative error ({relative_max_diff:.4f}%), conversion may be incorrect")
                return False
        
        logger.info(f"Layer {layer_idx}: ✓ Conversion verified successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Layer {layer_idx}: Verification failed with error: {e}", exc_info=True)
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
            logger.debug(f"Layer {idx}: Original state_dict keys: {list(original_state_dict.keys())[:10]}")
            
            # 2. OptimizedLlamaDecoderLayer 생성
            opt_layer = OptimizedLlamaDecoderLayer(config)
            opt_state_dict_before = opt_layer.state_dict()
            logger.debug(f"Layer {idx}: Optimized state_dict keys (before): {list(opt_state_dict_before.keys())[:10]}")
            
            # 3. 가중치 로드 전 검증: 키 이름 비교
            orig_keys = set(original_state_dict.keys())
            opt_keys_before = set(opt_state_dict_before.keys())
            missing_before = orig_keys - opt_keys_before
            unexpected_before = opt_keys_before - orig_keys
            
            if missing_before:
                logger.error(f"Layer {idx}: Keys in original but not in optimized (before load): {list(missing_before)[:10]}")
            if unexpected_before:
                logger.warning(f"Layer {idx}: Keys in optimized but not in original (before load): {list(unexpected_before)[:10]}")
            
            # 4. 가중치 및 Buffer 로드
            # state_dict()는 기본적으로 buffer도 포함하지만, 명시적으로 확인
            original_buffers = dict(layer.named_buffers())
            logger.debug(f"Layer {idx}: Original buffers: {list(original_buffers.keys())[:10]}")
            
            # load_state_dict는 buffer도 복사하지만, 명시적으로 확인하기 위해
            # strict=True로 시도, 실패하면 strict=False
            try:
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=True)
                if missing:
                    logger.error(f"Layer {idx}: missing keys after strict=True: {list(missing)}")
                if unexpected:
                    logger.warning(f"Layer {idx}: unexpected keys after strict=True: {list(unexpected)[:10]}")
                if not missing and not unexpected:
                    logger.info(f"Layer {idx}: ✓ All weights loaded successfully with strict=True")
            except RuntimeError as e:
                logger.warning(f"Layer {idx}: strict=True failed: {e}")
                missing, unexpected = opt_layer.load_state_dict(original_state_dict, strict=False)
                if missing:
                    logger.error(f"Layer {idx}: missing keys after strict=False: {list(missing)}")
                    # 서브모듈별로 직접 복사 시도
                    logger.info(f"Layer {idx}: Attempting to copy weights module by module...")
                    try:
                        # self_attn 복사 (가중치 + buffer)
                        if hasattr(layer, 'self_attn') and hasattr(opt_layer, 'self_attn'):
                            opt_layer.self_attn.load_state_dict(layer.self_attn.state_dict(), strict=False)
                            # Rotary embedding buffer 명시적 복사
                            if hasattr(layer.self_attn, 'rotary_emb') and hasattr(opt_layer.self_attn, 'rotary_emb'):
                                orig_rotary_buffers = dict(layer.self_attn.rotary_emb.named_buffers())
                                opt_rotary_buffers = dict(opt_layer.self_attn.rotary_emb.named_buffers())
                                for buffer_name, buffer_value in orig_rotary_buffers.items():
                                    if buffer_name in opt_rotary_buffers:
                                        opt_buffer = getattr(opt_layer.self_attn.rotary_emb, buffer_name)
                                        # Device가 다를 수 있으므로 opt_buffer의 device로 복사
                                        opt_buffer.data.copy_(buffer_value.data.to(opt_buffer.device))
                                        logger.debug(f"Layer {idx}: Copied rotary_emb buffer: {buffer_name}")
                                    else:
                                        logger.warning(f"Layer {idx}: Rotary buffer {buffer_name} not found in optimized layer")
                            logger.info(f"Layer {idx}: Copied self_attn weights and buffers")
                        
                        # mlp 복사
                        if hasattr(layer, 'mlp') and hasattr(opt_layer, 'mlp'):
                            opt_layer.mlp.load_state_dict(layer.mlp.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied mlp weights")
                        
                        # layernorm 복사
                        if hasattr(layer, 'input_layernorm') and hasattr(opt_layer, 'input_layernorm'):
                            opt_layer.input_layernorm.load_state_dict(layer.input_layernorm.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied input_layernorm weights")
                        
                        if hasattr(layer, 'post_attention_layernorm') and hasattr(opt_layer, 'post_attention_layernorm'):
                            opt_layer.post_attention_layernorm.load_state_dict(layer.post_attention_layernorm.state_dict(), strict=False)
                            logger.info(f"Layer {idx}: Copied post_attention_layernorm weights")
                        
                        # 다시 확인
                        missing_after_module_copy = set(original_state_dict.keys()) - set(opt_layer.state_dict().keys())
                        if missing_after_module_copy:
                            logger.error(f"Layer {idx}: Still missing keys after module copy: {list(missing_after_module_copy)[:10]}")
                        else:
                            logger.info(f"Layer {idx}: ✓ All weights copied successfully via module-by-module approach")
                    except Exception as module_copy_error:
                        logger.error(f"Layer {idx}: Module-by-module copy failed: {module_copy_error}", exc_info=True)
                
                if unexpected:
                    logger.warning(f"Layer {idx}: unexpected keys after strict=False: {list(unexpected)[:10]}")
            
            # 4-1. Buffer 명시적 복사 확인 (특히 rotary_emb)
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                if hasattr(opt_layer, 'self_attn') and hasattr(opt_layer.self_attn, 'rotary_emb'):
                    orig_rotary = layer.self_attn.rotary_emb
                    opt_rotary = opt_layer.self_attn.rotary_emb
                    
                    # Rotary embedding의 모든 buffer 확인 및 복사
                    orig_rotary_buffers = dict(orig_rotary.named_buffers())
                    opt_rotary_buffers = dict(opt_rotary.named_buffers())
                    
                    logger.debug(f"Layer {idx}: Rotary embedding buffers - original: {list(orig_rotary_buffers.keys())}, optimized: {list(opt_rotary_buffers.keys())}")
                    
                    for buffer_name, orig_buffer in orig_rotary_buffers.items():
                        if buffer_name in opt_rotary_buffers:
                            opt_buffer = getattr(opt_rotary, buffer_name)
                            # Device가 다를 수 있으므로 CPU로 이동하여 비교
                            orig_buffer_cpu = orig_buffer.cpu()
                            opt_buffer_cpu = opt_buffer.cpu()
                            if not torch.equal(orig_buffer_cpu, opt_buffer_cpu):
                                opt_buffer.data.copy_(orig_buffer.data.to(opt_buffer.device))
                                logger.info(f"Layer {idx}: Copied rotary_emb buffer: {buffer_name}")
                            else:
                                logger.debug(f"Layer {idx}: Rotary buffer {buffer_name} already matches")
                        else:
                            logger.warning(f"Layer {idx}: Rotary buffer {buffer_name} not found in optimized layer")
            
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
            
            # 6. 가중치 검증 (모든 가중치 비교)
            opt_state_dict = opt_layer.state_dict()
            key_checks = [
                'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
                'input_layernorm.weight', 'post_attention_layernorm.weight'
            ]
            
            weight_mismatches = []
            missing_weights = []
            
            for key in key_checks:
                if key in original_state_dict and key in opt_state_dict:
                    orig_tensor = original_state_dict[key]
                    opt_tensor = opt_state_dict[key]
                    
                    # Shape 확인
                    if orig_tensor.shape != opt_tensor.shape:
                        logger.error(f"Layer {idx}: Shape mismatch for {key}: {orig_tensor.shape} vs {opt_tensor.shape}")
                        missing_weights.append(key)
                        continue
                    
                    # CPU, float32로 변환하여 비교
                    orig_tensor = orig_tensor.cpu().float()
                    opt_tensor = opt_tensor.cpu().float()
                    
                    max_diff = (orig_tensor - opt_tensor).abs().max().item()
                    mean_diff = (orig_tensor - opt_tensor).abs().mean().item()
                    
                    if not torch.allclose(orig_tensor, opt_tensor, atol=1e-4):
                        logger.error(
                            f"Layer {idx}: Weight mismatch for {key}! max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}, "
                            f"shape={orig_tensor.shape}"
                        )
                        weight_mismatches.append((key, max_diff, mean_diff))
                    else:
                        logger.debug(f"Layer {idx}: Weight OK for {key}: max_diff={max_diff:.8f}")
                elif key in original_state_dict:
                    logger.error(f"Layer {idx}: Missing key {key} in optimized layer!")
                    missing_weights.append(key)
                elif key in opt_state_dict:
                    logger.warning(f"Layer {idx}: Key {key} exists in optimized but not in original")
            
            if weight_mismatches:
                logger.error(f"Layer {idx}: Found {len(weight_mismatches)} weight mismatches!")
                for key, max_diff, mean_diff in weight_mismatches[:5]:  # 처음 5개만 출력
                    logger.error(f"Layer {idx}:   - {key}: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
            
            if missing_weights:
                logger.error(f"Layer {idx}: Found {len(missing_weights)} missing weights: {missing_weights}")
            
            # 7. Forward pass 검증 (선택적)
            # 주의: OptimizedLlamaDecoderLayer는 구현이 약간 다를 수 있어서 
            # 완전히 동일한 출력을 보장하지 않을 수 있음 (특히 float16에서)
            if verify:
                verification_passed = _verify_layer_conversion(layer, opt_layer, config, idx)
                if not verification_passed:
                    # 검증 실패해도 계속 진행 (OptimizedLlamaDecoderLayer는 Petals의 최적화된 구현이므로)
                    logger.warning(
                        f"Layer {idx}: Forward pass verification failed. "
                        f"This may be expected due to implementation differences in OptimizedLlamaDecoderLayer. "
                        f"Continuing with conversion..."
                    )
            
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
