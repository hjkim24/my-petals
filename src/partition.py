import torch
import torch.nn as nn
import inspect
from transformers import AutoModelForCausalLM


def _get_past_from_output(out):
    """Return past cache (if any) from a transformer layer output."""
    # 방법 1: NamedTuple 또는 객체 속성 확인
    if hasattr(out, "past_key_values") and out.past_key_values is not None:
        return out.past_key_values
    if hasattr(out, "next_cache") and out.next_cache is not None:
        return out.next_cache
    if hasattr(out, "present") and out.present is not None:
        # GPT-2 style: present is (key, value) tuple
        return out.present
    
    # 방법 2: 튜플/리스트인 경우
    if isinstance(out, (tuple, list)):
        # GPT-2: (hidden_states, present) where present is (key, value) tuple
        # LLaMA/Qwen: (hidden_states, past_key_value) where past_key_value is (key, value) tuple
        if len(out) >= 2:
            # out[1]이 KV cache일 수 있음
            cache = out[1]
            if cache is not None:
                # GPT-2의 경우 present는 (key, value) 튜플
                # LLaMA의 경우 past_key_value는 (key, value) 튜플
                if isinstance(cache, (tuple, list)) and len(cache) == 2:
                    # (key, value) 형태인지 확인
                    return cache
                # 이미 올바른 형태일 수도 있음
                return cache
        
        # 일부 모델은 더 많은 요소를 반환할 수 있음 (예: attention weights)
        # 마지막 요소가 cache일 수도 있음
        if len(out) > 2:
            cache = out[-1]
            if cache is not None:
                # 마지막 요소가 (key, value) 튜플인지 확인
                if isinstance(cache, (tuple, list)) and len(cache) == 2:
                    return cache
                # Tensor가 아닌 경우 (예: tuple of (key, value))
                if not isinstance(cache, torch.Tensor):
                    return cache
    
    # 방법 3: BaseModelOutput 형식 확인
    if hasattr(out, "__class__"):
        class_name = out.__class__.__name__
        if "Output" in class_name:
            # transformers의 BaseModelOutput 형식
            if hasattr(out, "past_key_values"):
                return out.past_key_values
            if hasattr(out, "present"):
                return out.present
    
    return None


class GPT2BlockWrapper(nn.Module):
    """Wrapper for GPT2Block to ensure present is always returned when use_cache=True."""
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x, **kwargs):
        use_cache = kwargs.get('use_cache', False)
        layer_past = kwargs.get('layer_past', None)
        out = self.block(x, **kwargs)
        
        # GPT-2Block이 (hidden_states,)만 반환하는 경우 처리
        if use_cache and isinstance(out, (tuple, list)) and len(out) == 1:
            # attention 모듈을 직접 호출하여 present를 얻기
            if hasattr(self.block, 'attn') and hasattr(self.block, 'ln_1'):
                attn_input = self.block.ln_1(x)
                attn_kwargs = {}
                if 'attention_mask' in kwargs:
                    attn_kwargs['attention_mask'] = kwargs['attention_mask']
                if 'position_ids' in kwargs:
                    attn_kwargs['position_ids'] = kwargs['position_ids']
                attn_kwargs['layer_past'] = layer_past
                attn_kwargs['use_cache'] = use_cache
                
                try:
                    attn_out = self.block.attn(attn_input, **attn_kwargs)
                    if isinstance(attn_out, (tuple, list)) and len(attn_out) >= 2:
                        present = attn_out[1]
                        if present is not None:
                            return (out[0], present)
                    
                    # present가 None인 경우, GPT-2Attention의 내부 상태 확인
                    # GPT-2Attention은 forward에서 key와 value를 계산하고 저장할 수 있음
                    if hasattr(self.block.attn, 'key') and hasattr(self.block.attn, 'value'):
                        # attention 모듈의 key와 value를 직접 확인
                        # 하지만 이는 forward pass 후에만 사용 가능할 수 있음
                        pass
                    
                    # 마지막 방법: GPT-2Attention의 forward 메서드를 다시 호출하되,
                    # 이번에는 내부적으로 계산된 key와 value를 사용하여 present를 만들기
                    # 하지만 이는 복잡하므로, 일단 경고만 출력
                    import warnings
                    warnings.warn(
                        f"GPT2BlockWrapper: Could not extract present from attention module. "
                        f"attn_out type: {type(attn_out)}, attn_out len: {len(attn_out) if isinstance(attn_out, (tuple, list)) else 'N/A'}, "
                        f"attn_out[1]: {attn_out[1] if isinstance(attn_out, (tuple, list)) and len(attn_out) > 1 else 'N/A'}"
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(f"GPT2BlockWrapper: Failed to get present from attention module: {e}")
        
        return out


class Stage0(nn.Module):
    def __init__(self, full, end: int):
        super().__init__()
        self.is_gpt2 = (
            getattr(full.config, "model_type", "") == "gpt2"
            or (hasattr(full, "transformer") and hasattr(full.transformer, "h"))
        )
        if self.is_gpt2:
            # GPT-2: use the pruned GPT2Model so it builds attention masks / present internally
            self.model = full.transformer
            self.model.h = nn.ModuleList(self.model.h[:end])
            self.config = full.config
            # expose embeddings for dtype checks in the caller
            self.embed_tokens = self.model.wte
            self.pos_embed = getattr(self.model, "wpe", None)
            return
        if hasattr(full, 'model') and hasattr(full.model, 'embed_tokens'):
            self.embed_tokens = full.model.embed_tokens
            raw_layers = full.model.layers[:end]
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'wte'):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, 'wpe', None)
            raw_layers = full.transformer.h[:end]
        elif hasattr(full, 'model') and hasattr(full.model, 'embed_in'):
            self.embed_tokens = full.model.embed_in
            raw_layers = full.model.layers[:end]
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")
        self.config = full.config
        # GPT-2Block인 경우 래퍼 적용
        self.layers = nn.ModuleList()
        for layer in raw_layers:
            if type(layer).__name__ == 'GPT2Block':
                self.layers.append(GPT2BlockWrapper(layer))
            else:
                self.layers.append(layer)
        
        sig_params = [inspect.signature(layer.forward).parameters for layer in raw_layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]
        self._supports_position_embeddings = ['position_embeddings' in p for p in sig_params]
        self._supports_cache_position = ['cache_position' in p for p in sig_params]

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None, use_cache=True):
        if self.is_gpt2:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            hidden = outputs[0]
            pkv = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            return hidden, pkv
        x = self.embed_tokens(input_ids)
        # Position Embedding
        if hasattr(self, 'pos_embed') and self.pos_embed is not None and position_ids is not None:
            x = x + self.pos_embed(position_ids)
        # KV 캐시 리스트 초기화
        new_past = []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]

            # kwargs 세팅
            kwargs = dict(attention_mask=attention_mask, use_cache=use_cache)
            if self._supports_pos_ids[i]:
                kwargs['position_ids'] = position_ids
            if self._supports_past_key_value[i]:
                kwargs['past_key_value'] = pkv
            elif self._supports_layer_past[i]:
                # GPT-2 style: layer_past는 (key, value) 튜플 또는 None
                # None이어도 명시적으로 전달해야 함 (일부 구현에서 필요)
                kwargs['layer_past'] = pkv
            # GPT-2의 경우 layer_past가 없어도 use_cache=True면 present를 반환해야 함
            # 하지만 현재 출력이 len=1이므로, 직접 확인 필요
            # 실제로 레이어 통과하는 부분
            # GPT-2의 경우, use_cache=True일 때 (hidden_states, present)를 반환해야 함
            # 하지만 일부 버전에서는 (hidden_states,)만 반환할 수 있음
            # 이 경우 attention 모듈을 직접 호출하여 present를 얻어야 함
            try:
                out = layer(x, **kwargs)
            except TypeError as e:
                # 일부 모델이 position_embeddings/cache_position을 지원하지 않는 경우 제거 후 재시도
                if "position_embeddings" in kwargs:
                    kwargs.pop("position_embeddings", None)
                    kwargs.pop("cache_position", None)
                    out = layer(x, **kwargs)
                else:
                    raise
            
            # GPT-2 특별 처리: 출력이 len=1이고 use_cache=True인 경우
            # GPT-2Block이 use_cache=True일 때도 present를 반환하지 않는 경우
            # 이는 transformers 라이브러리의 버전 문제일 수 있음
            #
            # 참고: 일반적으로 GPT-2Block은 use_cache=True일 때
            # (hidden_states, present)를 반환해야 하지만,
            # 일부 버전에서는 (hidden_states,)만 반환할 수 있음
            #
            # 해결책: attention 모듈을 직접 호출하여 present를 얻기
            # 주의: 이는 중복 계산이 발생하므로 비효율적임
            # 하지만 현재로서는 유일한 해결책일 수 있음
            if use_cache and isinstance(out, (tuple, list)) and len(out) == 1:
                # GPT-2Block인지 확인 (layer_past를 사용하는 모델)
                layer_name = type(layer).__name__
                is_gpt2_style = (layer_name == 'GPT2Block' or 
                                (hasattr(layer, 'attn') and hasattr(layer, 'ln_1') and 
                                 hasattr(layer, 'ln_2') and hasattr(layer, 'mlp')))
                
                if is_gpt2_style and hasattr(layer, 'attn') and hasattr(layer, 'ln_1'):
                    # GPT-2Block 구조: ln_1 -> attn -> ln_2 -> mlp
                    # attention 모듈에 직접 접근하여 present를 얻기
                    # 주의: 레이어를 이미 호출했으므로, ln_1(x)를 다시 계산해야 함
                    attn_input = layer.ln_1(x)
                    
                    # attention 모듈 호출 (layer_past 전달)
                    # 주의: attention_mask, position_ids 등도 전달해야 함
                    attn_kwargs = {}
                    if 'attention_mask' in kwargs:
                        attn_kwargs['attention_mask'] = kwargs['attention_mask']
                    if 'position_ids' in kwargs:
                        attn_kwargs['position_ids'] = kwargs['position_ids']
                    attn_kwargs['layer_past'] = pkv  # None이어도 전달
                    attn_kwargs['use_cache'] = use_cache
                    
                    try:
                        attn_out = layer.attn(attn_input, **attn_kwargs)
                        # attn_out는 (attn_output, present) 또는 attn_output일 수 있음
                        # GPT-2의 경우, attention 모듈이 튜플을 반환할 수 있음
                        if isinstance(attn_out, (tuple, list)):
                            if len(attn_out) >= 2:
                                present = attn_out[1]
                                if present is not None:
                                    # out을 (hidden_states, present) 형태로 수정
                                    out = (out[0], present)
                                    import warnings
                                    warnings.warn(
                                        f"Layer {i} (GPT2Block): Successfully extracted present from attention module. "
                                        f"Present type: {type(present)}, present length: {len(present) if isinstance(present, (tuple, list)) else 'N/A'}"
                                    )
                                else:
                                    # present가 None인 경우
                                    # GPT-2Attention은 use_cache=True이고 layer_past=None일 때 present를 계산하지만 반환하지 않을 수 있음
                                    # 이 경우 attention 모듈의 내부 상태를 확인하거나, 직접 계산해야 함
                                    
                                    # 방법 1: attention 모듈의 내부 속성 확인
                                    if hasattr(layer.attn, 'present'):
                                        present = layer.attn.present
                                        if present is not None:
                                            out = (out[0], present)
                                            import warnings
                                            warnings.warn(f"Layer {i} (GPT2Block): Found present in attention module attribute")
                                    else:
                                        # 방법 2: GPT-2Attention의 forward 메서드를 직접 확인
                                        # GPT-2Attention은 use_cache=True일 때 present를 계산하지만,
                                        # layer_past=None일 때는 반환하지 않을 수 있음
                                        # 이 경우, attention 모듈을 다시 호출하거나 다른 방법 사용
                                        
                                        # 방법 3: 레이어의 forward 메서드를 직접 호출하여 present를 얻기
                                        # 하지만 이는 복잡하므로, 일단 None으로 두고 나중에 처리
                                        
                                        import warnings
                                        warnings.warn(
                                            f"Layer {i} (GPT2Block): Attention module returned None for present. "
                                            f"attn_out structure: type={type(attn_out)}, len={len(attn_out)}, "
                                            f"attn_out[0] type={type(attn_out[0]) if len(attn_out) > 0 else 'N/A'}, "
                                            f"attn_out[1]={attn_out[1] if len(attn_out) > 1 else 'N/A'}, "
                                            f"use_cache={use_cache}, layer_past={pkv is not None}"
                                        )
                            else:
                                # attn_out가 튜플이지만 len < 2인 경우
                                import warnings
                                warnings.warn(
                                    f"Layer {i} (GPT2Block): Attention module output structure unexpected: "
                                    f"type={type(attn_out)}, len={len(attn_out)}"
                                )
                        else:
                            # attn_out가 튜플이 아닌 경우
                            # GPT-2Attention이 단일 텐서만 반환할 수 있음
                            # 이 경우 present를 직접 계산해야 할 수 있음
                            import warnings
                            warnings.warn(
                                f"Layer {i} (GPT2Block): Attention module returned non-tuple output: "
                                f"type={type(attn_out)}. This may indicate a transformers version issue."
                            )
                    except Exception as e:
                        import warnings
                        warnings.warn(
                            f"Layer {i} (GPT2Block): Failed to get present from attention module: {e}. "
                            f"attn_kwargs keys: {list(attn_kwargs.keys())}, "
                            f"attn_input shape: {attn_input.shape if hasattr(attn_input, 'shape') else 'N/A'}"
                        )
            
            # 출력 구조 확인 및 hidden states 추출
            if isinstance(out, (tuple, list)):
                x = out[0]
                # GPT-2의 경우: use_cache=True일 때 (hidden_states, present) 반환해야 함
                # 하지만 현재 len=1이므로, 레이어 내부에서 present를 추출해야 할 수도 있음
                if i == 0 and use_cache:
                    import warnings
                    warnings.warn(
                        f"Layer {i} output structure: type={type(out)}, len={len(out)}, "
                        f"out[0] type={type(out[0])}, out[0] shape={out[0].shape if hasattr(out[0], 'shape') else 'N/A'}, "
                        f"out[1] type={type(out[1]) if len(out) > 1 else 'N/A'}, "
                        f"out[1] value={out[1] if len(out) > 1 else 'N/A'}, "
                        f"kwargs: {kwargs}"
                    )
            elif hasattr(out, "last_hidden_state"):
                x = out.last_hidden_state
            elif hasattr(out, "hidden_states"):
                x = out.hidden_states
            else:
                x = out
            
            # KV 캐시 저장
            if use_cache:
                kv_cache = _get_past_from_output(out)
                
                # GPT-2 특별 처리: 출력이 len=1인 경우, 레이어 내부에서 present 추출 시도
                if kv_cache is None and isinstance(out, (tuple, list)) and len(out) == 1:
                    # GPT-2Block의 경우, attention 모듈에서 present를 직접 가져와야 할 수 있음
                    # GPT-2Block은 내부적으로 attention을 호출하고 present를 저장할 수 있음
                    if hasattr(layer, 'attn'):
                        # GPT-2Block의 attention 모듈 확인
                        attn_module = layer.attn
                        # attention 모듈이 present를 저장했는지 확인
                        # 하지만 일반적으로는 레이어 출력에 포함되어야 함
                        
                        # GPT-2의 경우, 레이어가 (hidden_states,)만 반환하면
                        # attention 모듈을 직접 호출해야 할 수도 있음
                        # 하지만 이는 비효율적이므로, 레이어가 제대로 반환하도록 해야 함
                        
                        # 실제로 GPT-2Block의 forward를 확인해보면,
                        # use_cache=True일 때 (hidden_states, present)를 반환해야 함
                        # 하지만 현재는 (hidden_states,)만 반환되고 있음
                        
                        # 이는 transformers 라이브러리의 버전 문제일 수 있음
                        # 일부 버전에서는 GPT-2Block이 use_cache를 제대로 처리하지 않을 수 있음
                        
                        # 해결책: 레이어를 직접 수정하거나, attention을 직접 호출
                        # 하지만 가장 간단한 방법은 레이어가 제대로 작동하도록 하는 것
                        
                        # 임시 해결책: GPT-2의 경우, 레이어 출력이 (hidden_states,)만 반환되면
                        # attention 모듈을 직접 호출하여 present를 얻기
                        # 하지만 이는 복잡하고 비효율적임
                        
                        # 현재로서는 레이어가 제대로 작동하지 않는 것으로 보임
                        kv_cache = None
                
                if kv_cache is None:
                    # GPT-2의 경우 직접 확인
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        # out[1]이 present일 수 있음
                        potential_cache = out[1]
                        if potential_cache is not None:
                            # GPT-2: present는 (key, value) 튜플
                            if isinstance(potential_cache, (tuple, list)) and len(potential_cache) == 2:
                                kv_cache = potential_cache
                            else:
                                # 이미 올바른 형태일 수도 있음
                                kv_cache = potential_cache
                    
                    if kv_cache is None:
                        # GPT-2의 경우, 레이어가 present를 반환하지 않으면
                        # 첫 번째 호출(prefill)에서는 빈 튜플 생성
                        if pkv is None:
                            # Prefill 단계: GPT-2Block이 present를 반환하지 않으면
                            # 빈 (key, value) 튜플을 생성할 수 없음
                            # 대신 레이어를 다시 호출하거나, 다른 방법 사용
                            
                            # GPT-2Block의 경우, attention 모듈을 직접 호출하여 present를 얻을 수 있음
                            # 하지만 이는 복잡하므로, 일단 빈 튜플 대신 None 사용
                            # 나중에 decode 단계에서 처리
                            
                            # 실제로는 GPT-2Block이 use_cache=True일 때 present를 반환해야 함
                            # 하지만 현재는 반환하지 않으므로, 레이어를 수정하거나
                            # 다른 방법을 사용해야 함
                            
                            # 임시 해결책: 빈 (key, value) 튜플 생성
                            # 하지만 key와 value의 shape를 알 수 없으므로 불가능
                            
                            # 가장 좋은 방법: 레이어가 제대로 작동하도록 수정
                            # 하지만 현재로서는 레이어가 제대로 작동하지 않는 것으로 보임
                            
                            import warnings
                            warnings.warn(
                                f"Layer {i} (type={type(layer).__name__}) did not return KV cache during prefill. "
                                f"Output type: {type(out)}, output length: {len(out) if isinstance(out, (tuple, list)) else 'N/A'}, "
                                f"use_cache={use_cache}, kwargs keys: {list(kwargs.keys())}. "
                                f"This may indicate that the GPT-2Block is not returning 'present' even with use_cache=True. "
                                f"Consider checking the transformers library version or the model implementation."
                            )
                            # None으로 설정하면 나중에 에러 발생
                            kv_cache = None
                        else:
                            # Decode 단계에서는 이전 cache 재사용
                            kv_cache = pkv
                new_past.append(kv_cache)
        
        if use_cache:
            # new_past가 모두 None이 아닌지 확인
            if all(p is None for p in new_past):
                # 첫 번째 호출(prefill)에서는 past_key_values가 None이므로 레이어가 반드시 KV cache를 반환해야 함
                if past_key_values is None:
                    raise RuntimeError(
                        f"All layers returned None for KV cache during prefill. "
                        f"This suggests the model layers are not returning KV cache properly. "
                        f"Layer types: {[type(layer).__name__ for layer in self.layers]}, "
                        f"use_cache={use_cache}"
                    )
                else:
                    # Decode 단계에서도 문제가 있으면 이전 cache 재사용
                    import warnings
                    warnings.warn("All layers returned None for KV cache during decode, reusing previous cache")
                    return x, past_key_values
            return x, tuple(new_past)
        else:
            return x, None


class StageSegment(nn.Module):
    def __init__(self, full, start: int, end: int):
        super().__init__()
        if hasattr(full, 'model') and hasattr(full.model, 'layers'):
            raw_layers = full.model.layers[start:end]
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'h'):
            raw_layers = full.transformer.h[start:end]
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")
        self.config = full.config
        self.layers = nn.ModuleList()
        for layer in raw_layers:
            if type(layer).__name__ == "GPT2Block":
                self.layers.append(GPT2BlockWrapper(layer))
            else:
                self.layers.append(layer)

        sig_params = [inspect.signature(layer.forward).parameters for layer in raw_layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]
        self._supports_position_embeddings = ['position_embeddings' in p for p in sig_params]
        self._supports_cache_position = ['cache_position' in p for p in sig_params]

    def forward(self, hidden_states, position_ids, attention_mask, past_key_values=None, use_cache=True):
        x = hidden_states
        new_past = []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            kwargs = dict(attention_mask=attention_mask, use_cache=use_cache)
            if self._supports_pos_ids[i]:
                kwargs['position_ids'] = position_ids
            if self._supports_past_key_value[i]:
                kwargs['past_key_value'] = pkv
            elif self._supports_layer_past[i]:
                kwargs['layer_past'] = pkv
            try:
                out = layer(x, **kwargs)
            except TypeError as e:
                if "position_embeddings" in kwargs:
                    kwargs.pop("position_embeddings", None)
                    kwargs.pop("cache_position", None)
                    out = layer(x, **kwargs)
                else:
                    raise
            x = out[0]
            if use_cache:
                new_past.append(_get_past_from_output(out))
        return x, tuple(new_past) if use_cache else None


class StageLast(nn.Module):
    def __init__(self, full, start: int):
        super().__init__()
        if hasattr(full, 'model') and hasattr(full.model, 'layers'):
            raw_layers = full.model.layers[start:]
            if hasattr(full.model, 'norm'):
                self.norm = full.model.norm
            elif hasattr(full.model, 'final_layer_norm'):
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'h'):
            raw_layers = full.transformer.h[start:]
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")

        self.layers = nn.ModuleList()
        for layer in raw_layers:
            if type(layer).__name__ == "GPT2Block":
                self.layers.append(GPT2BlockWrapper(layer))
            else:
                self.layers.append(layer)

        self.lm_head = full.lm_head
        self.config = full.config
        sig_params = [inspect.signature(layer.forward).parameters for layer in raw_layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]
        self._supports_position_embeddings = ['position_embeddings' in p for p in sig_params]
        self._supports_cache_position = ['cache_position' in p for p in sig_params]

    def forward(self, hidden_states, position_ids, attention_mask, past_key_values=None, use_cache=True):
        x = hidden_states
        new_past = []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            kwargs = dict(attention_mask=attention_mask, use_cache=use_cache)
            if self._supports_pos_ids[i]:
                kwargs['position_ids'] = position_ids
            if self._supports_past_key_value[i]:
                kwargs['past_key_value'] = pkv
            elif self._supports_layer_past[i]:
                kwargs['layer_past'] = pkv
            try:
                out = layer(x, **kwargs)
            except TypeError as e:
                if "position_embeddings" in kwargs:
                    kwargs.pop("position_embeddings", None)
                    kwargs.pop("cache_position", None)
                    out = layer(x, **kwargs)
                else:
                    raise
            x = out[0]
            if use_cache:
                new_past.append(_get_past_from_output(out))
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, tuple(new_past) if use_cache else None

# Legacy: replaced with load_stage_model for memory efficiency
# def load_full_model(model_name: str, device: torch.device, dtype=torch.bfloat16):
#     full = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=dtype,
#         low_cpu_mem_usage=True,
#         device_map="cpu",
#     )
#     full.eval()
#     return full


def load_stage_model(
    model_name: str,
    device: torch.device,
    role: str,
    *, # arguments below this asterisk are keyword-only
    start: int = 0,
    end: int | None = None,
    dtype=torch.float16,
):
    """
    Load only the layers needed for a stage to reduce memory.

    role:
      - 'stage0': keep embeddings + layers[:end], drop head/norm
      - 'segment': keep layers[start:end], drop embeddings/head/norm
      - 'last': keep layers[start:], norm, lm_head, drop embeddings
    """
    full = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
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
        # Keep embeddings/position encodings for safety; stages consume hidden_states directly,
        # but removing them breaks some model implementations (e.g., GPT-2 expecting wpe).
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "last":
        _prune_layers(full, start, None)
        # Keep embeddings to avoid None access in some model forwards; unused by stage but harmless.
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    else:
        raise ValueError(f"Unknown role: {role}")

    full = full.to(device)
    return full
