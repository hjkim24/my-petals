import torch
import torch.nn as nn
import inspect
from transformers import AutoModelForCausalLM


def _get_past_from_output(out):
    """Return past cache (if any) from a transformer layer output."""
    if hasattr(out, "past_key_values") and out.past_key_values is not None:
        return out.past_key_values
    if hasattr(out, "next_cache") and out.next_cache is not None:
        return out.next_cache
    if isinstance(out, (tuple, list)) and len(out) > 1:
        return out[1]
    return None


class Stage0(nn.Module):
    def __init__(self, full, end: int):
        super().__init__()
        if hasattr(full, 'model') and hasattr(full.model, 'embed_tokens'):
            self.embed_tokens = full.model.embed_tokens
            self.layers = nn.ModuleList(full.model.layers[:end])
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'wte'):
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, 'wpe', None)
            self.layers = nn.ModuleList(full.transformer.h[:end])
        elif hasattr(full, 'model') and hasattr(full.model, 'embed_in'):
            self.embed_tokens = full.model.embed_in
            self.layers = nn.ModuleList(full.model.layers[:end])
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")
        self.config = full.config
        sig_params = [inspect.signature(layer.forward).parameters for layer in self.layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None, use_cache=True):
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
                kwargs['layer_past'] = pkv
            # 실제로 레이어 통과하는 부분
            out = layer(x, **kwargs)
            x = out[0]
            # KV 캐시 저장
            if use_cache:
                new_past.append(_get_past_from_output(out))
        return x, tuple(new_past) if use_cache else None


class StageSegment(nn.Module):
    def __init__(self, full, start: int, end: int):
        super().__init__()
        if hasattr(full, 'model') and hasattr(full.model, 'layers'):
            self.layers = nn.ModuleList(full.model.layers[start:end])
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'h'):
            self.layers = nn.ModuleList(full.transformer.h[start:end])
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")
        self.config = full.config
        sig_params = [inspect.signature(layer.forward).parameters for layer in self.layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]

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
            out = layer(x, **kwargs)
            x = out[0]
            if use_cache:
                new_past.append(_get_past_from_output(out))
        return x, tuple(new_past) if use_cache else None


class StageLast(nn.Module):
    def __init__(self, full, start: int):
        super().__init__()
        if hasattr(full, 'model') and hasattr(full.model, 'layers'):
            self.layers = nn.ModuleList(full.model.layers[start:])
            if hasattr(full.model, 'norm'):
                self.norm = full.model.norm
            elif hasattr(full.model, 'final_layer_norm'):
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'h'):
            self.layers = nn.ModuleList(full.transformer.h[start:])
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}.")

        self.lm_head = full.lm_head
        self.config = full.config
        sig_params = [inspect.signature(layer.forward).parameters for layer in self.layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]

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
            out = layer(x, **kwargs)
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
        low_cpu_mem_usage=True,
        device_map="cpu",
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
        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            full.model.embed_tokens = None
        if hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            full.transformer.wte = None
            if hasattr(full.transformer, "wpe"):
                full.transformer.wpe = None
        if hasattr(full, "model") and hasattr(full.model, "embed_in"):
            full.model.embed_in = None
        if hasattr(full, "lm_head"):
            full.lm_head = None
        if hasattr(full, "model") and hasattr(full.model, "norm"):
            full.model.norm = None
    elif role == "last":
        _prune_layers(full, start, None)
        if hasattr(full, "model") and hasattr(full.model, "embed_tokens"):
            full.model.embed_tokens = None
        if hasattr(full, "transformer") and hasattr(full.transformer, "wte"):
            full.transformer.wte = None
            if hasattr(full.transformer, "wpe"):
                full.transformer.wpe = None
        if hasattr(full, "model") and hasattr(full.model, "embed_in"):
            full.model.embed_in = None
    else:
        raise ValueError(f"Unknown role: {role}")

    full = full.to(device)
    return full
