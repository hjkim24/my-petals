import torch
import torch.nn as nn
import inspect
from transformers import AutoModelForCausalLM

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
        if hasattr(self, 'pos_embed') and self.pos_embed is not None and position_ids is not None:
            x = x + self.pos_embed(position_ids)
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
                new_past.append(out[1])
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
                new_past.append(out[1])
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
                new_past.append(out[1])
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, tuple(new_past) if use_cache else None

def load_partition(model_name: str, k: int, device: torch.device, dtype=torch.bfloat16):
    full = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    full.eval()
    return full
