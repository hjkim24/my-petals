import torch
import torch.nn as nn
import inspect
from transformers import AutoModelForCausalLM

class Stage0(nn.Module):
    def __init__(self, full, k: int):
        super().__init__()
        # Support different model architectures
        # LLaMA/Mistral/Qwen: model.embed_tokens, model.layers
        # GPT-2: transformer.wte, transformer.h
        # GPT-NeoX: model.embed_in, model.layers
        if hasattr(full, 'model') and hasattr(full.model, 'embed_tokens'):
            # LLaMA, Mistral, Qwen, etc.
            self.embed_tokens = full.model.embed_tokens
            self.layers = nn.ModuleList(full.model.layers[:k])
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'wte'):
            # GPT-2, GPT-Neo, etc.
            self.embed_tokens = full.transformer.wte
            self.pos_embed = getattr(full.transformer, 'wpe', None)
            self.layers = nn.ModuleList(full.transformer.h[:k])
        elif hasattr(full, 'model') and hasattr(full.model, 'embed_in'):
            # GPT-NeoX, etc.
            self.embed_tokens = full.model.embed_in
            self.layers = nn.ModuleList(full.model.layers[:k])
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}. "
                           "Expected LLaMA (model.embed_tokens), GPT-2 (transformer.wte), or GPT-NeoX (model.embed_in) structure.")
        self.config = full.config
        sig_params = [inspect.signature(layer.forward).parameters for layer in self.layers]
        self._supports_pos_ids = ['position_ids' in p for p in sig_params]
        self._supports_past_key_value = ['past_key_value' in p for p in sig_params]
        self._supports_layer_past = ['layer_past' in p for p in sig_params]

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None, use_cache=True):
        # input_ids: [B, T]
        x = self.embed_tokens(input_ids)  # [B, T, H]
        if hasattr(self, 'pos_embed') and self.pos_embed is not None and position_ids is not None:
            # GPT-2 style learned positional embeddings
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

class Stage1(nn.Module):
    def __init__(self, full, k: int):
        super().__init__()
        # Support different model architectures
        if hasattr(full, 'model') and hasattr(full.model, 'layers'):
            # LLaMA, Mistral, Qwen, GPT-NeoX, etc.
            self.layers = nn.ModuleList(full.model.layers[k:])
            if hasattr(full.model, 'norm'):
                # LLaMA, Mistral, Qwen
                self.norm = full.model.norm
            elif hasattr(full.model, 'final_layer_norm'):
                # GPT-NeoX
                self.norm = full.model.final_layer_norm
            else:
                raise ValueError(f"Unsupported model: no norm layer found in {type(full.model)}")
        elif hasattr(full, 'transformer') and hasattr(full.transformer, 'h'):
            # GPT-2, GPT-Neo
            self.layers = nn.ModuleList(full.transformer.h[k:])
            self.norm = full.transformer.ln_f
        else:
            raise ValueError(f"Unsupported model architecture: {type(full)}. "
                           "Expected LLaMA (model.layers), GPT-2 (transformer.h), or GPT-NeoX (model.layers) structure.")
        
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
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits, tuple(new_past) if use_cache else None

def load_partition(model_name: str, k: int, device: torch.device, dtype=torch.bfloat16):
    full = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    full.eval()
    return full
