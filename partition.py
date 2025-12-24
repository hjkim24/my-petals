import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class Stage0(nn.Module):
    def __init__(self, full, k: int):
        super().__init__()
        self.embed_tokens = full.model.embed_tokens
        self.layers = nn.ModuleList(full.model.layers[:k])
        self.config = full.config

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None, use_cache=True):
        # input_ids: [B, T]
        x = self.embed_tokens(input_ids)  # [B, T, H]
        new_past = []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            out = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            x = out[0]
            if use_cache:
                new_past.append(out[1])
        return x, tuple(new_past) if use_cache else None

class Stage1(nn.Module):
    def __init__(self, full, k: int):
        super().__init__()
        self.layers = nn.ModuleList(full.model.layers[k:])
        self.norm = full.model.norm
        self.lm_head = full.lm_head
        self.config = full.config

    def forward(self, hidden_states, position_ids, attention_mask, past_key_values=None, use_cache=True):
        x = hidden_states
        new_past = []
        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            out = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv,
                use_cache=use_cache,
            )
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