# single_gpu_check.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"  # 여러분이 쓰는 모델 경로로 교체
prompt = "Hello, how are you?"           # 비교할 프롬프트
dtype = torch.float16                    # fp16/bf16/fp32 선택

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
inputs = tok(prompt, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map=None,   # 단일 GPU
)
model.eval()

with torch.inference_mode():
    out = model(**inputs, use_cache=True)
    logits = out.logits  # [1, seq_len, vocab]
    last_logits = logits[:, -1, :]      # 마지막 토큰 위치
    topk_vals, topk_ids = last_logits.topk(5, dim=-1)

print("Top5 ids:", topk_ids[0].tolist())
print("Top5 logits:", topk_vals[0].tolist())

# greedy로 한 토큰 생성
with torch.inference_mode():
    next_id = int(torch.argmax(last_logits, dim=-1).item())
print("Greedy next token:", next_id, tok.decode([next_id], skip_special_tokens=True))
