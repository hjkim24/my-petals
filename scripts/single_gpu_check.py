# single_gpu_check.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"  # 여러분이 쓰는 모델 경로로 교체
prompt = "What is the captial of Korea?"           # 비교할 프롬프트
dtype = torch.float16                    # fp16/bf16/fp32 선택
max_new_tokens = 64                      # 생성할 최대 토큰 수
temperature = 1.0                        # 샘플링 temperature
top_p = 0.92                             # nucleus sampling
top_k = 50                               # top-k sampling

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map=None,   # 단일 GPU
)
model.to(device)
model.eval()

# 1. Prefill: 프롬프트 처리
print("=" * 80)
print("PREFILL STAGE")
print("=" * 80)
inputs = tok(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids

with torch.inference_mode():
    outputs = model(**inputs, use_cache=True)
    logits = outputs.logits  # [1, seq_len, vocab]
    past_key_values = outputs.past_key_values
    last_logits = logits[:, -1, :]      # 마지막 토큰 위치
    topk_vals, topk_ids = last_logits.topk(5, dim=-1)

print(f"Device: {device}, dtype: {logits.dtype}")
print(f"Prompt: '{prompt}'")
print(f"Input length: {input_ids.shape[1]}")
print(f"Prefill logits shape: {logits.shape}")
print(f"Prefill logits stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
print(f"Top5 ids: {topk_ids[0].tolist()}")
print(f"Top5 tokens: {[tok.decode([i], skip_special_tokens=True) for i in topk_ids[0].tolist()]}")
print(f"Top5 logits: {[f'{v:.2f}' for v in topk_vals[0].tolist()]}")

# Greedy로 첫 토큰 생성
with torch.inference_mode():
    next_id = int(torch.argmax(last_logits, dim=-1).item())
    next_token_text = tok.decode([next_id], skip_special_tokens=True)
print(f"Greedy next token: {next_id} ('{next_token_text}')")

# 2. Decode: 토큰 생성 반복
print("\n" + "=" * 80)
print("DECODE STAGE")
print("=" * 80)

generated = [next_id]
eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

for step in range(max_new_tokens):
    # 다음 토큰을 입력으로 사용
    next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
    
    with torch.inference_mode():
        outputs = model(
            input_ids=next_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits  # [1, 1, vocab]
        past_key_values = outputs.past_key_values
        next_token_logits = logits[:, -1, :]  # [1, vocab]
    
    # 샘플링 (temperature, top_p, top_k 적용)
    if temperature > 0:
        # Temperature 적용
        scaled_logits = next_token_logits / max(temperature, 1e-5)
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Top-k 필터링
        if top_k > 0 and top_k < probs.size(-1):
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter(-1, topk_idx, topk_probs)
            probs = mask
        
        # Top-p (nucleus) 필터링
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = sorted_probs * keep
            filtered = filtered.scatter(-1, sorted_idx, filtered)
            probs = filtered
        
        # 샘플링
        next_id = int(torch.multinomial(probs, num_samples=1).item())
    else:
        # Greedy
        next_id = int(torch.argmax(next_token_logits, dim=-1).item())
    
    next_token_text = tok.decode([next_id], skip_special_tokens=True)
    generated.append(next_id)
    
    # 로그 출력 (처음 5개와 마지막 5개만 상세히)
    if step < 5 or step >= max_new_tokens - 5:
        top5_vals, top5_ids = next_token_logits.topk(5, dim=-1)
        top5_tokens = [tok.decode([i], skip_special_tokens=True) for i in top5_ids[0].tolist()]
        print(f"Step {step+1}: token={next_id} ('{next_token_text}') | "
              f"top5={top5_ids[0].tolist()[:3]} ({top5_tokens[:3]}) | "
              f"logits: min={next_token_logits.min().item():.2f}, max={next_token_logits.max().item():.2f}")
    else:
        print(f"Step {step+1}: token={next_id} ('{next_token_text}')")
    
    # EOS 체크
    if eos_token_id is not None and next_id == eos_token_id:
        print(f"EOS token generated at step {step+1}")
        break

# 3. 최종 결과
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
generated_text = tok.decode(generated, skip_special_tokens=True)
print(f"Prompt: '{prompt}'")
print(f"Generated: '{generated_text}'")
print(f"Generated tokens: {len(generated)}")
print(f"Unique tokens: {len(set(generated))}")
print(f"Repetition ratio: {(1.0 - len(set(generated)) / len(generated)) * 100:.2f}%")
