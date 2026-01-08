# single_gpu_check.py
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llama_partition import load_stage_model, Stage0, StageLast


def main():
    parser = argparse.ArgumentParser(description="Single GPU model check for comparison with distributed inference")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for text generation")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                       help="Model dtype: fp16 (default), bf16, fp32")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.92,
                       help="Nucleus sampling p")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--use_cpu_offload", action="store_true",
                       help="Enable CPU offloading: keep model parameters on CPU and move to GPU only when needed")
    parser.add_argument("--keep_layers_on_gpu", type=int, default=0,
                       help="Number of recent layers to keep on GPU when using CPU offloading (default: 0)")
    
    args = parser.parse_args()
    
    # dtype 변환
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    model_name = args.model
    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # CPU 오프로딩 사용 여부에 따라 모델 로딩 방식 결정
    if args.use_cpu_offload:
        # 전체 모델을 StageLast로 구성 (모든 레이어 포함)
        # embeddings는 별도로 처리
        stage_last_full = load_stage_model(
            model_name, device, role="last",
            start=0, dtype=dtype, use_cpu_offload=True
        )
        stage_last = StageLast(
            stage_last_full, 0,
            gpu_device=device,
            keep_layers_on_gpu=args.keep_layers_on_gpu
        )
        
        # embeddings 가져오기
        temp_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if hasattr(temp_model, "model") and hasattr(temp_model.model, "embed_tokens"):
            embed_tokens = temp_model.model.embed_tokens
        elif hasattr(temp_model, "transformer") and hasattr(temp_model.transformer, "wte"):
            embed_tokens = temp_model.transformer.wte
        else:
            raise ValueError("Unsupported model architecture: cannot find embed_tokens")
        
        # embeddings를 CPU로 이동 (오프로딩 모드)
        embed_tokens = embed_tokens.to(torch.device("cpu"))
        del temp_model
        import gc
        gc.collect()
        
        # 모델을 래퍼로 구성
        from src.utils import default_position_ids
        
        class OffloadedModel(torch.nn.Module):
            def __init__(self, embed_tokens, stage_last):
                super().__init__()
                self.embed_tokens = embed_tokens
                self.stage_last = stage_last
                self.config = stage_last.config
            
            def eval(self):
                self.stage_last.eval()
                return self
            
            def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
                # Embeddings (CPU에서 GPU로 이동)
                hidden_states = self.embed_tokens(input_ids).to(device, non_blocking=True)
                
                # position_ids 생성
                position_ids = default_position_ids(past_key_values, hidden_states.shape[1], hidden_states.device)
                
                # StageLast: 모든 레이어 + norm + lm_head
                logits, stage_last_cache = self.stage_last(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=use_cache
                )
                
                from transformers.modeling_outputs import CausalLMOutputWithPast
                return CausalLMOutputWithPast(
                    logits=logits,
                    past_key_values=stage_last_cache if use_cache else None
                )
        
        model = OffloadedModel(embed_tokens, stage_last)
        model.eval()
        print(f"CPU offloading enabled: Model loaded with lazy GPU loading (keep {args.keep_layers_on_gpu} layers on GPU)")
    else:
        # 기존 방식: 전체 모델을 GPU에 로드
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


if __name__ == "__main__":
    main()
