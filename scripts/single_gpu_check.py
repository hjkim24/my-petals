# single_gpu_check.py
import argparse
import sys
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 분산 추론용 import 제거 - 단일 GPU에서는 불필요


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
    
    # 모델 로드 (CPU 오프로딩 지원)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    if args.use_cpu_offload:
        # CPU 오프로딩 모드: 모델을 CPU에 로드하고 forward 시 필요한 레이어만 GPU로 이동
        model = model.to(torch.device("cpu"))
        
        # 레이어 접근 (LLaMA 구조)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            raise ValueError("Unsupported model architecture: cannot find layers")
        
        # 레이어별 device 추적
        layer_devices = {}
        cpu_device = torch.device("cpu")
        
        # 모든 레이어를 CPU에 유지
        for i, layer in enumerate(layers):
            layer = layer.to(cpu_device)
            layer_devices[i] = cpu_device
        
        # 최근 N개 레이어는 GPU에 유지
        if args.keep_layers_on_gpu > 0:
            num_layers = len(layers)
            for i in range(max(0, num_layers - args.keep_layers_on_gpu), num_layers):
                layers[i] = layers[i].to(device)
                layer_devices[i] = device
        
        # 각 레이어의 forward를 래핑하여 호출 전에 GPU로 이동 (동기적으로)
        num_layers = len(layers)
        
        # 각 레이어마다 고유한 래퍼 함수 생성 (클로저 문제 해결)
        import types
        
        def create_wrapped_forward(idx, orig_fn, cli_args):
            """각 레이어마다 독립적인 래퍼 함수 생성"""
            def wrapped_forward(self, *args, **kwargs):
                # 현재 레이어를 GPU로 이동 (동기적으로 - 모든 서브모듈도 함께 이동)
                if layer_devices.get(idx) != device:
                    layers[idx] = layers[idx].to(device)
                    layer_devices[idx] = device
                    # CUDA 동기화로 이동 완료 보장 (모든 서브모듈도 함께 이동)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                
                # 이전 레이어를 CPU로 이동 (keep_layers_on_gpu 고려)
                if idx > 0:
                    prev_idx = idx - 1
                    if cli_args.keep_layers_on_gpu == 0 or prev_idx < num_layers - cli_args.keep_layers_on_gpu:
                        if layer_devices.get(prev_idx) == device:
                            layers[prev_idx] = layers[prev_idx].to(cpu_device)
                            layer_devices[prev_idx] = cpu_device
                
                # 입력 텐서를 GPU로 이동 (레이어가 GPU에 있으면)
                if layer_devices.get(idx) == device:
                    if args and len(args) > 0 and isinstance(args[0], torch.Tensor):
                        args = (args[0].to(device),) + args[1:]
                    if "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                        kwargs["hidden_states"] = kwargs["hidden_states"].to(device)
                    if "attention_mask" in kwargs and isinstance(kwargs["attention_mask"], torch.Tensor):
                        kwargs["attention_mask"] = kwargs["attention_mask"].to(device)
                    if "position_ids" in kwargs and isinstance(kwargs["position_ids"], torch.Tensor):
                        kwargs["position_ids"] = kwargs["position_ids"].to(device)
                    if "cache_position" in kwargs and isinstance(kwargs["cache_position"], torch.Tensor):
                        kwargs["cache_position"] = kwargs["cache_position"].to(device)
                
                # 원본 forward 호출
                return orig_fn(self, *args, **kwargs)
            return wrapped_forward
        
        for i, layer in enumerate(layers):
            original_forward = layer.__class__.forward
            wrapped_fn = create_wrapped_forward(i, original_forward, args)
            layer.forward = types.MethodType(wrapped_fn, layer)
        
        # Embeddings와 norm, lm_head는 항상 GPU에 유지 (작고 자주 사용)
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            model.transformer.wte = model.transformer.wte.to(device)

        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "rotary_emb"):
            model.transformer.rotary_emb = model.transformer.rotary_emb.to(device)
        
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.to(device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            model.transformer.ln_f = model.transformer.ln_f.to(device)
        
        if hasattr(model, "lm_head"):
            model.lm_head = model.lm_head.to(device)
        
        print(f"CPU offloading enabled: Model loaded with lazy GPU loading (keep {args.keep_layers_on_gpu} layers on GPU)")
    else:
        # 일반 모드: 전체 모델을 GPU에 로드
        model = model.to(device)
    
    model.eval()
    
    # 1. Prefill: 프롬프트 처리
    print("=" * 80)
    print("PREFILL STAGE")
    print("=" * 80)
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Prefill 시간 측정 시작
    t_prefill_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
        logits = outputs.logits  # [1, seq_len, vocab]
        past_key_values = outputs.past_key_values
        last_logits = logits[:, -1, :]      # 마지막 토큰 위치
        topk_vals, topk_ids = last_logits.topk(5, dim=-1)
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start
    
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
    print(f"TTFT (Time to First Token): {prefill_time:.3f}s")
    
    # 2. Decode: 토큰 생성 반복
    print("\n" + "=" * 80)
    print("DECODE STAGE")
    print("=" * 80)
    
    generated = [next_id]
    eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    
    # Decode 시간 측정 시작
    t_decode_start = time.perf_counter()
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
    
    # Decode 시간 측정 종료
    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    total_time = t_decode_end - t_prefill_start
    
    # 생성된 토큰 수 (prefill에서 생성된 첫 토큰 포함)
    num_generated_tokens = len(generated)  # prefill에서 1개 + decode에서 생성된 토큰들
    
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
    
    # 성능 지표 출력
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"TTFT (Time to First Token): {prefill_time:.3f}s")
    print(f"Decode time: {decode_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    if decode_time > 0:
        decode_tokens = num_generated_tokens - 1  # prefill에서 생성된 첫 토큰 제외
        decoding_speed = decode_tokens / decode_time if decode_tokens > 0 else 0.0
        print(f"Decoding speed: {decoding_speed:.2f} tokens/s")
    else:
        print(f"Decoding speed: N/A (no decode tokens)")
    if total_time > 0:
        throughput = num_generated_tokens / total_time
        print(f"Throughput: {throughput:.2f} tokens/s")
    else:
        print(f"Throughput: N/A")


if __name__ == "__main__":
    main()
