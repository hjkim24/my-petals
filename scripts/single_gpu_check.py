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

# 양자화 관련 import
from src.llama_partition import QuantType, quantize_module


def main():
    parser = argparse.ArgumentParser(description="Single GPU model check for comparison with distributed inference")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt for text generation")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "int4", "int8"],
                       help="Model dtype: fp16 (default), bf16, fp32, int4 (NF4), int8")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.92,
                       help="Nucleus sampling p")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--disable_eos", action="store_true",
                       help="Disable EOS token generation (mask EOS token in sampling)")
    parser.add_argument("--use_cpu_offload", action="store_true",
                       help="Enable CPU offloading: keep model parameters on CPU and move to GPU only when needed")
    parser.add_argument("--keep_layers_on_gpu", type=int, default=0,
                       help="Number of recent layers to keep on GPU when using CPU offloading (default: 0)")
    
    args = parser.parse_args()
    
    # dtype 변환 및 양자화 타입 결정
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    
    # 양자화 타입 결정
    if args.dtype == "int4":
        quant_type = QuantType.NF4
        dtype = torch.float16  # 양자화 시 base dtype은 fp16 사용
    elif args.dtype == "int8":
        quant_type = QuantType.INT8
        dtype = torch.float16  # 양자화 시 base dtype은 fp16 사용
    else:
        quant_type = QuantType.NONE
        dtype = dtype_map[args.dtype]
    
    model_name = args.model
    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    disable_eos = args.disable_eos
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # 모델 로드 (양자화를 위해 항상 CPU에 먼저 로드)
    print(f"Loading model: {model_name}")
    print(f"Quantization: {quant_type.name if quant_type != QuantType.NONE else 'None'}")
    
    # 양자화가 필요한 경우 CPU에 로드 (양자화는 CPU에서 수행)
    if quant_type != QuantType.NONE:
        print("Loading model on CPU for quantization...")
        # 대용량 모델 로딩을 위한 메모리 제한 및 디스크 오프로딩 설정
        import psutil
        import tempfile
        import os
        
        available_memory = psutil.virtual_memory().available
        # BLOOM-176B INT4는 약 93.5GB 필요, 피크 메모리 ~200GB
        # 사용 가능한 메모리의 75%를 사용하되, 최소 250GB는 보장 (충분한 메모리가 있을 때)
        min_required_mb = 250 * 1024  # 최소 250GB (충분한 메모리가 있을 때)
        calculated_mb = int(available_memory * 0.75 / (1024 * 1024))
        max_memory_mb = max(min_required_mb, calculated_mb)
        
        # 사용 가능한 메모리가 300GB 이상이면 제한을 더 완화
        if available_memory / (1024**3) >= 300:
            calculated_mb = int(available_memory * 0.8 / (1024 * 1024))
            max_memory_mb = calculated_mb
            print(f"Available memory: {available_memory / (1024**3):.1f}GB, using 80% = {max_memory_mb / 1024:.1f}GB (large memory mode)")
        else:
            print(f"Available memory: {available_memory / (1024**3):.1f}GB, using 75% = {max_memory_mb / 1024:.1f}GB")
        
        max_memory = {"cpu": f"{max_memory_mb}MiB"}
        
        # 디스크 오프로딩을 위한 임시 디렉토리 생성
        offload_folder = tempfile.mkdtemp(prefix="model_offload_")
        print(f"Using disk offloading folder: {offload_folder}")
        
        try:
            # 메모리 사용량을 최소화하기 위한 추가 옵션
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="cpu",  # 양자화를 위해 CPU에 로드
                max_memory=max_memory,  # 메모리 사용량 제한
                offload_folder=offload_folder,  # 디스크 오프로딩
                # 추가 메모리 최적화 옵션
                use_safetensors=True,  # safetensors 사용 (더 안전하고 메모리 효율적)
            )
        except Exception as e:
            # 오프로딩 폴더 정리
            import shutil
            try:
                shutil.rmtree(offload_folder)
            except:
                pass
            raise e
        
        # 양자화 적용
        # 4bit 양자화의 경우 compute_dtype을 float16으로 설정하여 입력 dtype과 일치시킴
        compute_dtype = torch.float16 if quant_type == QuantType.NF4 else None
        if compute_dtype is not None:
            print(f"Setting compute_dtype to {compute_dtype} for 4bit quantization")
        print(f"Applying {quant_type.name} quantization...")
        model = quantize_module(model, quant_type=quant_type, compute_dtype=compute_dtype)
        
        # 양자화 검증
        try:
            import bitsandbytes as bnb
            quantized_modules = []
            linear_modules = []
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    linear_modules.append(name)
                if isinstance(m, (bnb.nn.LinearNF4, bnb.nn.Linear8bitLt)):
                    quantized_modules.append(name)
            
            total_quantized = len(quantized_modules)
            total_linear = len(linear_modules)
            quant_ratio = (total_quantized / total_linear * 100) if total_linear > 0 else 0
            
            print(f"Quantization verification: {total_quantized} quantized Linear layers "
                  f"out of {total_linear} total Linear layers ({quant_ratio:.1f}%)")
        except ImportError:
            pass
        
        print(f"Quantization with {quant_type.name} completed")
        
        # 양자화 완료 후 오프로딩 폴더 정리 (메모리 절약)
        import shutil
        try:
            if 'offload_folder' in locals() and os.path.exists(offload_folder):
                print(f"Cleaning up offload folder: {offload_folder}")
                shutil.rmtree(offload_folder)
        except Exception as e:
            print(f"Warning: Failed to clean up offload folder: {e}")
    else:
        # 양자화 없이 일반 로드
        # CPU offload를 사용할 경우 메모리 제한 및 디스크 오프로딩 사용
        if args.use_cpu_offload:
            import psutil
            import tempfile
            
            available_memory = psutil.virtual_memory().available
            # 다른 프로세스와 공유하므로 매우 보수적으로 25%만 사용
            max_memory_mb = int(available_memory * 0.25 / (1024 * 1024))
            max_memory = {"cpu": f"{max_memory_mb}MiB"}
            print(f"Available memory: {available_memory / (1024**3):.1f}GB, limiting to {max_memory_mb / 1024:.1f}GB (25% for safety)")
            
            # 디스크 오프로딩을 위한 임시 디렉토리 생성
            offload_folder = tempfile.mkdtemp(prefix="model_offload_")
            print(f"Using disk offloading folder: {offload_folder}")
            
            try:
                # 메모리 사용량을 최소화하기 위한 추가 옵션
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    device_map="cpu",  # CPU에 직접 로드
                    max_memory=max_memory,  # 메모리 사용량 제한
                    offload_folder=offload_folder,  # 디스크 오프로딩
                    # 추가 메모리 최적화 옵션
                    use_safetensors=True,  # safetensors 사용 (더 안전하고 메모리 효율적)
                )
            except Exception as e:
                # 오프로딩 폴더 정리
                import shutil
                try:
                    shutil.rmtree(offload_folder)
                except:
                    pass
                raise e
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
    
    if args.use_cpu_offload:
        # CPU 오프로딩 모드: 모델을 CPU에 로드하고 forward 시 필요한 레이어만 GPU로 이동
        # 양자화된 모델은 이미 CPU에 있으므로 이동 불필요
        if quant_type == QuantType.NONE:
            model = model.to(torch.device("cpu"))
        
        # 레이어 접근 (다양한 모델 구조 지원)
        layers = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA/Mistral 구조
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT/BLOOM 구조
            layers = model.transformer.h
        else:
            raise ValueError(f"Unsupported model architecture: cannot find layers. Model type: {type(model)}, has model.model: {hasattr(model, 'model')}, has transformer: {hasattr(model, 'transformer')}")
        
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
        # LLaMA 구조
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            model.model.embed_tokens = model.model.embed_tokens.to(device)
        # GPT 구조
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            model.transformer.wte = model.transformer.wte.to(device)
        # BLOOM 구조
        elif hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
            # BLOOM의 word_embeddings_layernorm도 GPU로 이동
            if hasattr(model.transformer, "word_embeddings_layernorm"):
                model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)

        # Rotary embeddings (LLaMA/Mistral)
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "rotary_emb"):
            model.transformer.rotary_emb = model.transformer.rotary_emb.to(device)
        
        # Final layer norm
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            model.model.norm = model.model.norm.to(device)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            model.transformer.ln_f = model.transformer.ln_f.to(device)
        
        # Language model head
        if hasattr(model, "lm_head"):
            model.lm_head = model.lm_head.to(device)
        
        # 모델의 forward pass를 래핑하여 입력 처리 전에 필요한 모듈들을 GPU로 이동
        original_forward = model.__class__.forward
        
        def wrapped_model_forward(self, *args, **kwargs):
            # 입력 텐서를 GPU로 이동 (이미 GPU에 있으면 no-op)
            if args and len(args) > 0 and isinstance(args[0], torch.Tensor):
                args = (args[0].to(device),) + args[1:]
            if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
                kwargs["input_ids"] = kwargs["input_ids"].to(device)
            if "inputs_embeds" in kwargs and isinstance(kwargs["inputs_embeds"], torch.Tensor):
                kwargs["inputs_embeds"] = kwargs["inputs_embeds"].to(device)
            if "attention_mask" in kwargs and isinstance(kwargs["attention_mask"], torch.Tensor):
                kwargs["attention_mask"] = kwargs["attention_mask"].to(device)
            if "position_ids" in kwargs and isinstance(kwargs["position_ids"], torch.Tensor):
                kwargs["position_ids"] = kwargs["position_ids"].to(device)
            
            # Embeddings와 norm이 GPU에 있는지 확인하고 없으면 이동
            def check_and_move(module, target_device):
                """모듈이 target_device에 없으면 이동"""
                try:
                    param = next(module.parameters(), None)
                    if param is not None and param.device != target_device:
                        module.to(target_device)
                except StopIteration:
                    # 파라미터가 없는 경우에도 디바이스 확인
                    try:
                        buffer = next(module.buffers(), None)
                        if buffer is not None and buffer.device != target_device:
                            module.to(target_device)
                    except StopIteration:
                        # 파라미터와 버퍼가 모두 없는 경우에도 이동 시도
                        module.to(target_device)
            
            # LLaMA 구조
            if hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
                check_and_move(self.model.embed_tokens, device)
            # GPT 구조
            elif hasattr(self, "transformer") and hasattr(self.transformer, "wte"):
                check_and_move(self.transformer.wte, device)
            # BLOOM 구조
            elif hasattr(self, "transformer") and hasattr(self.transformer, "word_embeddings"):
                check_and_move(self.transformer.word_embeddings, device)
                if hasattr(self.transformer, "word_embeddings_layernorm"):
                    check_and_move(self.transformer.word_embeddings_layernorm, device)
            
            # Final norm
            if hasattr(self, "model") and hasattr(self.model, "norm"):
                check_and_move(self.model.norm, device)
            elif hasattr(self, "transformer") and hasattr(self.transformer, "ln_f"):
                check_and_move(self.transformer.ln_f, device)
            
            # LM head
            if hasattr(self, "lm_head"):
                check_and_move(self.lm_head, device)
            
            return original_forward(self, *args, **kwargs)
        
        model.forward = types.MethodType(wrapped_model_forward, model)
        
        print(f"CPU offloading enabled: Model loaded with lazy GPU loading (keep {args.keep_layers_on_gpu} layers on GPU)")
    else:
        # 일반 모드: 전체 모델을 GPU에 로드
        # 양자화된 모델은 bitsandbytes가 자동으로 device placement 처리
        if quant_type == QuantType.NONE:
            model = model.to(device)
        else:
            # 양자화된 모델은 GPU로 이동 (bitsandbytes가 자동 처리)
            print("Moving quantized model to GPU (bitsandbytes will handle device placement)...")
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
        
        # EOS 토큰 마스킹 (disable_eos 옵션이 켜져있으면)
        # inference_mode 밖에서 수정해야 하므로 clone 사용
        if disable_eos and eos_token_id is not None:
            next_token_logits = next_token_logits.clone()
            next_token_logits[0, eos_token_id] = float('-inf')
        
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
