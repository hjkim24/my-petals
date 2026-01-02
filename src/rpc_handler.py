"""
RPC handler for Stage1 server using hivemind ConnectionHandler.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, List, Optional

import torch
from hivemind import DHT, MSGPackSerializer, P2PContext, deserialize_torch_tensor, serialize_torch_tensor
from hivemind.compression.serialization import deserialize_tensor_stream

# Use asyncio.timeout for Python 3.11+, fallback to async_timeout for older versions
try:
    # Python 3.11+
    timeout = asyncio.timeout
except AttributeError:
    # Older Python versions
    try:
        from async_timeout import timeout
    except ImportError:
        # Last resort: create a simple timeout context manager
        @asynccontextmanager
        async def timeout(seconds):
            task = asyncio.current_task()
            if task:
                async def cancel_after():
                    await asyncio.sleep(seconds)
                    task.cancel()
                cancel_task = asyncio.create_task(cancel_after())
                try:
                    yield
                finally:
                    cancel_task.cancel()
            else:
                yield
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.proto import runtime_pb2
from hivemind.utils.logging import get_logger

try:
    from .llama_partition import StageSegment, StageLast
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.llama_partition import StageSegment, StageLast

logger = get_logger(__name__)


class StageConnectionHandler(ConnectionHandler):
    """Connection handler for Stage1 that processes forward requests."""

    def __init__(
        self,
        dht: DHT,
        stage_model,
        device: torch.device,
        request_timeout: float = 30.0,
        final_stage: bool = True,
    ):
        """
        Args:
            dht: DHT instance for peer discovery
            stage_model: Stage model instance
            device: PyTorch device
            request_timeout: Timeout for RPC requests in seconds
            final_stage: If True, sample and return token; else return hidden states
        """
        # ConnectionHandler expects module_backends dict, but we only have one model
        # Create a dummy dict with a single entry
        module_backends = {"stage1": stage_model}
        super().__init__(dht, module_backends, start=False)
        
        self.stage_model = stage_model
        self.device = device
        self.request_timeout = request_timeout
        self._kv_cache: Dict[str, Optional[tuple]] = {}
        self._default_temperature = 0.8
        self._default_top_p = 0.9
        self._default_top_k = 0
        self.final_stage = final_stage

    def _convert_cache_dtype(self, past_key_values, target_dtype: torch.dtype, device: torch.device):
        """Convert past_key_values (KV cache) to target dtype and device."""
        try:
            from transformers.cache_utils import Cache
        except Exception:
            Cache = None
        
        # Handle transformers Cache objects
        if Cache is not None and isinstance(past_key_values, Cache):
            # Cache 객체는 내부적으로 dtype 변환을 지원할 수 있음
            # 필요시 각 key/value를 변환
            return past_key_values
        
        # Handle legacy tuple cache: tuple of tuples ((key, value), ...)
        if isinstance(past_key_values, (tuple, list)):
            converted = []
            for layer_cache in past_key_values:
                if layer_cache is None:
                    converted.append(None)
                elif isinstance(layer_cache, (tuple, list)) and len(layer_cache) == 2:
                    # (key, value) tuple
                    key, value = layer_cache
                    if key is not None:
                        key = key.to(device=device, dtype=target_dtype)
                    if value is not None:
                        value = value.to(device=device, dtype=target_dtype)
                    converted.append((key, value))
                else:
                    converted.append(layer_cache)
            return tuple(converted)
        
        return past_key_values

    @staticmethod
    def _past_len(past_key_values, cur_len: int, chunk_len: int) -> int:
        """Safely derive past sequence length for cache-aware masking."""
        # HuggingFace Cache object (e.g., DynamicCache for LLaMA)
        try:
            from transformers.cache_utils import Cache  # type: ignore
        except Exception:
            Cache = None

        if Cache is not None and isinstance(past_key_values, Cache):
            return past_key_values.get_seq_length(0)
        if not past_key_values:
            return max(cur_len - chunk_len, 0)
        first = past_key_values[0] if len(past_key_values) > 0 else None
        if first is None:
            return max(cur_len - chunk_len, 0)
        if isinstance(first, (tuple, list)) and len(first) > 0 and first[0] is not None:
            # KV cache의 실제 sequence length 반환
            kv_seq_len = first[0].shape[-2]
            # KV cache 길이가 예상과 다를 수 있으므로, 더 안전한 방법 사용
            # decode 단계에서는 항상 1개 토큰씩 처리하므로, past_len은 kv_seq_len과 같아야 함
            return kv_seq_len
        return max(cur_len - chunk_len, 0)

    def _build_masks(
        self, seq_len: int, cur_len: int, is_prefill: bool, hidden_states: torch.Tensor, past_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create attention mask and position ids for prefill/decode."""
        if is_prefill:
            attn_mask = None  # causal mask internal to GPT-style blocks
            # Prefill: 전체 시퀀스의 position IDs [0, 1, 2, ..., seq_len-1]
            pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
        else:
            attn_mask = None
            # Decode: 새 토큰의 position ID는 past_len부터 시작
            # hidden_states.shape[1]은 항상 1 (decode 단계에서는 1개 토큰씩 처리)
            new_token_pos = past_len
            pos_ids = torch.tensor([[new_token_pos]], device=self.device, dtype=torch.long)
        return attn_mask, pos_ids

    def _run_forward(
        self, hidden_states: torch.Tensor, metadata: Dict
    ) -> runtime_pb2.ExpertResponse:
        """Shared forward logic for unary/stream requests."""
        session_id = metadata.get("session_id")
        if session_id is None:
            raise ValueError("request.metadata must contain session_id")

        is_prefill = bool(metadata.get("is_prefill", False))
        seq_len = int(metadata.get("seq_len", hidden_states.shape[1]))
        cur_len = int(metadata.get("cur_len", seq_len))
        temperature = float(metadata.get("temperature", self._default_temperature))
        top_p = float(metadata.get("top_p", self._default_top_p))
        top_k = int(metadata.get("top_k", self._default_top_k))
        repetition_penalty = float(metadata.get("repetition_penalty", 1.5))  # 기본값 증가
        generated_tokens = metadata.get("generated_tokens", [])

        # ========== PREFILL vs DECODE 구분 로그 ==========
        stage_name = "PREFILL" if is_prefill else "DECODE"
        logger.info(f"[{session_id[:8]}] ========== {stage_name} ==========")
        logger.info(f"[{session_id[:8]}] Input hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}, "
                   f"min={hidden_states.min().item():.4f}, max={hidden_states.max().item():.4f}, "
                   f"mean={hidden_states.mean().item():.4f}, std={hidden_states.std().item():.4f}")
        
        if is_prefill:
            past_key_values = None
            past_len = 0
            logger.info(f"[{session_id[:8]}] PREFILL: seq_len={seq_len}, cur_len={cur_len}, past_len=0 (no cache)")
        else:
            past_key_values = self._kv_cache.get(session_id)
            if past_key_values is None:
                raise ValueError(f"Missing past_key_values for session_id={session_id}")
            past_len = StageConnectionHandler._past_len(past_key_values, cur_len, hidden_states.shape[1])
            # 디버깅: past_len이 올바른지 확인 (INFO 레벨로 변경하여 항상 출력)
            expected_past_len = cur_len - hidden_states.shape[1]
            if past_len != expected_past_len:
                pkv_type = type(past_key_values).__name__
                try:
                    from transformers.cache_utils import Cache  # type: ignore
                except Exception:
                    Cache = None
                cache_len = None
                if Cache is not None and isinstance(past_key_values, Cache):
                    try:
                        cache_len = past_key_values.get_seq_length()
                    except Exception:
                        cache_len = "error"
                logger.warning(
                    f"[{session_id[:8]}] DECODE: Past len mismatch! past_len={past_len}, cur_len={cur_len}, "
                    f"hidden_shape={hidden_states.shape[1]}, expected={expected_past_len}, "
                    f"pkv_type={pkv_type}, cache_len={cache_len}"
                )
            else:
                logger.info(
                    f"[{session_id[:8]}] DECODE: seq_len={seq_len}, cur_len={cur_len}, past_len={past_len}, "
                    f"hidden_shape={hidden_states.shape}"
                )

        attn_mask, pos_ids = self._build_masks(seq_len, cur_len, is_prefill, hidden_states, past_len)
        
        # Position IDs 로그
        logger.info(f"[{session_id[:8]}] {stage_name}: Position IDs={pos_ids.tolist()}, past_len={past_len}")

        with torch.inference_mode():
            cfg_dtype = getattr(getattr(self.stage_model, "config", None), "torch_dtype", None)
            first_param = next(self.stage_model.parameters(), None)
            model_dtype = cfg_dtype or (first_param.dtype if first_param is not None else hidden_states.dtype)
            inputs = hidden_states.to(self.device, dtype=model_dtype)
            logger.info(f"[{session_id[:8]}] {stage_name}: Converted inputs dtype={inputs.dtype} (model_dtype={model_dtype})")
            
            # Convert past_key_values to match model dtype and device
            if past_key_values is not None:
                past_key_values = self._convert_cache_dtype(past_key_values, model_dtype, self.device)
                logger.info(f"[{session_id[:8]}] {stage_name}: Converted past_key_values to dtype={model_dtype}")
            
            try:
                outputs, new_past = self.stage_model(
                    inputs,
                    position_ids=pos_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            except StopIteration as e:
                logger.error(
                    f"[{session_id[:8]}] StopIteration from stage_model: "
                    f"inputs dtype={inputs.dtype}, shape={inputs.shape}, "
                    f"pos_ids={pos_ids}, past_type={type(past_key_values)}"
                )
                raise RuntimeError("stage_model raised StopIteration") from e

        self._kv_cache[session_id] = new_past

        if self.final_stage:
            logits = outputs
            # ========== FINAL STAGE (Stage3) 로그 ==========
            logger.info(f"[{session_id[:8]}] {stage_name} [FINAL]: logits shape={logits.shape}, dtype={logits.dtype}, "
                       f"min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}, std={logits.std().item():.2f}")
            # logits shape: [batch, seq_len, vocab_size] 또는 [batch, vocab_size]
            if logits.dim() == 3:
                # [batch, seq_len, vocab_size] -> [batch, vocab_size] (마지막 토큰)
                next_token_logits = logits[:, -1, :]
            elif logits.dim() == 2:
                # 이미 [batch, vocab_size] 형태
                next_token_logits = logits
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            # 디버깅: 샘플링 전 top5 logits 확인
            top5_logits, top5_indices = next_token_logits.topk(5, dim=-1)
            logger.info(f"[{session_id[:8]}] {stage_name} [FINAL]: Top5 logits before sampling: "
                       f"indices={top5_indices[0].tolist()}, values={[f'{v:.2f}' for v in top5_logits[0].tolist()]}")
            
            next_token_id = int(self._sample_token(
                next_token_logits, 
                temperature, 
                top_p, 
                top_k,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated_tokens
            ))
            # 샘플링된 토큰의 확률 확인
            probs = torch.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
            sampled_prob = probs[0, next_token_id].item()
            logger.info(f"[{session_id[:8]}] {stage_name} [FINAL]: Sampled token={next_token_id}, "
                       f"probability={sampled_prob:.4f}, logits shape={next_token_logits.shape}")
            response_metadata = {"token_id": next_token_id, "session_id": session_id}
            token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            serialized_token = serialize_torch_tensor(token_tensor.cpu())
            return runtime_pb2.ExpertResponse(
                tensors=[serialized_token],
                metadata=MSGPackSerializer.dumps(response_metadata),
            )
        else:
            hidden_out = outputs
            # ========== INTERMEDIATE STAGE (Stage1, Stage2) 로그 ==========
            logger.info(f"[{session_id[:8]}] {stage_name} [INTERMEDIATE]: output shape={hidden_out.shape}, input shape={hidden_states.shape}")
            logger.info(f"[{session_id[:8]}] {stage_name} [INTERMEDIATE]: output stats - "
                       f"min={hidden_out.min().item():.4f}, max={hidden_out.max().item():.4f}, "
                       f"mean={hidden_out.mean().item():.4f}, std={hidden_out.std().item():.4f}, dtype={hidden_out.dtype}")
            
            # 활성화값 폭발 감지
            if abs(hidden_out.min().item()) > 100 or abs(hidden_out.max().item()) > 100:
                logger.warning(f"[{session_id[:8]}] {stage_name} [INTERMEDIATE]: ⚠️ Large activation values detected! "
                             f"min={hidden_out.min().item():.4f}, max={hidden_out.max().item():.4f}")
            serialized_hidden = serialize_torch_tensor(hidden_out.cpu())
            response_metadata = {"session_id": session_id}
            return runtime_pb2.ExpertResponse(
                tensors=[serialized_hidden],
                metadata=MSGPackSerializer.dumps(response_metadata),
            )

    def _sample_token(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int, 
                      repetition_penalty: float = 1.2, generated_tokens: Optional[List[int]] = None) -> int:
        """Apply temperature / nucleus / top-k sampling with repetition penalty."""
        # Greedy path to avoid div/0 and CUDA asserts when temperature==0
        if temperature <= 0.0:
            last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
            topk_vals, topk_ids = last_logits.topk(5, dim=-1)
            logger.info(f"Top5 (greedy) ids={topk_ids[0].tolist()}, vals={topk_vals[0].tolist()}")
            return int(torch.argmax(logits, dim=-1).item())

        temp = max(temperature, 1e-5)
        last_logits = logits[:, -1, :] if logits.dim() == 3 else logits
        topk_vals, topk_ids = last_logits.topk(5, dim=-1)
        logger.info(f"Top5 ids={topk_ids[0].tolist()}, vals={topk_vals[0].tolist()}, temp={temperature}, top_p={top_p}, top_k={top_k}")
        
        # Repetition penalty 적용 - 더 강하게
        if repetition_penalty != 1.0 and generated_tokens:
            # 최근 50개 토큰 체크 (범위 확대)
            recent_tokens = generated_tokens[-50:] if len(generated_tokens) > 50 else generated_tokens
            
            # 각 토큰의 반복 횟수에 따라 패널티 적용
            token_counts = {}
            for token_id in recent_tokens:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            for token_id, count in token_counts.items():
                if token_id < logits.shape[-1]:
                    # 반복 횟수에 따라 패널티 증가
                    penalty = repetition_penalty ** count
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= penalty
                    else:
                        logits[0, token_id] *= penalty
            
            # 연속 반복 방지 (최근 3개 토큰이 모두 같으면 강한 패널티)
            if len(generated_tokens) >= 3:
                last_three = generated_tokens[-3:]
                if len(set(last_three)) == 1:  # 모두 같은 토큰
                    repeated_token = last_three[0]
                    if repeated_token < logits.shape[-1]:
                        # 매우 강한 패널티
                        strong_penalty = repetition_penalty ** 3
                        if logits[0, repeated_token] > 0:
                            logits[0, repeated_token] /= strong_penalty
                        else:
                            logits[0, repeated_token] *= strong_penalty
        
        probs = torch.softmax(logits / temp, dim=-1)

        if top_k > 0 and top_k < probs.size(-1):
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter(-1, topk_idx, topk_probs)
            probs = mask
            # 디버깅: top_k 필터링 후 top5
            top5_after_topk = probs.topk(5, dim=-1)
            logger.debug(f"After top_k={top_k}: top5_indices={top5_after_topk.indices[0].tolist()}, top5_probs={top5_after_topk.values[0].tolist()}")

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            # 디버깅: top_p 필터링 전 통계
            logger.debug(f"Before top_p={top_p}: sorted_probs[0][:5]={sorted_probs[0][:5].tolist()}, cumsum[0][:5]={cum[0][:5].tolist()}")
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = sorted_probs * keep
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, filtered)
            # 디버깅: top_p 필터링 후 top5
            top5_after_topp = probs.topk(5, dim=-1)
            logger.info(f"After top_p={top_p}: top5_indices={top5_after_topp.indices[0].tolist()}, top5_probs={[f'{v:.4f}' for v in top5_after_topp.values[0].tolist()]}, sampled_from={probs.nonzero().shape[0]} tokens")

        probs = probs / probs.sum(dim=-1, keepdim=True)
        token = torch.multinomial(probs, 1)
        # 디버깅: 샘플링된 토큰의 확률
        sampled_prob = probs[0, token.item()].item()
        logger.info(f"Sampled token {token.item()} with probability {sampled_prob:.4f}")
        return int(token.item())

    async def rpc_forward(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        """
        Process forward request: receive hidden states, run Stage, return logits.
        
        Expected request format:
        - request.tensors[0]: hidden states tensor [batch, seq_len, hidden_size]
        - request.metadata: dict with 'seq_len' (int) and 'cur_len' (int) for decode step
        """
        # Use timeout context manager
        async with timeout(self.request_timeout):
            try:
                # Deserialize input tensors
                if not request.tensors:
                    raise ValueError("No tensors in request")
                
                hidden_states = deserialize_torch_tensor(request.tensors[0])
                
                # Parse metadata
                metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                return self._run_forward(hidden_states, metadata)
                
            except Exception as e:
                logger.error(f"Error in rpc_forward: {e}", exc_info=True)
                raise

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        """Streaming version of rpc_forward."""
        async with timeout(self.request_timeout):
            try:
                tensor_parts = []
                metadata = None

                async for req in requests:
                    if metadata is None:
                        metadata = MSGPackSerializer.loads(req.metadata) if req.metadata else {}
                    tensor_parts.extend(req.tensors)

                if not tensor_parts:
                    raise ValueError("rpc_forward_stream received no tensors")
                if metadata is None:
                    raise ValueError("rpc_forward_stream missing metadata")

                async def tensor_iter():
                    for tensor in tensor_parts:
                        yield tensor

                tensors = await deserialize_tensor_stream(tensor_iter())
                if not tensors:
                    raise ValueError("Failed to deserialize tensors from stream")

                response = self._run_forward(tensors[0], metadata)
                yield response

            except Exception as e:
                logger.error(f"Error in rpc_forward_stream: {e}", exc_info=True)
                raise
