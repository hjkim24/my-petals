"""
RPC handler for Stage1 server using hivemind ConnectionHandler.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

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

from partition import StageSegment, StageLast

logger = get_logger(__name__)


class Stage1ConnectionHandler(ConnectionHandler):
    """Connection handler for Stage1 that processes forward requests."""

    def __init__(
        self,
        dht: DHT,
        stage1_model,
        device: torch.device,
        request_timeout: float = 30.0,
        final_stage: bool = True,
    ):
        """
        Args:
            dht: DHT instance for peer discovery
            stage1_model: Stage1 model instance
            device: PyTorch device
            request_timeout: Timeout for RPC requests in seconds
            final_stage: If True, sample and return token; else return hidden states
        """
        # ConnectionHandler expects module_backends dict, but we only have one model
        # Create a dummy dict with a single entry
        module_backends = {"stage1": stage1_model}
        super().__init__(dht, module_backends, start=False)
        
        self.stage1_model = stage1_model
        self.device = device
        self.request_timeout = request_timeout
        self._kv_cache: Dict[str, Optional[tuple]] = {}
        self._default_temperature = 0.8
        self._default_top_p = 0.9
        self._default_top_k = 0
        self.final_stage = final_stage

    def _build_masks(
        self, seq_len: int, cur_len: int, is_prefill: bool, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create attention mask and position ids for prefill/decode."""
        if is_prefill:
            attn_mask = torch.ones(1, seq_len, device=self.device, dtype=hidden_states.dtype)
            pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
        else:
            attn_mask = torch.ones(1, cur_len, device=self.device, dtype=hidden_states.dtype)
            pos_ids = torch.arange(cur_len, device=self.device, dtype=torch.long).unsqueeze(0)
            pos_ids = pos_ids[:, -hidden_states.shape[1]:]
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

        if is_prefill:
            past_key_values = None
        else:
            past_key_values = self._kv_cache.get(session_id)
            if past_key_values is None:
                raise ValueError(f"Missing past_key_values for session_id={session_id}")

        attn_mask, pos_ids = self._build_masks(seq_len, cur_len, is_prefill, hidden_states)

        with torch.inference_mode():
            outputs, new_past = self.stage1_model(
                hidden_states.to(self.device),
                position_ids=pos_ids,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        self._kv_cache[session_id] = new_past

        if self.final_stage:
            logits = outputs
            next_token_id = int(self._sample_token(logits[:, -1, :], temperature, top_p, top_k))
            response_metadata = {"token_id": next_token_id, "session_id": session_id}
            token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
            serialized_token = serialize_torch_tensor(token_tensor.cpu())
            return runtime_pb2.ExpertResponse(
                tensors=[serialized_token],
                metadata=MSGPackSerializer.dumps(response_metadata),
            )
        else:
            hidden_out = outputs
            serialized_hidden = serialize_torch_tensor(hidden_out.cpu())
            response_metadata = {"session_id": session_id}
            return runtime_pb2.ExpertResponse(
                tensors=[serialized_hidden],
                metadata=MSGPackSerializer.dumps(response_metadata),
            )

    def _sample_token(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
        """Apply temperature / nucleus / top-k sampling to reduce repetition."""
        temp = max(temperature, 1e-5)
        probs = torch.softmax(logits / temp, dim=-1)

        if top_k > 0 and top_k < probs.size(-1):
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter(-1, topk_idx, topk_probs)
            probs = mask

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = sorted_probs * keep
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, filtered)

        probs = probs / probs.sum(dim=-1, keepdim=True)
        token = torch.multinomial(probs, 1)
        return int(token.item())

    async def rpc_forward(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        """
        Process forward request: receive hidden states, run Stage1, return logits.
        
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
