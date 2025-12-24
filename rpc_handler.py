"""
RPC handler for Stage1 server using hivemind ConnectionHandler.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional

import torch
from hivemind import DHT, MSGPackSerializer, P2PContext, deserialize_torch_tensor, serialize_torch_tensor

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

from partition import Stage1

logger = get_logger(__name__)


class Stage1ConnectionHandler(ConnectionHandler):
    """Connection handler for Stage1 that processes forward requests."""

    def __init__(
        self,
        dht: DHT,
        stage1_model: Stage1,
        device: torch.device,
        request_timeout: float = 30.0,
    ):
        """
        Args:
            dht: DHT instance for peer discovery
            stage1_model: Stage1 model instance
            device: PyTorch device
            request_timeout: Timeout for RPC requests in seconds
        """
        # ConnectionHandler expects module_backends dict, but we only have one model
        # Create a dummy dict with a single entry
        module_backends = {"stage1": stage1_model}
        super().__init__(dht, module_backends)
        
        self.stage1_model = stage1_model
        self.device = device
        self.request_timeout = request_timeout

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
                hidden_states = hidden_states.to(self.device)
                
                # Parse metadata
                metadata = MSGPackSerializer.loads(request.metadata) if request.metadata else {}
                seq_len = metadata.get("seq_len", hidden_states.shape[1])
                cur_len = metadata.get("cur_len", seq_len)
                is_prefill = metadata.get("is_prefill", False)
                
                logger.debug(f"rpc_forward: seq_len={seq_len}, cur_len={cur_len}, is_prefill={is_prefill}")
                
                # Build attention mask and position ids
                if is_prefill:
                    # Prefill: process full sequence
                    attn_mask = torch.ones(1, seq_len, device=self.device, dtype=torch.long)
                    pos_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
                    past_key_values = None
                else:
                    # Decode: process only last token
                    attn_mask = torch.ones(1, cur_len, device=self.device, dtype=torch.long)
                    pos_ids = torch.arange(cur_len, device=self.device, dtype=torch.long).unsqueeze(0)
                    # For decode, we only use the last position
                    pos_ids = pos_ids[:, -1:]
                    # past_key_values should be maintained by the client, but for simplicity
                    # we'll handle it here. In a real implementation, this should be cached.
                    past_key_values = None
                
                # Run Stage1 forward
                with torch.inference_mode():
                    logits, past_key_values = self.stage1_model(
                        hidden_states,
                        position_ids=pos_ids,
                        attention_mask=attn_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                
                # Get the token ID from the last position
                next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                
                # Serialize response
                # We return the token ID as metadata and logits (or just token ID)
                # For simplicity, we'll return token ID in metadata
                response_metadata = {"token_id": next_token_id}
                
                # Create response with token ID as a tensor
                token_tensor = torch.tensor([[next_token_id]], device=self.device, dtype=torch.long)
                serialized_token = serialize_torch_tensor(token_tensor.cpu(), compression=None)
                
                return runtime_pb2.ExpertResponse(
                    tensors=[serialized_token],
                    metadata=MSGPackSerializer.dumps(response_metadata),
                )
                
            except Exception as e:
                logger.error(f"Error in rpc_forward: {e}", exc_info=True)
                raise

