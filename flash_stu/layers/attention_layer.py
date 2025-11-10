import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from flash_stu.modules.attention import Attention
from flash_stu.modules.swiglu import MLP

try:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP as TritonMLP
    triton_mlp = True
except ImportError as e:
    print(
        f"Unable to import Triton-based MLP: {e}. Falling back to vanilla SwiGLU MLP instead."
    )
    triton_mlp = False

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError as e:
    print(
        f"Unable to import Triton-based RMSNorm: {e}. Falling back to PyTorch implementation."
    )
    from torch.nn import RMSNorm

    triton_norm = False


class AttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super(AttentionLayer, self).__init__()
        self.config = config
        self.attn_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.attn = Attention(config)
        self.mlp_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.mlp = MLP(config, dtype=config.torch_dtype)

        # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for MLP
        self.attn_norm = self.attn_norm.to(dtype=config.torch_dtype)
        self.mlp_norm = self.mlp_norm.to(dtype=config.torch_dtype)

    def _forward_impl(self, x: torch.Tensor, past_key_value=None, use_cache=False):
        """Internal forward implementation (for checkpointing)."""
        # Attention with residual connection
        attn_input = self.attn_norm(x).to(x.dtype)
        if use_cache:
            attn_output, present_key_value = self.attn(attn_input, past_key_value=past_key_value, use_cache=True)
            x = x + attn_output
        else:
            x = x + self.attn(attn_input, past_key_value=past_key_value, use_cache=False)
            present_key_value = None
        
        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x).to(x.dtype))
        
        if use_cache:
            return x, present_key_value
        return x

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache=True):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            past_key_value: Optional past key/value for attention layer
            use_cache: Whether to return key/value for caching
        
        Returns:
            If use_cache=False: output tensor
            If use_cache=True: (output tensor, present_key_value)
        """
        # Note: Gradient checkpointing is incompatible with KV caching
        # because caching requires returning intermediate states
        if self.config.use_gradient_checkpointing and self.training and not use_cache:
            # Can only checkpoint when not using cache
            return checkpoint(self._forward_impl, x, past_key_value, use_cache, use_reentrant=False)
        else:
            return self._forward_impl(x, past_key_value, use_cache)
