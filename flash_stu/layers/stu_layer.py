import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from flash_stu.modules.stu import STU
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


class STULayer(nn.Module):
    def __init__(self, config, phi, n):
        super(STULayer, self).__init__()
        self.config = config
        self.stu_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        
        # STU sandwich MLP (optional)
        self.stu_mlp_enabled = config.stu_enable_mlp_sandwich
        if self.stu_mlp_enabled:
            self.stu_mlp_hidden_size = (
                config.stu_mlp_hidden_size
                if config.stu_mlp_hidden_size is not None
                else config.intermediate_size
            )
            # Up-project to hidden size
            self.stu_mlp_in_proj = nn.Linear(
                config.n_embd,
                self.stu_mlp_hidden_size,
                bias=config.bias,
                dtype=config.torch_dtype,
            )
            # Activation (SiLU/Swish)
            self.stu_mlp_act = nn.SiLU()
            # STU operates on the activated dimension
            self.stu = STU(config, phi, n, feature_dim=self.stu_mlp_hidden_size)
            # Down-project back to d_model
            self.stu_mlp_out_proj = nn.Linear(
                self.stu_mlp_hidden_size,
                config.n_embd,
                bias=config.bias,
                dtype=config.torch_dtype,
            )
        else:
            self.stu = STU(config, phi, n)
        
        self.mlp_norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        self.mlp = MLP(config, dtype=config.torch_dtype)

        # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for MLP
        self.stu_norm = self.stu_norm.to(dtype=config.torch_dtype)
        self.mlp_norm = self.mlp_norm.to(dtype=config.torch_dtype)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation (for checkpointing)."""
        # STU path with optional sandwiching
        h = self.stu_norm(x).to(x.dtype)
        
        if self.stu_mlp_enabled:
            # Apply sandwich MLP: up-project -> activate -> STU -> down-project
            h = self.stu_mlp_in_proj(h)
            h = self.stu_mlp_act(h)
            h = self.stu(h)
            h = self.stu_mlp_out_proj(h)
        else:
            # Standard STU without sandwiching
            h = self.stu(h)
        
        x = x + h
        
        # MLP path
        x = x + self.mlp(self.mlp_norm(x).to(x.dtype))
        return x
    
    def forward(self, x: torch.Tensor, past_key_value=None, use_cache=False):
        """
        Forward pass. STU layers don't use KV cache, so past_key_value is ignored.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            past_key_value: Ignored (for interface compatibility with AttentionLayer)
            use_cache: Ignored (for interface compatibility with AttentionLayer)
        
        Returns:
            If use_cache=False: output tensor
            If use_cache=True: (output tensor, None)
        """
        # Use gradient checkpointing if enabled and training
        if self.config.use_gradient_checkpointing and self.training:
            x = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x = self._forward_impl(x)
        
        if use_cache:
            return x, None  # since STU layers don't have KV cache
        return x