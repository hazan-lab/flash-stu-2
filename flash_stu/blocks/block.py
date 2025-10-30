import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Any

from flash_stu.config import FlashSTUConfig
from flash_stu.layers.stu_layer import STULayer
from flash_stu.layers.attention_layer import AttentionLayer
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.utils.numerics import nearest_power_of_two

class FlashSTUBlock(nn.Module):
    """
    A single hybrid block combining STU and/or Attention layers.
    
    Args:
        d_model: Hidden dimension (replaces n_embd) 
        sequence_length: Max sequence length (replaces seq_len)
        num_filters: Number of spectral filters (replaces num_eigh)
        attention_window: Window size for local attention
        use_flash_fft: Whether to use FlashFFT
        use_attention: Whether to include attention in this block
        attention_config: Dict with attention-specific settings (n_heads, softcap, etc.)
        stu_enable_mlp_sandwich: Whether to wrap STU with MLP projections (up-project -> activate -> STU -> down-project)
        stu_mlp_hidden_size: Hidden size for sandwich MLP. Defaults to intermediate_size if not set
        **kwargs: Other config options (dropout, bias, etc.)
    """
    def __init__(
        self,

        # mode 1: from config
        config: Optional[FlashSTUConfig] = None,
        phi: Optional[torch.Tensor] = None,
        n: Optional[int] = None,

        # mode 2: from parameters
        d_model: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_filters: Optional[int] = None,
        attention_window: Optional[int] = None,
        use_attention: bool = False,
        use_flash_fft: bool = True,
        attention_config: Optional[Dict[str, Any]] = None,
        
        # optional params
        n_heads: Optional[int] = None,
        mlp_scale: int = 4,
        dropout: float = 0.0, 
        bias: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        stu_enable_mlp_sandwich: bool = False,
        stu_mlp_hidden_size: Optional[int] = None,
        **kwargs,
    ):    
        super(FlashSTUBlock, self).__init__()

        # determine mode
        if config is not None:
            # mode 1: use provided config
            self.config = config
            phi_temp = phi
            n_temp = n
        else:
            # mode 2: build config from parameters 

            # Validation
            if d_model is None:
                raise ValueError("d_model is required when config is not provided")
            if sequence_length is None:
                raise ValueError("sequence_length is required when config is not provided")
            
            # Build config dict with user parameters
            config_dict = {
                'n_embd': d_model,
                'seq_len': sequence_length,
                'num_eigh': num_filters if num_filters else 24,
                'window_size': attention_window if attention_window else sequence_length // 4,
                'use_flash_fft': use_flash_fft,
                'n_heads': n_heads if n_heads else max(1, d_model // 64),
                'mlp_scale': mlp_scale,
                'dropout': dropout,
                'bias': bias,
                'torch_dtype': torch_dtype,
                'use_attn': use_attention,
                'use_approx': kwargs.get('use_approx', True),
                'use_hankel_L': kwargs.get('use_hankel_L', False),
                'softcap': 50.0,
                'n_layers': 1,
                'bsz': 1,
                'vocab_size': 1,
                'stu_enable_mlp_sandwich': stu_enable_mlp_sandwich,
                'stu_mlp_hidden_size': stu_mlp_hidden_size,
            }
            
            # Override with attention_config if provided
            if attention_config:
                for key, value in attention_config.items():
                    if key in config_dict:
                        config_dict[key] = value
                    
            self.config = FlashSTUConfig(**config_dict)
            phi_temp = None
            n_temp = None
        
        # compute phi and n if not provided
        if phi_temp is None:
            phi_tensor = get_spectral_filters(self.config.num_eigh, self.config.seq_len)
            self.register_buffer('phi', phi_tensor, persistent=True)
        else:
            self.register_buffer('phi', phi_temp, persistent=True)
        if n_temp is None:
            self.n = nearest_power_of_two(self.config.seq_len * 2 - 1, round_up=True)
        else:
            self.n = n_temp

        # create the layer
        if use_attention or self.config.use_attn:
            self.layer = AttentionLayer(self.config)
        else:
            self.layer = STULayer(self.config, self.phi, self.n)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block. 

        Args: 
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns: 
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        return self.layer(x)

    @property
    def is_attention_block(self) -> bool:
        """Returns True if this block uses attention, False if STU."""
        return isinstance(self.layer, AttentionLayer)

    def get_num_params(self) -> int:
        """Return number of parameters in this block."""
        return sum(p.numel() for p in self.parameters())