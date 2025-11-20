import torch
from typing import Union, Tuple

from transformers import PretrainedConfig


class FlashSTUConfig(PretrainedConfig):
    """
    Configuration class for Flash STU model.
    
    Flash STU is a hybrid architecture that interleaves spectral state space model (STU) layers 
    with sliding window attention layers for efficient sequence modeling.
    
    Args:
        bsz (int, optional): Batch size. Defaults to 1.
        n_embd (int, optional): Embedding/hidden dimension. Defaults to 1536.
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        n_layers (int, optional): Total number of layers (STU + Attention). Defaults to 26.
        seq_len (int, optional): Maximum sequence length. Defaults to 8192.
        window_size (Union[int, Tuple[int, int]], optional): Sliding window size for attention.
            Can be int (symmetric) or tuple (left, right). Defaults to 1024.
        vocab_size (int, optional): Vocabulary size. Defaults to 200064.
        mlp_scale (int, optional): MLP hidden dimension multiplier (hidden_dim = n_embd * mlp_scale). 
            Defaults to 12.
        bias (bool, optional): Whether to use bias in linear layers. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        num_eigh (int, optional): Number of spectral filters (eigenvalues) for STU. Defaults to 24.
        use_hankel_L (bool, optional): Whether to use Hankel-L formulation (single branch). 
            Defaults to False.
        use_flash_fft (bool, optional): Whether to use Flash FFT for convolutions. Defaults to True.
        use_approx (bool, optional): Whether to use approximation mode for STU (project then convolve).
            When True, uses ~50x fewer parameters per STU layer. Recommended for scalability. 
            Defaults to True.
        use_attn (bool, optional): Whether to use attention layers (if False, STU-only). 
            Defaults to True.
        use_cache (bool, optional): Whether to enable KV caching for generation. Defaults to True.
        use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. 
            Defaults to False.
        softcap (float, optional): Softcap value for attention logits. Defaults to 50.0.
        torch_dtype (torch.dtype, optional): Data type for model parameters. Defaults to torch.bfloat16.
        tie_word_embeddings (bool, optional): Whether to tie input/output embeddings. Defaults to True.
        stu_enable_mlp_sandwich (bool, optional): Whether to enable MLP sandwiching for STU layers. 
            Defaults to False.
        stu_mlp_hidden_size (int, optional): Hidden dimension for STU sandwiching MLP. 
            If None, uses n_embd. Defaults to None.
        filter_path (str, optional): Path to pre-computed spectral filters. Defaults to None.
        d_in_tile (int, optional): Input dimension tile size for memory optimization. Defaults to None.
        d_out_tile (int, optional): Output dimension tile size for memory optimization. Defaults to None.
    
    Example:
        >>> from flash_stu import FlashSTUConfig, FlashSTU
        >>> 
        >>> # Small model configuration
        >>> config = FlashSTUConfig(
        ...     n_embd=512,
        ...     n_layers=12,
        ...     n_heads=8,
        ...     seq_len=2048,
        ... )
        >>> model = FlashSTU(config)
    """
    model_type = "FlashSTU"
    
    def to_dict(self):
        """Override to_dict to properly serialize torch_dtype."""
        output = super().to_dict()
        if hasattr(self, 'torch_dtype') and isinstance(self.torch_dtype, torch.dtype):
            output['torch_dtype'] = str(self.torch_dtype).replace('torch.', '')
        return output

    def __init__(
        self,
        bsz: int = 1,
        n_embd: int = 1536,
        n_heads: int = 8,
        n_layers: int = 26,
        seq_len: int = 8192,
        window_size: Union[int, Tuple[int, int]] = 1024,
        vocab_size: int = 200064,
        mlp_scale: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_flash_fft: bool = True,
        use_approx: bool = True,
        use_attn: bool = True,
        use_cache: bool = True,
        use_gradient_checkpointing: bool = False,
        softcap: float = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        tie_word_embeddings: bool = True,
        stu_enable_mlp_sandwich: bool = False,
        stu_mlp_hidden_size: int = None,
        filter_path: str = None,  # Path to load pre-computed spectral filters
        d_in_tile: int = None,
        d_out_tile: int = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.bsz = bsz
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.intermediate_size = n_embd * mlp_scale
        self.hidden_act = "swish"
        self.bias = bias
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.use_attn = use_attn
        self.use_cache = use_cache
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.softcap = softcap
        self.stu_enable_mlp_sandwich = stu_enable_mlp_sandwich
        self.stu_mlp_hidden_size = stu_mlp_hidden_size
        self.filter_path = filter_path
        self.d_in_tile = d_in_tile
        self.d_out_tile = d_out_tile
        if torch_dtype is None:
            self.torch_dtype = torch.float32
        elif isinstance(torch_dtype, str):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype
        self.mlp_scale = mlp_scale
