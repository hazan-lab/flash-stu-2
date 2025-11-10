import torch

from transformers import PretrainedConfig


class FlashSTUConfig(PretrainedConfig):
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
        window_size: int = 1024,
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
        softcap: float = 50.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        tie_word_embeddings: bool = True,
        stu_enable_mlp_sandwich: bool = False,
        stu_mlp_hidden_size: int = None,
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
        self.softcap = softcap
        self.stu_enable_mlp_sandwich = stu_enable_mlp_sandwich
        self.stu_mlp_hidden_size = stu_mlp_hidden_size
        if torch_dtype is None:
            self.torch_dtype = torch.float32
        elif isinstance(torch_dtype, str):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype
        self.mlp_scale = mlp_scale
