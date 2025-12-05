import torch
import torch.nn as nn

from .filters import get_spectral_filters
from .convolution import convolve
from .utils import nearest_power_of_two


class MiniSTU(nn.Module):
    """
    Simplified Spectral Transform Unit (STU) for learning linear dynamical systems.
    
    MiniSTU uses spectral filtering in the frequency domain to efficiently model
    long-range dependencies in sequential data. Unlike the full Flash STU model,
    this version focuses on the core spectral transform without attention layers
    or transformer components.
    
    The module transforms inputs through:
    1. Spectral convolution with learned filters
    2. Projection through learned matrices (Φ⁺, Φ⁻)
    3. Combination of positive and negative frequency branches
    4. Optional MLP for non-linear transformation
    
    Args:
        seq_len: Maximum sequence length
        num_filters: Number of spectral filters (K) to use
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        use_hankel_L: If True, use single-branch Hankel-L (faster).
                     If False, use two-branch standard Hankel (more expressive).
        use_mlp: If True, apply an MLP after the spectral transform.
        mlp_hidden_dim: Hidden dimension for the MLP (default: output_dim * 2).
        mlp_num_layers: Number of layers in the MLP (default: 2).
        mlp_dropout: Dropout rate for the MLP (default: 0.0).
        mlp_activation: Activation function for the MLP (default: 'relu').
        dtype: Data type for parameters
        device: Device to place module on
        precomputed_filters: Optional pre-computed spectral filters [seq_len, num_filters].
                            If provided, these are used instead of computing new ones.
    
    Shape:
        - Input: [batch_size, seq_len, input_dim] or [seq_len, input_dim]
        - Output: [batch_size, seq_len, output_dim]
    """
    
    def __init__(
        self,
        seq_len: int,
        num_filters: int,
        input_dim: int,
        output_dim: int,
        use_hankel_L: bool = False,
        use_mlp: bool = False,
        mlp_hidden_dim: int = None,
        mlp_num_layers: int = 2,
        mlp_dropout: float = 0.0,
        mlp_activation: str = 'relu',
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        precomputed_filters: torch.Tensor = None,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_hankel_L = use_hankel_L
        self.use_mlp = use_mlp
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        
        # Spectral filters: shape [seq_len, num_filters]
        # Register as buffer so it moves with .to(device)
        if precomputed_filters is not None:
            phi = precomputed_filters.to(self.device, self.dtype)
        else:
            phi = get_spectral_filters(
                seq_len, num_filters, use_hankel_L, self.device, self.dtype
            )
        self.register_buffer("phi", phi, persistent=False)
        
        # FFT length for convolution
        self.n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)
        
        # Learnable projection matrices
        # Φ⁺: projects convolved features (positive branch) [num_filters, input_dim, output_dim]
        scale = (num_filters * input_dim) ** -0.5  # Xavier-style initialization
        self.M_phi_plus = nn.Parameter(
            torch.randn(num_filters, input_dim, output_dim, dtype=dtype, device=self.device) * scale
        )
        
        # Φ⁻: projects convolved features (negative branch)
        # Only needed for standard Hankel (not Hankel-L)
        if not use_hankel_L:
            self.M_phi_minus = nn.Parameter(
                torch.randn(num_filters, input_dim, output_dim, dtype=dtype, device=self.device) * scale
            )
        
        # Optional MLP for non-linear transformation
        if use_mlp:
            if mlp_hidden_dim is None:
                mlp_hidden_dim = output_dim * 2
            
            # Build MLP layers
            mlp_layers = []
            mlp_input_dim = output_dim  # MLP operates on the output of spectral transform
            
            for i in range(mlp_num_layers):
                if mlp_num_layers == 1:
                    # Single layer: output_dim -> output_dim (just a linear projection)
                    mlp_layers.append(nn.Linear(mlp_input_dim, output_dim, dtype=dtype, device=self.device))
                    break  # No activation for single layer
                elif i == 0:
                    # First layer: output_dim -> mlp_hidden_dim
                    mlp_layers.append(nn.Linear(mlp_input_dim, mlp_hidden_dim, dtype=dtype, device=self.device))
                elif i == mlp_num_layers - 1:
                    # Last layer: mlp_hidden_dim -> output_dim
                    mlp_layers.append(nn.Linear(mlp_hidden_dim, output_dim, dtype=dtype, device=self.device))
                else:
                    # Middle layers: mlp_hidden_dim -> mlp_hidden_dim
                    mlp_layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, dtype=dtype, device=self.device))
                
                # Add activation (except for last layer)
                if i < mlp_num_layers - 1:
                    if mlp_activation.lower() == 'relu':
                        mlp_layers.append(nn.ReLU())
                    elif mlp_activation.lower() == 'gelu':
                        mlp_layers.append(nn.GELU())
                    elif mlp_activation.lower() == 'tanh':
                        mlp_layers.append(nn.Tanh())
                    else:
                        raise ValueError(f"Unsupported activation: {mlp_activation}")
                    
                    # Add dropout if specified
                    if mlp_dropout > 0.0:
                        mlp_layers.append(nn.Dropout(mlp_dropout))
            
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MiniSTU.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        # Handle unbatched input
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        assert x.dim() == 3, f"Expected x with shape [B, L, I] or [L, I]; got {tuple(x.shape)}"
        B, L, I = x.shape
        
        # Ensure correct dtype
        x = x.to(self.M_phi_plus.dtype)
        
        # Spectral convolution: convolve input with spectral filters
        # U_plus, U_minus: [B, L, K, I]
        U_plus, U_minus = convolve(x, self.phi, self.n, use_approx=False)
        
        # Project convolved features through learned matrices
        # [B, L, K, I] ⊗ [K, I, O] -> [B, L, O]
        spectral_plus = torch.einsum('blki,kio->blo', U_plus, self.M_phi_plus)
        
        if self.use_hankel_L:
            # Single branch (Hankel-L)
            output = spectral_plus
        else:
            # Two branches (standard Hankel)
            spectral_minus = torch.einsum('blki,kio->blo', U_minus, self.M_phi_minus)
            output = spectral_plus + spectral_minus
        
        # Apply optional MLP
        if self.mlp is not None:
            output = self.mlp(output)
        
        return output
    
    def get_num_params(self) -> int:
        """Return the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

