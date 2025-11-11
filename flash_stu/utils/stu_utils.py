import numpy as np
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError:
    FlashFFTConv = None
    flash_fft_available = False

from flash_stu.utils.numerics import nearest_power_of_two


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z

def save_spectral_filters(
    phi: torch.Tensor,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save spectral filters to a file for reuse.
    
    Args:
        phi: Spectral filter tensor [seq_len, K]
        save_path: Path to save the filters (will add .pt extension if not present)
        metadata: Optional metadata dict (seq_len, K, use_hankel_L, etc.)
    
    Example:
        >>> phi = get_spectral_filters(128, 24)
        >>> save_spectral_filters(phi, "filters/custom_128_24.pt", 
        ...                       metadata={"seq_len": 128, "K": 24, "use_hankel_L": False})
    """
    save_path = Path(save_path)
    if save_path.suffix != '.pt':
        save_path = save_path.with_suffix('.pt')
    
    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU for saving
    phi_cpu = phi.cpu()
    
    # Prepare save dict
    save_dict = {
        'phi': phi_cpu,
        'shape': list(phi_cpu.shape),
        'dtype': str(phi_cpu.dtype),
    }
    
    # Add metadata if provided
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, save_path)
    print(f"Saved spectral filters to {save_path}")
    print(f"  Shape: {phi_cpu.shape}")
    if metadata:
        print(f"  Metadata: {metadata}")


def load_spectral_filters(
    load_path: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    verify_shape: Optional[tuple] = None
) -> torch.Tensor:
    """
    Load spectral filters from a file.
    
    Args:
        load_path: Path to the saved filters
        device: Device to load filters to (default: cuda if available)
        dtype: Data type to convert to (default: keep saved dtype)
        verify_shape: Optional shape tuple to verify (seq_len, K)
    
    Returns:
        Loaded spectral filter tensor
    
    Example:
        >>> phi = load_spectral_filters("filters/custom_128_24.pt")
        >>> phi = load_spectral_filters("filters/custom_128_24.pt", 
        ...                             device='cuda', dtype=torch.bfloat16,
        ...                             verify_shape=(128, 24))
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Filter file not found: {load_path}")
    
    # Load the saved dict
    # Use weights_only=False since we're loading trusted filter data (not model weights)
    save_dict = torch.load(load_path, map_location='cpu', weights_only=False)
    
    # Extract phi
    if isinstance(save_dict, dict):
        phi = save_dict['phi']
        metadata = save_dict.get('metadata', {})
        print(f"Loaded spectral filters from {load_path}")
        print(f"  Shape: {phi.shape}")
        if metadata:
            print(f"  Metadata: {metadata}")
    else:
        # Backward compatibility: if just tensor was saved
        phi = save_dict
        print(f"Loaded spectral filters from {load_path}")
        print(f"  Shape: {phi.shape}")
    
    # Verify shape if requested
    if verify_shape is not None:
        expected_shape = tuple(verify_shape)
        if phi.shape != expected_shape:
            raise ValueError(
                f"Filter shape mismatch: expected {expected_shape}, got {phi.shape}"
            )
    
    # Convert dtype if specified
    if dtype is not None:
        phi = phi.to(dtype=dtype)
    
    # Move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi = phi.to(device=device)
    
    return phi


def get_spectral_filters(
    seq_len: int, 
    K: int, 
    use_hankel_L: bool = False, 
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    load_from: Optional[str] = None,
) -> torch.Tensor:
    """
    Get spectral filters - either by computing or loading from file.
    
    Args:
        seq_len: Sequence length
        K: Number of eigenvalues/filters to keep
        use_hankel_L: Whether to use Hankel-L formulation
        device: Device to place filters on
        dtype: Data type for filters
        load_from: Optional path to load pre-computed filters from
    
    Returns:
        Spectral filter tensor [seq_len, K]
    
    Example:
        >>> # Compute filters
        >>> phi = get_spectral_filters(128, 24)
        >>> 
        >>> # Load pre-computed filters
        >>> phi = get_spectral_filters(128, 24, load_from="filters/custom_128_24.pt")
    """
    if load_from is not None:
        # Load from file
        return load_spectral_filters(
            load_from, 
            device=device, 
            dtype=dtype,
            verify_shape=(seq_len, K)
        )
    
    # Compute filters
    assert torch.cuda.is_available(), "CUDA is required."
    if device is None:
        device = torch.device('cuda')
    
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma ** 0.25
    return torch.tensor(phi, device=device, dtype=dtype)

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_approx: bool = True, return_both: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, seq_len, d_in = u.shape
    input_dtype = u.dtype

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    
    if return_both:
        # Compute both branches (for standard Hankel)
        if use_approx:
            _, d_out = v.shape
            v = v.view(1, -1, d_out, 1).to(torch.float32)
        else:
            _, K = v.shape
            sgn = sgn.unsqueeze(-1)
            v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, d_in, stack)
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        v = torch.fft.rfft(v, n=n, dim=1)
        U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
        U = torch.fft.rfft(U, n=n, dim=1)
        U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
        U_plus, U_minus = torch.unbind(U_conv, dim=-1)
        U_minus = U_minus * sgn
        return U_plus.to(input_dtype), U_minus.to(input_dtype)
    else:
        # Only compute plus branch (for Hankel-L)
        if use_approx:
            _, d_out = v.shape
            v = v.view(1, -1, d_out).to(torch.float32)  # No stack dimension
        else:
            _, K = v.shape
            v = v.view(1, -1, K, 1).to(torch.float32)  # (bsz, seq_len, K, d_in) - no stack dimension
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        v = torch.fft.rfft(v, n=n, dim=1)
        U = u.to(torch.float32)
        U = torch.fft.rfft(U, n=n, dim=1)
        U_plus = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
        return U_plus.to(input_dtype), None

def flash_convolve(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_approx: bool = True, return_both: bool = True
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    if return_both:
        # Compute both branches (for standard Hankel)
        sgn = torch.full((1, 1, padded_len), 1, device=u.device)
        sgn[:, :, 1::2] = -1

        if use_approx:
            u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
            u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
        else:
            u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
            u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

        U_conv = flash_fft(u_conv, v_padded)
        U_conv = U_conv[..., :seq_len]
        u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

        if use_approx:
            u_minus = u_minus * sgn[:, :, :seq_len]
            U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
        else:
            sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
            U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
            U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

        return U_plus, U_minus
    else:
        # Only compute plus branch (for Hankel-L)
        if use_approx:
            u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        else:
            u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
            v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()

        U_conv = flash_fft(u_padded, v_padded)
        U_conv = U_conv[..., :seq_len]

        if use_approx:
            U_plus = U_conv.transpose(1, 2)
        else:
            U_plus = U_conv.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()

        return U_plus, None
