from typing import Optional

import torch


def convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    n: int,
    use_approx: bool = False,
    sgn: Optional[torch.Tensor] = None,
    v_fft: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convolve input with spectral filters using FFT.

    This implements the core spectral convolution operation that transforms
    the input through the frequency domain using the spectral filters.

    Args:
        u: Input tensor of shape [batch_size, seq_len, d_in]
        v: Spectral filters of shape [seq_len, K] for standard mode
        n: FFT length (typically nearest power of 2 >= 2*seq_len - 1)
        use_approx: If True, use approximation mode (not typically used in MiniSTU)
        sgn: Optional pre-computed sign alternation tensor. If None, created fresh.
        v_fft: Optional pre-computed FFT of reshaped filters. If None, computed fresh.

    Returns:
        Tuple of (U_plus, U_minus):
            - U_plus: Positive branch convolution [batch_size, seq_len, K, d_in]
            - U_minus: Negative branch convolution [batch_size, seq_len, K, d_in]

    Note:
        The two branches (plus/minus) correspond to the positive and negative
        frequency components in the spectral representation.
    """
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    # Preserve input dtype for all operations
    dtype = u.dtype

    # Create sign alternation pattern for negative frequencies
    if sgn is None:
        sgn = torch.full((1, seq_len, 1), 1, device=u.device, dtype=dtype)
        sgn[:, 1::2] *= -1

    if use_approx:
        # Approximation mode (less common)
        _, d_out = v.shape
        if v_fft is None:
            v_fft = torch.fft.rfft(v.view(1, -1, d_out, 1).to(dtype=dtype).contiguous(), n=n, dim=1)
    else:
        # Standard mode
        sgn = sgn.unsqueeze(-1) if sgn.dim() == 3 else sgn
        if v_fft is None:
            v_fft = torch.fft.rfft(v.view(1, -1, K, 1, 1).to(dtype=dtype).contiguous(), n=n, dim=1)

    # Expand u to match filter dimensions: [bsz, seq_len, K, d_in]
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    # Create positive and negative frequency versions of input
    U = torch.stack([u, u * sgn], dim=-1).to(dtype=dtype).contiguous()

    # FFT of input
    U = torch.fft.rfft(U, n=n, dim=1)

    # Multiply in frequency domain (convolution in time domain)
    U_conv = torch.fft.irfft(v_fft * U, n=n, dim=1)[:, :seq_len]

    # Separate positive and negative branches
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)

    # Apply sign alternation to negative branch
    U_minus = U_minus * sgn

    return U_plus, U_minus

