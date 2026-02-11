from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError:
    FlashFFTConv = None
    flash_fft_available = False


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
            v_fft = torch.fft.rfft(v.view(1, -1, d_out, 1).to(torch.float32).contiguous(), n=n, dim=1)
    else:
        # Standard mode
        sgn = sgn.unsqueeze(-1) if sgn.dim() == 3 else sgn
        if v_fft is None:
            v_fft = torch.fft.rfft(v.view(1, -1, K, 1, 1).to(torch.float32).contiguous(), n=n, dim=1)

    # Expand u to match filter dimensions: [bsz, seq_len, K, d_in]
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    # Create positive and negative frequency versions of input
    # FFT requires float32 (bf16 not supported by torch.fft)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()

    # FFT of input
    U = torch.fft.rfft(U, n=n, dim=1)

    # Multiply in frequency domain (convolution in time domain)
    U_conv = torch.fft.irfft(v_fft * U, n=n, dim=1)[:, :seq_len]

    # Separate positive and negative branches
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)

    # Apply sign alternation to negative branch
    U_minus = U_minus * sgn

    return U_plus.to(dtype), U_minus.to(dtype)


def flash_convolve(
    u: torch.Tensor,
    v: torch.Tensor,
    flash_fft: "FlashFFTConv",
    sgn: Optional[torch.Tensor] = None,
    return_both: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convolve input with spectral filters using FlashFFTConv.

    Follows the same contract as convolve() but uses the optimized
    FlashFFTConv CUDA kernel (monarch decomposition) instead of
    torch.fft.  Operates in standard (non-approx) mode matching MiniSTU.

    Args:
        u: Input tensor [batch_size, seq_len, d_in]
        v: Spectral filters [seq_len, K]
        flash_fft: Pre-instantiated FlashFFTConv object
        sgn: Optional pre-computed sign tensor [1, 1, padded_len] (channel-last layout).
             If None, created fresh.
        return_both: If True return (U_plus, U_minus); if False return (U_plus, None).

    Returns:
        (U_plus, U_minus) with shapes [batch_size, seq_len, K, d_in]
        U_minus is None when return_both=False.
    """
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    from .utils import nearest_power_of_two
    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    # FlashFFTConv expects layout: (batch, channels, length)
    # We need to convolve each (d_in) channel with each of K filters → expand to K*d_in channels
    # u: [B, L, d_in] → pad → [B, d_in, padded_len] → repeat_interleave K → [B, K*d_in, padded_len]
    # v: [L, K] → pad → [K, padded_len] → repeat d_in → [K*d_in, padded_len]
    u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
    v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()

    if return_both:
        if sgn is None:
            sgn = torch.full((1, 1, padded_len), 1, device=u.device)
            sgn[:, :, 1::2] = -1

        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)
        U_conv = flash_fft(u_conv, v_padded)
        U_conv = U_conv[..., :seq_len]

        u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

        sgn_trunc = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)  # [1, seq_len, 1, 1]
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn_trunc

        return U_plus, U_minus
    else:
        U_conv = flash_fft(u_padded, v_padded)
        U_conv = U_conv[..., :seq_len]
        U_plus = U_conv.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        return U_plus, None

