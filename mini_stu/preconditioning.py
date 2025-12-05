# mini_stu/preconditioning.py

import torch
import math


def chebyshev_monic_coeffs(
    degree: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute coefficients c of the monic Chebyshev polynomial M_n of degree `degree`
    on [-1, 1], expressed in the "reversed" basis used for sequence preconditioning:
    
        M_n(x) = sum_{i=0}^degree c[i] * x^{degree - i},
    
    so c has length degree+1 and c[0] = 1 (monic).

    This matches the polynomial structure used in the USP paper:
        p_n(x) = sum_{i=0}^n c_i x^{n-i}.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    
    device = device or torch.device("cpu")
    
    # n = 0: constant 1
    if degree == 0:
        return torch.ones(1, device=device, dtype=dtype)
    
    # Represent polynomials as lists of coefficients in increasing degree:
    # coeffs[k] = coefficient for x^k.
    T_prev = [1.0]       # T_0(x) = 1
    T_curr = [0.0, 1.0]  # T_1(x) = x
    
    if degree == 1:
        T_n = T_curr
    else:
        for n in range(1, degree):
            # Chebyshev recurrence:
            #   T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
            # Compute 2x * T_curr:
            two_x_T = [0.0] + [2.0 * c for c in T_curr]  # shift by 1 degree
            # Pad T_prev to same length
            if len(T_prev) < len(two_x_T):
                T_prev_padded = T_prev + [0.0] * (len(two_x_T) - len(T_prev))
            else:
                T_prev_padded = T_prev
            T_next = [a - b for a, b in zip(two_x_T, T_prev_padded)]
            T_prev, T_curr = T_curr, T_next
        T_n = T_curr
    
    # For n >= 1, the leading coefficient of T_n is 2^{n-1}.
    leading = T_n[-1]
    monic = [c / leading for c in T_n]  # scale so leading coefficient is 1
    
    # monic[k] = coefficient of x^k, k = 0..degree
    # We want c[i] such that:
    #   M_n(x) = sum_{i=0}^degree c[i] x^{degree - i}
    # So c[i] = monic[degree - i].
    c = [monic[degree - i] for i in range(degree + 1)]
    return torch.tensor(c, device=device, dtype=dtype)


def precondition_sequence(
    y: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Apply causal polynomial preconditioning to a sequence.
    
    Given coefficients c of length n+1 (with c[0] = 1), we define
    
        y_tilde[t] = sum_{i=0}^n c[i] * y[t - i],
    
    with zero-padding for t - i < 0.
    
    Args:
        y: Tensor of shape [batch_size, seq_len, d] or [seq_len, d].
        c: Tensor of shape [n+1] (c[0] = 1).
    
    Returns:
        y_tilde: same shape as y.
    """
    if y.dim() == 2:
        # [L, d] -> add batch dim
        y = y.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, L, D = y.shape
    n = c.shape[0] - 1
    if n < 0:
        raise ValueError("c must have length at least 1")
    
    if n == 0:
        # Just scale by c[0]
        y_tilde = y * c[0]
    else:
        # Zero-pad on the time dimension at the front by n steps.
        # pad format: (pad_last_dim_left, pad_last_dim_right, pad_L_left, pad_L_right)
        pad = (0, 0, n, 0)
        y_pad = torch.nn.functional.pad(y, pad)  # [B, L + n, D]
        
        # Compute y_tilde via a small causal convolution
        y_tilde = torch.zeros_like(y)
        for i in range(n + 1):
            # For each i, we want y[t - i], which corresponds to y_pad[:, n-i + t].
            y_shift = y_pad[:, n - i : n - i + L, :]
            y_tilde = y_tilde + c[i] * y_shift
    
    if squeeze_batch:
        y_tilde = y_tilde.squeeze(0)
    
    return y_tilde
