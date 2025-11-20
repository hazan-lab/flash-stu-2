import torch


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    """
    Generate Hankel matrix for spectral filter computation.
    
    The Hankel matrix is a symmetric matrix used to capture the structure of
    linear dynamical systems in the frequency domain.
    
    Args:
        seq_len: Sequence length (size of Hankel matrix)
        use_hankel_L: If True, use Hankel-L formulation (single branch, faster).
                     If False, use standard Hankel (two branches, more expressive).
    
    Returns:
        Hankel matrix of shape [seq_len, seq_len]
    """
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    
    if use_hankel_L:
        # Hankel-L: single branch formulation
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        # Standard Hankel: two branch formulation
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    
    return Z


def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate spectral filters using Hankel matrix eigendecomposition.
    
    This computes the top-K eigenvectors of the Hankel matrix, scaled by the
    fourth root of their eigenvalues, which gives optimal spectral filters
    for approximating linear dynamical systems.
    
    Args:
        seq_len: Sequence length
        K: Number of spectral filters (eigenvalues) to keep
        use_hankel_L: Whether to use Hankel-L formulation (single branch)
        device: Device to place filters on
        dtype: Data type for filters
        
    Returns:
        Spectral filters of shape [seq_len, K]
        
    Note:
        This computation can be slow for large sequence lengths. Consider
        pre-computing and saving filters for production use.
    """
    device = device or torch.device('cpu')
    
    # Generate Hankel matrix
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    
    # Compute eigendecomposition
    sigma, phi = torch.linalg.eigh(Z)
    
    # Take top-K eigenvalues and eigenvectors
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    
    # Scale eigenvectors by fourth root of eigenvalues
    phi_k *= sigma_k ** 0.25
    
    return phi_k.to(dtype=dtype)

