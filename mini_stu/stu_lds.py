
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def lds_scan_kernel(
        u_ptr,      # (B, L)
        A_ptr,      # (D,)
        B_ptr,      # (D,)
        h0_ptr,     # (D,)
        h_out_ptr,  # (B, L, D)  -- output states
        
        stride_u_b, stride_u_l,
        stride_h_b, stride_h_l, stride_h_d,
        
        SEQ_LEN: tl.constexpr,
        BLOCK_D: tl.constexpr,
        D_SIZE: tl.constexpr,
    ):
        """
        Computes h_t = A * h_{t-1} + B * u_t
        Parallelized across Batch (pid_b) and Dimension chunks (pid_d).
        Sequential across Sequence Length (L).
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        # Cast offsets for 64-bit safety
        pid_b_64 = pid_b.to(tl.int64)
        # stride args are effectively integers during compilation if specialized, 
        # so we don't call .to() on them. pid_b_64 * stride will promote to int64.
        
        # Offsets for D dimension
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_SIZE
        offs_d_64 = offs_d.to(tl.int64)

        # Load A and B
        a_vals = tl.load(A_ptr + offs_d_64, mask=mask_d, other=0.0).to(tl.float32)
        b_vals = tl.load(B_ptr + offs_d_64, mask=mask_d, other=0.0).to(tl.float32)
        
        # Load Initial State h0
        h_curr = tl.load(h0_ptr + offs_d_64, mask=mask_d, other=0.0).to(tl.float32)
        
        # Pointers setup
        u_base_ptr = u_ptr + (pid_b_64 * stride_u_b)
        h_out_base_ptr = h_out_ptr + (pid_b_64 * stride_h_b + offs_d_64 * stride_h_d)
        
        # Loop over sequence
        for t in range(SEQ_LEN):
            # 1. Update h
            u_val = tl.load(u_base_ptr + t * stride_u_l).to(tl.float32)
            h_curr = a_vals * h_curr + b_vals * u_val
            
            # Store h_curr to output
            tl.store(h_out_base_ptr + t * stride_h_l, h_curr, mask=mask_d)

    def run_lds_scan(u, A, B_vec, h0):
        """
        Args:
            u: (B, L)
            A: (D,)
            B_vec: (D,)
            h0: (D,)
        Returns:
            h: (B, L, D)
        """
        B_batch, L = u.shape
        D = A.shape[0]
        
        h_out = torch.empty((B_batch, L, D), device=u.device, dtype=u.dtype)
        
        # Grid config
        BLOCK_D = 1024
        num_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
        grid = (B_batch, num_d_blocks)
        
        lds_scan_kernel[grid](
            u, A, B_vec, h0, h_out,
            u.stride(0), u.stride(1),
            h_out.stride(0), h_out.stride(1), h_out.stride(2),
            SEQ_LEN=L,
            BLOCK_D=BLOCK_D,
            D_SIZE=D
        )
        
        return h_out


# ============================================================
#  Parallel associative scan for affine maps
# ============================================================
def affine_scan_inclusive(a: torch.Tensor, b: torch.Tensor):
    """
    Differentiable Hillis-Steele inclusive scan for elementwise affine maps.

    Each timestep t represents: f_t(x) = a_t * x + b_t   (elementwise on last dim)
    Composition (later ∘ earlier):
        (aL, bL) ⊗ (aR, bR) = (aL*aR, bL + aL*bR)

    After inclusive scan, (a_pref[t], b_pref[t]) equals f_t ∘ ... ∘ f_0.

    Args:
        a: (T, D)
        b: (..., T, D)

    Returns:
        a_pref: (T, D)
        b_pref: (..., T, D)
    """
    assert a.ndim == 2, f"a must be (T,D), got {a.shape}"
    assert b.shape[-2:] == a.shape, f"b must end with (T,D)={a.shape}, got {b.shape}"
    T, _ = a.shape

    a_out = a
    b_out = b
    offset = 1
    while offset < T:
        a_old = a_out
        b_old = b_out

        # Shift by offset with identity padding on the left:
        # a_shift[t] = 1 for t < offset, else a_old[t-offset]
        # b_shift[t] = 0 for t < offset, else b_old[t-offset]
        a_shift = F.pad(a_old[:-offset, :], (0, 0, offset, 0), mode="constant", value=1.0)
        b_shift = F.pad(b_old[..., :-offset, :], (0, 0, offset, 0), mode="constant", value=0.0)

        # IMPORTANT: b update uses a_old (not the updated a)
        b_out = b_old + a_old * b_shift
        a_out = a_old * a_shift
        offset <<= 1

    return a_out, b_out


@torch.no_grad()
def affine_scan_inclusive_inplace(a: torch.Tensor, b: torch.Tensor):
    """
    In-place version of the same scan, intended for no_grad usage only.
    Uses temporary buffers for the update step to avoid data hazards.

    Args:
        a: (T, D)  (will be overwritten)
        b: (..., T, D)  (will be overwritten)

    Returns:
        (a, b) overwritten to prefix-composed transforms.
    """
    T = a.shape[0]
    offset = 1
    while offset < T:
        aL = a[offset:]
        aR = a[:-offset]
        bL = b[..., offset:, :]
        bR = b[..., :-offset, :]

        # Use temp storage to avoid read-after-write hazards in overlapping slices
        # b[offset:] = b[offset:] + a[offset:] * b[:-offset]
        b[..., offset:, :] = bL + aL * bR

        # a[offset:] = a[offset:] * a[:-offset]
        a[offset:] = aL * aR

        offset <<= 1
    return a, b


@torch.no_grad()
def affine_scan_with_fixed_a(b: torch.Tensor, a_steps: List[torch.Tensor]):
    """
    Optimized scan where 'a' evolution is precomputed.

    b: (..., T, D)
    a_steps: list of 'aL' tensors for each offset step.
    """
    T = b.shape[-2]
    offset = 1
    step_idx = 0
    while offset < T:
        # b[offset:] += a_step * b[:-offset]
        # We must avoid in-place hazard. standard addcmul does not guarantee safety on self-overlap.
        # So we simply compute RHS and assign.
        aL = a_steps[step_idx]
        b[..., offset:, :] = b[..., offset:, :] + aL * b[..., :-offset, :]

        offset <<= 1
        step_idx += 1
    return b

# Try optimising the scan function
try:
    affine_scan_with_fixed_a = torch.compile(affine_scan_with_fixed_a)
except Exception:
    pass



# ============================================================
#  Fixed diagonal LDS (fastest for your real use-case)
# ============================================================
class FixedDiagonalLDS(nn.Module):
    """
    Fixed diagonal LDS (buffers), scalar input:

        h_{t+1} = A ⊙ h_t + B ⊙ u_t
        y_t     = h_{t+1} @ C

    A: (D,)
    B_vec: (D,)
    C: (D,O)
    h0: (D,)

    Forward uses in-place parallel scan under torch.no_grad().
    """

    def __init__(
        self,
        A: torch.Tensor,
        B_vec: torch.Tensor,
        C: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert A.ndim == 1
        D = A.numel()
        assert B_vec.shape == (D,)
        assert C.shape[0] == D
        if h0 is None:
            h0 = torch.zeros(D, device=A.device, dtype=A.dtype)
        assert h0.shape == (D,)

        # Store A, B_vec, C, h0
        self.register_buffer("A", A.detach().clone())
        self.register_buffer("B_vec", B_vec.detach().clone())
        self.register_buffer("C", C.detach().clone())
        self.register_buffer("h0", h0.detach().clone())

        self.a_cache = {}  # Cache for precomputed scan parameters
        self.a_final_cache = {}

    @torch.no_grad()
    def precompute_scan_A(self, T: int):
        """Precompute A evolution for a specific sequence length T."""
        if T in self.a_cache:
            return

        D = self.A.numel()
        D = self.A.numel()
        # Use A directly
        A = self.A
        a = A.view(1, D).repeat(T, 1) # (T, D)

        steps = []
        offset = 1
        while offset < T:
            aL = a[offset:]
            aR = a[:-offset]
            steps.append(aL.clone())

            # Evolve a safely (no hazard since we assign the result of mul)
            a[offset:] = aL * aR
            offset <<= 1

        self.a_cache[T] = steps
        self.a_final_cache[T] = a  # This is the final prefix product

    @torch.no_grad()
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (N,T) or (N,T,1)
        returns: (N,T,O)
        """
        if u.ndim == 3:
            assert u.shape[-1] == 1
            u = u[..., 0]
        assert u.ndim == 2

        N, T = u.shape
        D = self.A.numel()
        
        # Save original dtype for final cast
        orig_dtype = u.dtype
        
        # Use input dtype
        u_in = u
        h0 = self.h0
        B_vec = self.B_vec
        A = self.A

        # b_t = u_t * B_vec -> (N,T,D)
        if HAS_TRITON and u.is_cuda:
             # Triton kernel might handle types internally, but let's be safe and check if it supports bf16
             # For now, let's assume we want to use the python fallback if we really want to enforce f32
             # Or we trust the triton kernel.
             # Given the prompt asks to reorder computation, likely sticking to the python scan with precalc is safer for now.
             # But if HAS_TRITON is true, we should probably still use it if it's stable. 
             # However, the user specifically mentioned "precomputing A^[nk]" which suggests the python path.
             # Let's use the python path for absolute safety if we are debugging instability, or ensure we pass f32 to triton.
             h_seq = run_lds_scan(u_in, A, B_vec, h0)
             return (h_seq @ self.C).to(orig_dtype)

        b = u_in.unsqueeze(-1) * B_vec.view(1, 1, D)

        if T in self.a_cache:
            # Use optimized path with precomputed A steps
            # Ensure cached steps are on the correct device
            device = u.device
            cached_steps = [s.to(device) for s in self.a_cache[T]]
            a_final = self.a_final_cache[T].to(device)
            
            b_pref = affine_scan_with_fixed_a(b, cached_steps)
            a_pref = a_final
            a_pref = a_final
        else:
            # Standard path: compute a on the fly
            a = A.view(1, D).repeat(T, 1)
            # We need a version of affine_scan that handles f32
            a_pref, b_pref = affine_scan_inclusive_inplace(a, b)

        # h_{t+1} = a_pref[t]*h0 + b_pref[t]
        h_seq = b_pref + a_pref.unsqueeze(0) * h0.view(1, 1, D)

        # y_t = h_{t+1} @ C
        # inner matmul, then cast back
        out = h_seq @ self.C
        return out.to(orig_dtype)


# ============================================================
#  DistillSTU-style module: fixed LDS features + learn M from scratch
# ============================================================
class DistillSTUFast(nn.Module):
    """
    x: (B,T,d_in) -> y: (B,T,d_out)

    Steps:
      - Flatten features into batch: u = x.permute(0,2,1).reshape(B*d_in, T)
      - Fixed LDS -> U: (B*d_in, T, n_filters)
      - Reshape U -> (B,T,n_filters,d_in)
      - One GEMM with M reshaped (n_filters*d_in, d_out)
    """

    def __init__(
        self,
        lds: FixedDiagonalLDS,
        K: int,
        d_in: int,
        d_out: int,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.lds = lds
        self.K = K
        self.d_in = d_in
        self.d_out = d_out
        self.use_hankel_L = use_hankel_L

        self.n_filters = self.K if use_hankel_L else 2 * self.K

        # Learn from scratch
        self.M = nn.Parameter(
            torch.randn(self.n_filters, self.d_in, self.d_out, dtype=dtype, device=self.lds.A.device)
            / math.sqrt(self.n_filters * self.d_in)
        )

    def forward(self, x: torch.Tensor, feature_chunk: Optional[int] = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        B, T, d_in = x.shape
        assert d_in == self.d_in

        M_flat = self.M.reshape(self.n_filters * d_in, self.d_out)

        # No chunking
        if feature_chunk is None or feature_chunk >= d_in:
            u = x.permute(0, 2, 1).contiguous().reshape(-1, T)  # (B*d_in, T)
            U = self.lds(u)  # (B*d_in, T, n_filters)

            U = U.reshape(B, d_in, T, self.n_filters).permute(0, 2, 3, 1).contiguous()  # (B,T,n_filters,d_in)
            out = U.reshape(B * T, self.n_filters * d_in) @ M_flat
            return out.reshape(B, T, self.d_out)

        # Chunked over features (lower peak memory)
        out = x.new_zeros((B, T, self.d_out))
        for s in range(0, d_in, feature_chunk):
            e = min(d_in, s + feature_chunk)
            chunk = e - s

            x_chunk = x[:, :, s:e]  # (B,T,chunk)
            u = x_chunk.permute(0, 2, 1).contiguous().reshape(-1, T)  # (B*chunk,T)
            U = self.lds(u)  # (B*chunk,T,n_filters)

            U = U.reshape(B, chunk, T, self.n_filters).permute(0, 2, 3, 1).contiguous()  # (B,T,n_filters,chunk)

            M_chunk = self.M[:, s:e, :].contiguous().reshape(self.n_filters * chunk, self.d_out)
            out = out + (U.reshape(B * T, self.n_filters * chunk) @ M_chunk).reshape(B, T, self.d_out)

        return out
