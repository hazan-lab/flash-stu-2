
import torch
import triton
import triton.language as tl

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
        
        # Store h_curr to output (cast back to original dtype implicitly or explicitly)
        # We need to know the output dtype. tl.store will cast if needed? 
        # Actually it's better to cast explicitly to matching ptr type or let Triton handle it.
        # But we don't know the ptr type inside the kernel easily without extra args or assume it matches.
        # As long as h_out_ptr is a pointer to the correct type, tl.store should work.
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
    # We parallelize over batch and D blocks.
    # D is typically small (e.g. 862 ~ 1024). We can probably cover D with one or few blocks.
    BLOCK_D = 1024 # Tuneable
    num_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
    
    grid = (B_batch, num_d_blocks)
    
    # Check contiguous for strides
    # u = u.contiguous() # Ensure in caller
    
    lds_scan_kernel[grid](
        u, A, B_vec, h0, h_out,
        u.stride(0), u.stride(1),
        h_out.stride(0), h_out.stride(1), h_out.stride(2),
        SEQ_LEN=L,
        BLOCK_D=BLOCK_D,
        D_SIZE=D
    )
    
    return h_out
