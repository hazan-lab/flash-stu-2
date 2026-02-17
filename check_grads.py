
import torch
from mini_stu.mamba_kernels import run_lds_scan

def check_grads():
    device = torch.device("cuda")
    dtype = torch.float32 # Use FP32 for simplicity of check
    
    B, L, D = 2, 16, 32
    u = torch.randn(B, L, device=device, dtype=dtype, requires_grad=True)
    A = torch.randn(D, device=device, dtype=dtype)
    B_vec = torch.randn(D, device=device, dtype=dtype)
    h0 = torch.randn(D, device=device, dtype=dtype)
    
    # Forward
    h = run_lds_scan(u, A, B_vec, h0)
    
    # Loss
    loss = h.sum()
    
    # Backward
    try:
        loss.backward()
        if u.grad is None:
            print("GRADIENT MISSING: u.grad is None")
        else:
            print(f"GRADIENT FOUND: u.grad norm = {u.grad.norm().item()}")
            if u.grad.abs().sum() == 0:
                print("GRADIENT IS ZERO")
    except Exception as e:
        print(f"Backward failed: {e}")

if __name__ == "__main__":
    check_grads()
