
import torch
import time
from mini_stu.stu import MiniSTU

def benchmark():
    seq_len = 8192
    num_filters = 24
    input_dim = 64
    output_dim = 64
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Benchmarking MiniSTU with:")
    print(f"  seq_len={seq_len}")
    print(f"  num_filters={num_filters}")
    print(f"  input_dim={input_dim}")
    print(f"  output_dim={output_dim}")
    print(f"  batch_size={batch_size}")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    print("-" * 40)

    # Initialize Spectral MiniSTU
    stu_spectral = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=input_dim,
        output_dim=output_dim,
        use_hankel_L=False,
        use_flash_fft=True,
        dtype=dtype,
        device=device,
        backend='spectral'
    )
    
    # Initialize Spectral MiniSTU (Vanilla)
    stu_spectral_vanilla = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=input_dim,
        output_dim=output_dim,
        use_hankel_L=False,
        use_flash_fft=False,
        dtype=dtype,
        device=device,
        backend='spectral'
    )
    
    # Initialize separate LDS for compilation test (optional)
    # ...

    # Initialize LDS MiniSTU
    stu_lds = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=input_dim,
        output_dim=output_dim,
        use_hankel_L=False,
        dtype=dtype,
        device=device,
        backend='lds'
    )
    
    # Check if Triton is actually being used
    from mini_stu.MiniSTU_lds import HAS_TRITON
    print(f"LDS backend initialized. Triton enabled: {HAS_TRITON}")

    # Dummy input
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = stu_spectral(x)
        _ = stu_spectral_vanilla(x)
        _ = stu_lds(x)
    
    torch.cuda.synchronize()
    
    # Benchmark Spectral (Flash)
    start = time.time()
    for _ in range(20):
        _ = stu_spectral(x)
    torch.cuda.synchronize()
    spectral_time = (time.time() - start) / 20
    print(f"Spectral (Flash) average time: {spectral_time * 1000:.2f} ms")

    # Benchmark Spectral (Vanilla)
    start = time.time()
    for _ in range(20):
        _ = stu_spectral_vanilla(x)
    torch.cuda.synchronize()
    spectral_vanilla_time = (time.time() - start) / 20
    print(f"Spectral (Vanilla) average time: {spectral_vanilla_time * 1000:.2f} ms")
    
    # Benchmark LDS
    start = time.time()
    for _ in range(20):
        _ = stu_lds(x)
    torch.cuda.synchronize()
    lds_time = (time.time() - start) / 20
    print(f"LDS average time:                {lds_time * 1000:.2f} ms")
    
    print("-" * 40)
    print(f"Flash vs Vanilla speedup: {spectral_vanilla_time / spectral_time:.2f}x")
    print(f"Flash vs LDS speedup:     {lds_time / spectral_time:.2f}x")
    print(f"Vanilla vs LDS speedup:   {lds_time / spectral_vanilla_time:.2f}x")
    
    # Verification (Outputs won't match exactly because they use different weights/initialization)
    # But we can check shapes and basic execution
    out_spectral = stu_spectral(x)
    out_lds = stu_lds(x)
    
    print("-" * 40)
    print(f"Spectral output shape: {out_spectral.shape}")
    print(f"LDS output shape:      {out_lds.shape}")
    
    assert out_spectral.shape == out_lds.shape
    print("Shapes match!")

if __name__ == "__main__":
    benchmark()
