"""
Benchmark tensordot vs einsum for STU contraction operations.

This benchmark compares different methods of performing the tensor contraction
in the STU module: U[batch, seq, K, d_in] @ M[K, d_in, d_out] -> output[batch, seq, d_out]

Tested methods:
- torch.tensordot (current implementation)
- torch.matmul with reshape
- torch.einsum with various formulations
"""
import os
os.environ['TRITON_CACHE_DIR'] = os.path.expanduser('~/.triton_cache_test')

import torch
import time

def benchmark_contraction():
    """Compare tensordot vs einsum for the STU contraction."""
    print("="*60)
    print("Tensordot vs Einsum Benchmark")
    print("="*60)
    
    # Test configurations (similar to STU module)
    configs = [
        {"name": "Small", "batch": 2, "seq_len": 128, "K": 24, "d_in": 512, "d_out": 512},
        {"name": "Medium", "batch": 4, "seq_len": 256, "K": 32, "d_in": 1024, "d_out": 1024},
        {"name": "Large", "batch": 8, "seq_len": 512, "K": 48, "d_in": 2048, "d_out": 2048},
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    warmup = 10
    iterations = 100
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Shape: U [{config['batch']}, {config['seq_len']}, {config['K']}, {config['d_in']}]")
        print(f"         M [{config['K']}, {config['d_in']}, {config['d_out']}]")
        
        # Create test tensors
        U = torch.randn(
            config['batch'], config['seq_len'], config['K'], config['d_in'],
            device=device, dtype=dtype
        )
        M = torch.randn(
            config['K'], config['d_in'], config['d_out'],
            device=device, dtype=dtype
        )
        
        # Warmup
        for _ in range(warmup):
            _ = torch.tensordot(U, M, dims=([2, 3], [0, 1]))
            _ = torch.einsum('bsKd,Kde->bse', U, M)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark tensordot
        start = time.perf_counter()
        for _ in range(iterations):
            result_td = torch.tensordot(U, M, dims=([2, 3], [0, 1]))
        if device == 'cuda':
            torch.cuda.synchronize()
        time_tensordot = (time.perf_counter() - start) / iterations * 1000
        
        # Benchmark einsum
        start = time.perf_counter()
        for _ in range(iterations):
            result_es = torch.einsum('bsKd,Kde->bse', U, M)
        if device == 'cuda':
            torch.cuda.synchronize()
        time_einsum = (time.perf_counter() - start) / iterations * 1000
        
        # Verify results match
        max_diff = (result_td - result_es).abs().max().item()
        
        print(f"  tensordot: {time_tensordot:.3f} ms")
        print(f"  einsum:    {time_einsum:.3f} ms")
        print(f"  Speedup:   {time_tensordot/time_einsum:.2f}x {'(einsum faster)' if time_einsum < time_tensordot else '(tensordot faster)'}")
        print(f"  Max diff:  {max_diff:.2e}")
    
    # Test alternative einsum formulations
    print("\n" + "="*60)
    print("Alternative Formulations (Medium config)")
    print("="*60)
    
    config = configs[1]
    U = torch.randn(
        config['batch'], config['seq_len'], config['K'], config['d_in'],
        device=device, dtype=dtype
    )
    M = torch.randn(
        config['K'], config['d_in'], config['d_out'],
        device=device, dtype=dtype
    )
    
    alternatives = [
        ("tensordot", lambda: torch.tensordot(U, M, dims=([2, 3], [0, 1]))),
        ("matmul (reshape)", lambda: (
            U.reshape(config['batch'] * config['seq_len'], config['K'] * config['d_in']) @ 
            M.reshape(config['K'] * config['d_in'], config['d_out'])
        ).reshape(config['batch'], config['seq_len'], config['d_out'])),
        ("einsum 'bsKd,Kde->bse'", lambda: torch.einsum('bsKd,Kde->bse', U, M)),
        ("einsum 'bski,kio->bso'", lambda: torch.einsum('bski,kio->bso', U, M)),
        ("einsum (with spaces)", lambda: torch.einsum('b s k d, k d e -> b s e', U, M)),
    ]
    
    results = []
    for name, fn in alternatives:
        # Warmup
        for _ in range(warmup):
            _ = fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            result = fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations * 1000
        results.append((name, elapsed))
    
    # Sort by speed
    results.sort(key=lambda x: x[1])
    fastest = results[0][1]
    
    for name, elapsed in results:
        slowdown = ((elapsed / fastest) - 1) * 100
        marker = " [FASTEST]" if elapsed == fastest else f" (+{slowdown:.0f}%)"
        print(f"  {name:30s}: {elapsed:.3f} ms{marker}")


if __name__ == '__main__':
    benchmark_contraction()

