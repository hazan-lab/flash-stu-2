import torch
import torch.nn as nn
from typing import Dict
import time

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_stu.modules.stu import STU
from flash_stu.config import FlashSTUConfig
from flash_stu.utils.stu_utils import get_spectral_filters


def compare_outputs(output1: torch.Tensor, output2: torch.Tensor, name1: str = "A", name2: str = "B") -> Dict:
    abs_diff = torch.abs(output1 - output2)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    output1_norm = torch.abs(output1).mean().item()
    rel_error = mean_abs_error / output1_norm if output1_norm > 0 else 0
    
    rtol, atol = 1e-5, 1e-8
    is_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    
    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'rel_error_percent': rel_error * 100,
        'is_close': is_close,
        'rtol': rtol,
        'atol': atol,
    }


def benchmark_memory_and_speed(
    stu_module: nn.Module,
    batch_size: int,
    seq_len: int,
    n_warmup: int = 3,
    n_iterations: int = 10,
    name: str = "STU",
    enable_profiling: bool = False,
    output_dir: Path = None
) -> Dict:
    """Benchmark memory and speed for a given STU module."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stu_module = stu_module.to(device)
    stu_module.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, stu_module.d_in, 
                    dtype=stu_module.config.torch_dtype, device=device)
    
    # Reset memory stats
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = stu_module(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Measure memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    if enable_profiling and device.type == 'cuda':
        from torch.profiler import profile, ProfilerActivity, record_function
        
        trace_file = output_dir / f"{name}_trace.json"
        memory_profile_file = output_dir / f"{name}_memory_profile.txt"
        start_time = time.time()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
            with_stack=False
        ) as prof:
            with torch.no_grad():
                for i in range(n_iterations):
                    with record_function(f"iteration_{i}"):
                        output = stu_module(x)
                        torch.cuda.synchronize()
        
        end_time = time.time()

        prof.export_chrome_trace(str(trace_file))
        print(f"Chrome trace saved to: {trace_file}")
        
        with open(memory_profile_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Memory Profile for: {name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Top operations by CUDA memory:\n")
            f.write("-" * 80 + "\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_memory_usage", 
                row_limit=20, 
                
            ))
            f.write("\n\n")
            
            f.write("Top operations by CUDA time:\n")
            f.write("-" * 80 + "\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total", 
                row_limit=20
            ))
            f.write("\n")
        
        print(f"Memory profile saved to: {memory_profile_file}")
    else:
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_iterations):
                output = stu_module(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        end_time = time.time()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = 0.0
    
    avg_time = (end_time - start_time) / n_iterations * 1000
    
    return {
        'name': name,
        'peak_memory_mb': peak_memory,
        'avg_time_ms': avg_time,
        'output_shape': list(output.shape),
        'output': output.detach().clone(),
    }


def main(enable_profiling: bool = False, profile_dir: str = "./profiling_results"):    
    if enable_profiling:
        output_dir = Path(profile_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProfiling enabled. Results will be saved to: {output_dir}")
    else:
        output_dir = None
    
    config = FlashSTUConfig(
        n_embd=2048,
        seq_len=4096,
        num_eigh=24,
        use_approx=True,
        use_hankel_L=False,
        use_flash_fft=False,
        torch_dtype=torch.bfloat16
    )

    config_hybrid = FlashSTUConfig(
        n_embd=2048,
        seq_len=4096,
        num_eigh=24,
        use_approx=True,
        use_hankel_L=False,
        use_flash_fft=False, 
        torch_dtype=torch.bfloat16, 
        d_in_tile = 1024,
        d_out_tile = 1024
    )
    config_d_in = FlashSTUConfig(
        n_embd=2048,
        seq_len=4096,
        num_eigh=24,
        use_approx=True,
        use_hankel_L=False,
        use_flash_fft=False, 
        torch_dtype=torch.bfloat16, 
        d_in_tile = 1024,
    )

    config_d_out = FlashSTUConfig(
        n_embd=2048,
        seq_len=4096,
        num_eigh=24,
        use_approx=True,
        use_hankel_L=False,
        use_flash_fft=False, 
        torch_dtype=torch.bfloat16, 
        d_out_tile = 1024,
    )
    
    batch_size = 1
    seq_len = 4096
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config.n_embd}")
    print(f"  seq_len: {seq_len}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_eigh (K): {config.num_eigh}")
    print(f"  use_approx: {config.use_approx}")
    print(f"  dtype: {config.torch_dtype}")
    
    # Get spectral filters
    print("\n" + "=" * 80)
    print("Setting up modules...")
    phi = get_spectral_filters(config.seq_len, config.num_eigh, dtype=config.torch_dtype)
    from flash_stu.utils.numerics import nearest_power_of_two
    n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
    
    modules = {}
    
    print("\n1. Standard STU (no tiling)")
    modules['standard'] = STU(config, phi, n)
    
    print("2. STU with hybrid tiling")
    modules['hybrid'] = STU(config_hybrid, phi, n)

    print("3. STU with d_in tiling")
    modules['d_in'] = STU(config_d_in, phi, n)

    print("4. STU with d_out tiling")
    modules['d_out'] = STU(config_d_out, phi, n)

    # Benchmark each module
    print("\n" + "=" * 80)
    print("Running benchmarks...")
    print("=" * 80)
    
    results = []
    for name, module in modules.items():
        print(f"\nBenchmarking: {name}...")
        try:
            result = benchmark_memory_and_speed(
                module, batch_size, seq_len, 
                n_warmup=3, n_iterations=10, name=name,
                enable_profiling=enable_profiling,
                output_dir=output_dir
            )
            results.append(result)
            print(f"  ✓ Peak Memory: {result['peak_memory_mb']:.2f} MB")
            print(f"  ✓ Avg Time: {result['avg_time_ms']:.2f} ms")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        baseline = next((r for r in results if r['name'] == 'standard'), results[0])
        baseline_mem = baseline['peak_memory_mb']
        baseline_time = baseline['avg_time_ms']
        
        print(f"\n{'Module':<20} {'Peak Memory':<15} {'Memory vs Base':<20} {'Avg Time':<12} {'Time vs Base':<15}")
        print("-" * 90)
        
        for result in results:
            mem_ratio = result['peak_memory_mb'] / baseline_mem if baseline_mem > 0 else 1.0
            time_ratio = result['avg_time_ms'] / baseline_time if baseline_time > 0 else 1.0
            mem_reduction = (mem_ratio - 1) * 100
            time_overhead = (time_ratio - 1) * 100
            
            print(f"{result['name']:<20} "
                  f"{result['peak_memory_mb']:>8.2f} MB    "
                  f"{mem_ratio:>5.2f}x ({mem_reduction:>+6.1f}%)    "
                  f"{result['avg_time_ms']:>7.2f} ms   "
                  f"{time_ratio:>5.2f}x ({time_overhead:>+6.1f}%)")
        
        print("\n" + "=" * 80)
        print("OUTPUT ACCURACY COMPARISON")
        print("=" * 80)
        print(f"\nComparing all outputs against: {baseline['name']}")
        print(f"{'Module':<20} {'Max Abs Error':<15} {'Mean Abs Error':<16} {'Rel Error %':<14} {'Match':<10}")
        print("-" * 80)
        
        baseline_output = baseline['output']
        rtol, atol = 1e-5, 1e-8
        
        for result in results:
            if result['name'] == baseline['name']:
                print(f"{result['name']:<20} {'---':<15} {'---':<16} {'---':<14} {'✓ (baseline)':<10}")
            else:
                comparison = compare_outputs(baseline_output, result['output'], 
                                            baseline['name'], result['name'])
                match_symbol = "✓" if comparison['is_close'] else "✗"
                
                print(f"{result['name']:<20} "
                      f"{comparison['max_abs_error']:<15.2e} "
                      f"{comparison['mean_abs_error']:<16.2e} "
                      f"{comparison['rel_error_percent']:<14.4f} "
                      f"{match_symbol:<10}")
        
        print(f"\nTolerance: rtol={rtol}, atol={atol}")
        print("Note: ✓ means outputs match within tolerance, ✗ means they differ significantly")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark STU tiling"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="./profiling_results",
    )
    
    args = parser.parse_args()
    main(enable_profiling=args.profile, profile_dir=args.profile_dir)

