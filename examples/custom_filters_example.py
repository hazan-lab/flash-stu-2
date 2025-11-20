#!/usr/bin/env python3
"""
Example: Using Custom Spectral Filters

This example demonstrates how to:
1. Generate and save spectral filters
2. Load pre-computed filters in your model
3. Use filters for faster initialization
"""

import torch
from flash_stu import FlashSTU, FlashSTUConfig
from flash_stu.utils.stu_utils import (
    get_spectral_filters, 
    save_spectral_filters,
    load_spectral_filters
)

def example_1_save_and_load_filters():
    """Example 1: Generate, save, and load filters."""
    print("="*60)
    print("Example 1: Save and Load Filters")
    print("="*60)
    
    # Generate filters
    print("\n1. Generating filters...")
    phi = get_spectral_filters(
        seq_len=128,
        K=24,
        use_hankel_L=False,
        dtype=torch.bfloat16
    )
    print(f"   Generated filters: shape={phi.shape}, dtype={phi.dtype}")
    
    # Save to file
    print("\n2. Saving filters...")
    save_spectral_filters(
        phi, 
        "filters/standard_128_24.pt",
        metadata={
            "seq_len": 128,
            "K": 24,
            "use_hankel_L": False,
        }
    )
    
    # Load filters
    print("\n3. Loading filters...")
    phi_loaded = load_spectral_filters(
        "filters/standard_128_24.pt",
        dtype=torch.bfloat16
    )
    
    # Verify they match
    diff = (phi - phi_loaded).abs().max().item()
    print(f"\n4. Verification: max difference = {diff:.6e}")
    print(f"   Filters match!" if diff < 1e-6 else "   Warning: Filters differ")


def example_2_use_filters_in_model():
    """Example 2: Use pre-saved filters in a model."""
    print("\n" + "="*60)
    print("Example 2: Use Saved Filters in Model")
    print("="*60)
    
    # First, generate and save some filters
    print("\n1. Preparing filters...")
    phi = get_spectral_filters(256, 32, dtype=torch.bfloat16)
    save_spectral_filters(phi, "filters/custom_256_32.pt", 
                         metadata={"seq_len": 256, "K": 32})
    
    # Create model with saved filters
    print("\n2. Creating model with saved filters...")
    config = FlashSTUConfig(
        n_embd=512,
        n_layers=4,
        seq_len=256,
        num_eigh=32,
        vocab_size=1000,
        filter_path="filters/custom_256_32.pt",  # Load from file!
        torch_dtype=torch.bfloat16
    )
    
    model = FlashSTU(config).cuda()
    print(f"   Model created with loaded filters")
    print(f"   Phi buffer shape: {model.phi.shape}")
    
    # Test the model
    print("\n3. Testing model...")
    input_ids = torch.randint(0, 1000, (1, 10)).cuda()
    with torch.no_grad():
        outputs = model(input_ids)
        print(f"   Forward pass successful")
        print(f"   Output shape: {outputs.logits.shape}")


def example_3_compare_computed_vs_loaded():
    """Example 3: Compare initialization time."""
    print("\n" + "="*60)
    print("Example 3: Initialization Time Comparison")
    print("="*60)
    
    import time
    
    # Prepare saved filters
    phi = get_spectral_filters(512, 48, dtype=torch.bfloat16)
    save_spectral_filters(phi, "filters/perf_512_48.pt")
    
    # Time: Compute filters
    print("\n1. Creating model with computed filters...")
    start = time.time()
    config1 = FlashSTUConfig(
        n_embd=256,
        n_layers=4,
        seq_len=512,
        num_eigh=48,
        vocab_size=1000,
        torch_dtype=torch.bfloat16
    )
    model1 = FlashSTU(config1).cuda()
    time_compute = time.time() - start
    print(f"   Time: {time_compute:.3f}s")
    
    # Time: Load filters
    print("\n2. Creating model with loaded filters...")
    start = time.time()
    config2 = FlashSTUConfig(
        n_embd=256,
        n_layers=4,
        seq_len=512,
        num_eigh=48,
        vocab_size=1000,
        filter_path="filters/perf_512_48.pt",
        torch_dtype=torch.bfloat16
    )
    model2 = FlashSTU(config2).cuda()
    time_load = time.time() - start
    print(f"   Time: {time_load:.3f}s")
    
    # Compare
    speedup = time_compute / time_load
    print(f"\n3. Speedup: {speedup:.2f}x faster with loaded filters")


def example_4_hankel_L_filters():
    """Example 4: Use Hankel-L formulation."""
    print("\n" + "="*60)
    print("Example 4: Hankel-L Filters")
    print("="*60)
    
    # Generate Hankel-L filters
    print("\n1. Generating Hankel-L filters...")
    phi_L = get_spectral_filters(
        seq_len=128,
        K=24,
        use_hankel_L=True,
        dtype=torch.bfloat16
    )
    save_spectral_filters(
        phi_L,
        "filters/hankel_L_128_24.pt",
        metadata={"seq_len": 128, "K": 24, "use_hankel_L": True}
    )
    print(f"   Saved Hankel-L filters")
    
    # Generate standard filters for comparison
    print("\n2. Generating standard filters...")
    phi_std = get_spectral_filters(
        seq_len=128,
        K=24,
        use_hankel_L=False,
        dtype=torch.bfloat16
    )
    
    # Compare
    print("\n3. Comparing filter types...")
    diff = (phi_L - phi_std).abs().max().item()
    print(f"   Max difference: {diff:.4f}")
    print(f"   Hankel-L range: [{phi_L.min().item():.4f}, {phi_L.max().item():.4f}]")
    print(f"   Standard range: [{phi_std.min().item():.4f}, {phi_std.max().item():.4f}]")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPECTRAL FILTERS - EXAMPLES")
    print("="*60)
    
    # Run examples
    example_1_save_and_load_filters()
    example_2_use_filters_in_model()
    example_3_compare_computed_vs_loaded()
    example_4_hankel_L_filters()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use tools/generate_filters.py to pre-generate filter libraries")
    print("2. Share filter files across experiments for reproducibility")
    print("3. Enjoy faster model initialization!")
    print("="*60)
