"""
Example usage of MiniSTU for learning linear dynamical systems.

This script demonstrates:
1. Creating a random LDS
2. Training a MiniSTU to approximate it
3. Evaluating the learned model
4. Comparing STU vs LDS outputs
"""

import torch
import matplotlib.pyplot as plt
from mini_stu import MiniSTU, LDS, random_LDS, train_stu_on_lds


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # =========================================================================
    # 1. Create a Random LDS
    # =========================================================================
    print("=" * 70)
    print("Creating Random Linear Dynamical System")
    print("=" * 70)
    
    state_dim = 20
    input_dim = 10
    output_dim = 5
    
    lds = random_LDS(
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        lower_bound=0.1,
        device=device
    )
    
    print(f"State dimension:  {state_dim}")
    print(f"Input dimension:  {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"LDS parameters:   {sum(p.numel() for p in lds.parameters())}")
    print()
    
    # =========================================================================
    # 2. Train MiniSTU to Learn the LDS
    # =========================================================================
    print("=" * 70)
    print("Training MiniSTU to Approximate LDS")
    print("=" * 70)
    
    seq_len = 128
    num_filters = 24
    num_steps = 1000
    batch_size = 32
    learning_rate = 1e-3
    
    print(f"Sequence length:  {seq_len}")
    print(f"Spectral filters: {num_filters}")
    print(f"Training steps:   {num_steps}")
    print(f"Batch size:       {batch_size}")
    print()
    
    stu, losses = train_stu_on_lds(
        lds=lds,
        seq_len=seq_len,
        num_filters=num_filters,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=True,
    )
    
    print(f"\nSTU parameters: {stu.get_num_params()}")
    print(f"Final loss: {losses[-1]:.6f}")
    print()
    
    # =========================================================================
    # 3. Evaluate on Test Data
    # =========================================================================
    print("=" * 70)
    print("Evaluating on Test Data")
    print("=" * 70)
    
    test_batch_size = 10
    test_inputs = torch.randn(
        test_batch_size, seq_len, input_dim,
        device=device, dtype=torch.float32
    )
    
    with torch.no_grad():
        lds_outputs = lds(test_inputs)
        stu_outputs = stu(test_inputs)
    
    test_mse = torch.mean((stu_outputs - lds_outputs) ** 2).item()
    
    print(f"Test MSE: {test_mse:.6f}")
    
    # Per-sequence MSE
    per_seq_mse = torch.mean((stu_outputs - lds_outputs) ** 2, dim=(1, 2))
    print(f"Mean MSE per sequence: {per_seq_mse.mean().item():.6f}")
    print(f"Std MSE per sequence:  {per_seq_mse.std().item():.6f}")
    print()
    
    # =========================================================================
    # 4. Visualize Results
    # =========================================================================
    print("=" * 70)
    print("Visualization")
    print("=" * 70)
    
    # Plot training loss
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Compare outputs for first sequence
    seq_idx = 0
    time_steps = range(seq_len)
    output_dim_to_plot = 0
    
    axes[1].plot(
        time_steps,
        lds_outputs[seq_idx, :, output_dim_to_plot].cpu().numpy(),
        label='LDS (Ground Truth)',
        linewidth=2
    )
    axes[1].plot(
        time_steps,
        stu_outputs[seq_idx, :, output_dim_to_plot].cpu().numpy(),
        label='STU (Learned)',
        linewidth=2,
        linestyle='--'
    )
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Output Value')
    axes[1].set_title(f'Output Comparison (Dim {output_dim_to_plot})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Error over time
    error = torch.abs(stu_outputs[seq_idx] - lds_outputs[seq_idx]).cpu().numpy()
    axes[2].plot(time_steps, error.mean(axis=1))
    axes[2].fill_between(
        time_steps,
        error.mean(axis=1) - error.std(axis=1),
        error.mean(axis=1) + error.std(axis=1),
        alpha=0.3
    )
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Absolute Error')
    axes[2].set_title('Prediction Error Over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mini_stu_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: mini_stu_results.png")
    print()
    
    # =========================================================================
    # 5. Test with Different Sequence Lengths
    # =========================================================================
    print("=" * 70)
    print("Testing Generalization to Different Lengths")
    print("=" * 70)
    
    for test_len in [64, 128, 256]:
        if test_len <= seq_len:
            test_input = torch.randn(1, test_len, input_dim, device=device)
            
            with torch.no_grad():
                lds_out = lds(test_input)
                stu_out = stu(test_input[:, :test_len, :])  # Truncate if needed
            
            mse = torch.mean((stu_out - lds_out) ** 2).item()
            print(f"Length {test_len:3d}: MSE = {mse:.6f}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

