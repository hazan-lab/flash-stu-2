
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from mini_stu.lds import LDS, random_LDS
from mini_stu.stu import MiniSTU

def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_comparison():
    # Configuration
    seq_len = 1024 # Use reasonable length for training loop speed
    input_dim = 64
    output_dim = 64
    state_dim = 64 # Target LDS dimension
    num_filters = 24 # Fixed for LDS backend compatibility
    batch_size = 16
    steps = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"Comparison Configuration:")
    print(f"  Seq Len: {seq_len}")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Steps: {steps}")
    print(f"  Device: {device}")
    
    # 1. Create Target LDS (Ground Truth)
    print("\nInitializing Target LDS...")
    target_lds = random_LDS(state_dim, input_dim, output_dim, device=device)
    target_lds.to(dtype=dtype)
    
    # 2. Initialize Models
    print("Initializing Models...")
    
    # Spectral Backend
    model_spectral = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=input_dim,
        output_dim=output_dim,
        use_hankel_L=False, # Match LDS backend requirement
        backend='spectral',
        device=device,
        dtype=dtype
    )
    
    # LDS Backend
    model_lds = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=input_dim,
        output_dim=output_dim,
        use_hankel_L=False,
        backend='lds',
        device=device,
        dtype=dtype
    )
    
    # Compile Models
    print("Compiling Models...")
    # Disable dynamic shapes to avoid issues with fixed-size buffers in MiniSTU
    model_spectral = torch.compile(model_spectral, dynamic=False)
    model_lds = torch.compile(model_lds, dynamic=False)
    
    # Optimizers
    opt_spectral = torch.optim.Adagrad(model_spectral.parameters(), lr=1e-1)
    opt_lds = torch.optim.Adagrad(model_lds.parameters(), lr=1e-1)
    
    # Metrics
    losses_spectral = []
    times_spectral = []
    losses_lds = []
    times_lds = []
    
    print("\nStarting Training Loop...")
    
    # Warmup
    x_dummy = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)
    with torch.no_grad():
        y_dummy = target_lds(x_dummy)
    train_step(model_spectral, opt_spectral, x_dummy, y_dummy)
    train_step(model_lds, opt_lds, x_dummy, y_dummy)
    torch.cuda.synchronize()
    
    # Training
    iterator = tqdm(range(steps))
    for i in iterator:
        # Generate data
        x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)
        with torch.no_grad():
            y = target_lds(x)
            
        # Train Spectral
        torch.cuda.synchronize()
        start = time.time()
        loss_s = train_step(model_spectral, opt_spectral, x, y)
        torch.cuda.synchronize()
        end = time.time()
        losses_spectral.append(loss_s)
        times_spectral.append(end - start)
        
        # Train LDS
        torch.cuda.synchronize()
        start = time.time()
        loss_l = train_step(model_lds, opt_lds, x, y)
        torch.cuda.synchronize()
        end = time.time()
        losses_lds.append(loss_l)
        times_lds.append(end - start)
        
        if i % 10 == 0:
            iterator.set_description(f"Loss S: {loss_s:.7f} | Loss L: {loss_l:.7f}")
            
    # Analysis
    avg_time_s = np.mean(times_spectral) * 1000
    avg_time_l = np.mean(times_lds) * 1000
    
    print("\n" + "="*40)
    print("Results")
    print("="*40)
    print(f"Spectral Backend Average Step Time: {avg_time_s:.2f} ms")
    print(f"LDS Backend Average Step Time:      {avg_time_l:.2f} ms")
    print(f"Speedup (Spectral / LDS):           {avg_time_s / avg_time_l:.2f}x")
    
    print(f"Final Loss Spectral: {losses_spectral[-1]:.7f}")
    print(f"Final Loss LDS:      {losses_lds[-1]:.7f}")
    
    # Check if Triton was used
    from mini_stu.MiniSTU_lds import HAS_TRITON
    print(f"\nTriton Enabled for LDS: {HAS_TRITON}")
    
    # Plotting
    print("\nGenerating Plots...")
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(losses_spectral, label='Spectral')
    plt.plot(losses_lds, label='LDS')
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.title('Training Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Prediction Plot (First feature of first batch)
    plt.subplot(1, 2, 2)
    with torch.no_grad():
         # Generate a fresh sequence with full length (MiniSTU requires fixed length)
         x_test = torch.randn(1, seq_len, input_dim, device=device, dtype=dtype)
         y_true = target_lds(x_test)
         y_spec = model_spectral(x_test)
         y_lds = model_lds(x_test)
         
         # Plot first output dimension for first 200 steps
         plot_len = 200
         t_steps = torch.arange(plot_len).cpu().numpy()
         y_t = y_true[0, :plot_len, 0].float().cpu().numpy()
         y_s = y_spec[0, :plot_len, 0].float().cpu().numpy()
         y_l = y_lds[0, :plot_len, 0].float().cpu().numpy()
         
         plt.plot(t_steps, y_t, 'k-', alpha=0.6, label='Ground Truth')
         plt.plot(t_steps, y_s, 'b--', label='Spectral Pred')
         plt.plot(t_steps, y_l, 'r:', label='LDS Pred')
         
    plt.xlabel('Time')
    plt.ylabel('Output [Dim 0]')
    plt.title('Prediction Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("Saved comparison_results.png")

if __name__ == "__main__":
    run_comparison()
