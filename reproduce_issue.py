
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from mini_stu.lds import random_LDS
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
    seq_len = 1024
    input_dim = 64
    output_dim = 64
    state_dim = 64
    num_filters = 24
    batch_size = 16
    steps = 50 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 # Changed to bfloat16
    
    print(f"Comparison Configuration:")
    print(f"  Dtype: {dtype}")
    print(f"  Device: {device}")
    
    # 1. Create Target LDS (Ground Truth)
    # Target in float32 for stability of generation
    target_lds = random_LDS(state_dim, input_dim, output_dim, device=device)
    target_lds.to(dtype=torch.float32) 
    
    # 2. Initialize Models
    print("Initializing Models...")
    
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
    
    # Move to device and dtype
    model_lds = model_lds.to(device=device, dtype=dtype)
    
    # Compile
    # model_lds = torch.compile(model_lds, dynamic=False) # Skip compile for easier debugging
    
    # Optimizer
    opt_lds = torch.optim.Adagrad(model_lds.parameters(), lr=1e-1)
    
    losses_lds = []
    
    print("\nStarting Training Loop...")
    
    # Training
    for i in range(steps):
        # Generate data in float32 then cast to dtype
        x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float32)
        with torch.no_grad():
            y = target_lds(x)
            
        x = x.to(dtype=dtype)
        y = y.to(dtype=dtype)
        
        # Train LDS
        loss_l = train_step(model_lds, opt_lds, x, y)
        losses_lds.append(loss_l)
        
        if i % 10 == 0:
            print(f"Step {i}: Loss L: {loss_l:.7f}")
            if np.isnan(loss_l) or np.isinf(loss_l):
                print("Instability detected!")
                break
            
    print(f"Final Loss LDS: {losses_lds[-1]:.7f}")

if __name__ == "__main__":
    run_comparison()
