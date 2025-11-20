import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .stu import MiniSTU


class LDS(nn.Module):
    """
    Linear Dynamical System (LDS) implementation.
    
    An LDS represents a dynamical system with linear state transitions:
        h_{t+1} = A * h_t + B * u_t  (state update)
        y_t = C * h_t                 (output)
    
    Where:
        - h_t: hidden state at time t
        - u_t: input at time t
        - y_t: output at time t
        - A: state transition matrix (diagonal)
        - B: input-to-state matrix
        - C: state-to-output matrix
    
    Args:
        state_dim: Dimension of hidden state
        input_dim: Dimension of input
        output_dim: Dimension of output
        dtype: Data type for parameters
        device: Device to place module on
    
    Shape:
        - Input: [batch_size, seq_len, input_dim]
        - Output: [batch_size, seq_len, output_dim]
        
    Example:
        >>> lds = LDS(state_dim=20, input_dim=10, output_dim=5)
        >>> inputs = torch.randn(4, 100, 10)
        >>> outputs = lds(inputs)
        >>> outputs.shape
        torch.Size([4, 100, 5])
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # LDS parameters
        # A: state transition (diagonal, clipped to [-1, 1] for stability)
        self.A = nn.Parameter(
            torch.clip(torch.randn(state_dim), -1.0, 1.0).to(dtype).to(self.device)
        )
        
        # B: input to state transformation
        self.B = nn.Parameter(
            (torch.randn(input_dim, state_dim) / input_dim).to(dtype).to(self.device)
        )
        
        # C: state to output transformation
        self.C = nn.Parameter(
            (torch.randn(state_dim, output_dim) / state_dim).to(dtype).to(self.device)
        )
        
        # h0: initial hidden state
        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype).to(self.device))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Generate LDS trajectory for given input sequence.
        
        Args:
            inputs: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            Output sequence [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, input_dim = inputs.shape
        outputs = []
        
        # Initialize hidden state for all sequences in batch
        h = self.h0.unsqueeze(0).expand(batch_size, -1)
        
        for t in range(seq_len):
            # State update: h_{t+1} = A * h_t + B * u_t
            h = h * self.A + inputs[:, t, :] @ self.B
            
            # Output: y_t = C * h_t
            y_t = h @ self.C
            outputs.append(y_t.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)


def random_LDS(
    state_dim: int,
    input_dim: int,
    output_dim: int,
    lower_bound: float = 0.1,
    device: torch.device = None
) -> LDS:
    """
    Create a random LDS with specified dimensions.
    
    The state transition eigenvalues (A) are randomly initialized in the
    range [lower_bound, 1.0] with random signs for stability.
    
    Args:
        state_dim: Dimension of hidden state
        input_dim: Dimension of input
        output_dim: Dimension of output
        lower_bound: Minimum absolute value for state transition eigenvalues
        device: Device to place LDS on
        
    Returns:
        Randomly initialized LDS
    """
    device = device or torch.device('cpu')
    lds = LDS(state_dim, input_dim, output_dim, device=device)
    
    # Custom initialization for A parameter to ensure stability
    A_values = torch.rand(state_dim, device=device) * (1 - lower_bound) + lower_bound
    signs = torch.randint(0, 2, (state_dim,), device=device) * 2 - 1
    lds.A = nn.Parameter((A_values * signs.float()).to(device))
    
    return lds


def train_stu_on_lds(
    lds: LDS,
    seq_len: int,
    num_filters: int,
    num_steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> tuple[MiniSTU, list]:
    """
    Train a MiniSTU to approximate an LDS.
    
    This trains a MiniSTU to learn the input-output behavior of a given LDS
    by minimizing the MSE between STU outputs and LDS outputs on random inputs.
    
    Args:
        lds: Target LDS to learn
        seq_len: Sequence length for training
        num_filters: Number of spectral filters for STU
        num_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: If True, show progress bar and periodic loss
        
    Returns:
        Tuple of (trained_stu, loss_history)
    """
    # Create STU model matching LDS dimensions
    stu = MiniSTU(
        seq_len=seq_len,
        num_filters=num_filters,
        input_dim=lds.input_dim,
        output_dim=lds.output_dim,
        device=lds.device,
        dtype=lds.dtype
    ).to(lds.device)
    
    # Setup training
    optimizer = torch.optim.Adam(stu.parameters(), lr=learning_rate)
    losses = []
    
    stu.train()
    iterator = tqdm(range(num_steps), desc="Training STU") if verbose else range(num_steps)
    
    for step in iterator:
        # Generate random input sequences
        inputs = torch.randn(
            batch_size, seq_len, lds.input_dim,
            device=lds.device, dtype=lds.dtype
        )
        
        # Generate target outputs from LDS (no gradients needed)
        with torch.no_grad():
            targets = lds(inputs)
        
        # STU forward pass
        outputs = stu(inputs)
        
        # Compute loss
        loss = F.mse_loss(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print periodic updates
        if verbose and step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
    
    stu.eval()
    return stu, losses

