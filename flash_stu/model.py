import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from flash_stu.modules.stu import STU
from flash_stu.modules.attention import Attention
from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.config import FlashSTUConfig
from flash_stu.layers.stu_layer import STULayer
from flash_stu.layers.attention_layer import AttentionLayer

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as TritonNorm
    triton_norm = True
except ImportError:
    from torch.nn import RMSNorm
    triton_norm = False


class FlashSTU(PreTrainedModel):
    """
    FlashSTU: Spectral Transform Unit Language Model
    
    A HuggingFace-compatible language model using Spectral Transform Units (STU)
    and optional attention layers for sequence modeling.
    
    Args:
        config: FlashSTUConfig containing model hyperparameters
    
    Example:
        >>> from flash_stu import FlashSTU, FlashSTUConfig
        >>> config = FlashSTUConfig(n_embd=768, n_layers=12, seq_len=2048)
        >>> model = FlashSTU(config)
        >>> 
        >>> # Forward pass
        >>> input_ids = torch.randint(0, config.vocab_size, (1, 128))
        >>> outputs = model(input_ids)
        >>> logits = outputs.logits
        >>> 
        >>> # Generation
        >>> generated = model.generate(input_ids, max_length=256)
    """
    config_class = FlashSTUConfig
    _tied_weights_keys = ["lm_head.weight"]  # Tied with tok_emb.weight

    def __init__(self, config: FlashSTUConfig) -> None:
        super(FlashSTU, self).__init__(config)
        self.config = config
        self.n_layers = config.n_layers
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.use_approx = config.use_approx
        self.use_hankel_L = config.use_hankel_L
        
        # Compute and register phi as a buffer
        phi = get_spectral_filters(config.num_eigh, config.seq_len)
        self.register_buffer('phi', phi, persistent=True)

        # Embedding layer
        self.tok_emb = nn.Embedding(
            config.vocab_size, config.n_embd, dtype=config.torch_dtype
        )
        self.dropout = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            # For more complex %-split arrangements, see https://arxiv.org/pdf/2406.07887
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, self.phi, self.n))
            else:
                self.layers.append(
                    AttentionLayer(config)
                    if config.use_attn
                    else STULayer(config, self.phi, self.n)
                )

        # Final layer norm
        self.norm = (
            TritonNorm(config.n_embd)
            if triton_norm
            else RMSNorm(config.n_embd, dtype=config.torch_dtype)
        )
        # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for RMS Norm
        self.norm = self.norm.to(dtype=config.torch_dtype)
        
        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.bias, dtype=config.torch_dtype
        )
        self.tok_emb.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.std = (config.n_embd) ** -0.5
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the FlashSTU model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask (currently unused, for HF compatibility)
            labels: Labels for computing language modeling loss [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict (always True for HF compatibility)
        
        Returns:
            Dictionary with:
                - logits: Output logits [batch_size, seq_len, vocab_size]
                - loss: Language modeling loss (if labels provided)
                - hidden_states: List of hidden states (if output_hidden_states=True)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict if hasattr(self.config, 'use_return_dict') else True
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        # Input validation
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        # Store hidden states if requested
        all_hidden_states = () if output_hidden_states else None

        # Embedding
        x = self.tok_emb(input_ids)
        x = self.dropout(x)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        # Final layer norm and fix dtype
        x = self.norm(x).to(x.dtype)
        
        # Language modeling head
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # We don't use KV cache yet
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p (nucleus sampling)
            do_sample: Whether to sample or use greedy decoding
        
        Returns:
            Generated token IDs [batch_size, max_length]
        
        Example:
            >>> input_ids = torch.tensor([[1, 2, 3]])
            >>> generated = model.generate(input_ids, max_length=50, temperature=0.9, top_k=50)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Start with the input
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get predictions for the last token
            # Truncate to max sequence length if needed
            if generated.shape[1] > self.config.seq_len:
                input_seq = generated[:, -self.config.seq_len:]
            else:
                input_seq = generated
            
            outputs = self(input_seq)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
            
            # Get logits for the last token and clamp extreme values
            next_token_logits = logits[:, -1, :] / temperature
            next_token_logits = torch.clamp(next_token_logits, min=-1e4, max=1e4)
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    next_token = torch.randint(0, self.config.vocab_size, (batch_size, 1), device=device)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
        
        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation. Required by HuggingFace generation utilities.
        
        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of inputs for the model
        """
        return {
            "input_ids": input_ids,
        }

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save model and configuration to a directory.
        
        Args:
            save_directory: Directory to save to
            **kwargs: Additional arguments passed to parent
        """
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load model from a directory or HuggingFace Hub.
        
        Args:
            *args: Arguments passed to parent from_pretrained
            **kwargs: Keyword arguments passed to parent
        
        Returns:
            Loaded FlashSTU model
        """
        return super().from_pretrained(*args, **kwargs)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters
        
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            # Subtract embedding parameters
            if hasattr(self, "tok_emb"):
                n_params -= self.tok_emb.weight.numel()
            # Account for weight tying
            if hasattr(self, "tok_emb") and hasattr(self, "lm_head"):
                if self.tok_emb.weight is self.lm_head.weight:
                    # Weights are tied, already subtracted
                    pass
                else:
                    # Not tied, subtract lm_head too
                    n_params -= self.lm_head.weight.numel()
        else:
            # Account for weight tying in total count
            if hasattr(self, "tok_emb") and hasattr(self, "lm_head"):
                if self.tok_emb.weight is self.lm_head.weight:
                    n_params -= self.tok_emb.weight.numel()
        
        return n_params

    def _get_num_params(self):
        """Backward compatibility with old code."""
        return self.get_num_params(non_embedding=False)

    def _init_weights(self, module):
        """Initialize weights for different module types."""
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, STU):
            if self.use_approx:
                torch.nn.init.xavier_normal_(module.M_inputs)
                torch.nn.init.xavier_normal_(module.M_filters)
            else:
                torch.nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    torch.nn.init.xavier_normal_(module.M_phi_minus)
        elif isinstance(module, Attention):
            torch.nn.init.xavier_normal_(module.c_attn.weight)
            torch.nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                torch.nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                torch.nn.init.zeros_(module.c_proj.bias)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resize token embeddings. Useful when adding special tokens.
        
        Args:
            new_num_tokens: New vocabulary size
        
        Returns:
            The resized embedding layer
        """
        old_num_tokens = self.config.vocab_size
        
        if new_num_tokens == old_num_tokens:
            return self.tok_emb
        
        # Create new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens, 
            self.config.n_embd, 
            dtype=self.config.torch_dtype
        )
        new_embeddings.to(self.tok_emb.weight.device)
        
        # Initialize new embeddings
        torch.nn.init.normal_(new_embeddings.weight, mean=0.0, std=self.std)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = self.tok_emb.weight.data[:num_tokens_to_copy]
        
        # Update model
        self.tok_emb = new_embeddings
        
        # Resize lm_head
        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(
            self.config.n_embd,
            new_num_tokens,
            bias=self.config.bias,
            dtype=self.config.torch_dtype
        )
        self.lm_head.to(old_lm_head.weight.device)
        
        # Initialize
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.std)
        if self.lm_head.bias is not None:
            torch.nn.init.zeros_(self.lm_head.bias)
        
        # Copy old weights
        self.lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[:num_tokens_to_copy]
        if self.lm_head.bias is not None and old_lm_head.bias is not None:
            self.lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        
        # Re-tie weights if they were tied
        self.tok_emb.weight = self.lm_head.weight
        
        # Update config
        self.config.vocab_size = new_num_tokens
        
        return self.tok_emb
