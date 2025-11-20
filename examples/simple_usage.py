"""
Simple usage examples for FlashSTU model.

This script demonstrates:
1. Creating a FlashSTU model
2. Forward pass with automatic loss computation
3. Text generation
4. Saving and loading models
5. Using FlashSTUBlock as a standalone component
"""

import torch
from flash_stu import FlashSTU, FlashSTUConfig, FlashSTUBlock

def example_1_basic_forward():
    """Example 1: Basic forward pass with loss computation"""
    print("=" * 60)
    print("Example 1: Basic Forward Pass")
    print("=" * 60)
    
    # Create config
    config = FlashSTUConfig(
        n_embd=512,
        n_layers=6,
        seq_len=1024,
        vocab_size=50257,
        use_attn=True,
        torch_dtype=torch.bfloat16
    )
    
    # Create model (phi is computed internally now!)
    model = FlashSTU(config)
    model = model.cuda()
    model.eval()
    
    print(f"Model created with {model.get_num_params():,} parameters")
    print(f"Phi device: {model.phi.device}")
    print(f"Phi shape: {model.phi.shape}")
    
    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
    labels = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"Forward pass successful")
    print(f"  - Logits shape: {outputs.logits.shape}")
    print(f"  - Loss: {outputs.loss.item():.4f}")
    print()


def example_2_generation():
    """Example 2: Text generation"""
    print("=" * 60)
    print("Example 2: Text Generation")
    print("=" * 60)
    
    config = FlashSTUConfig(
        n_embd=256,
        n_layers=4,
        seq_len=512,
        vocab_size=1000,
        torch_dtype=torch.bfloat16
    )
    
    model = FlashSTU(config).cuda()
    model.eval()
    
    # Start with a prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]], device='cuda')
    
    # Generate with different sampling strategies
    print("Generating with temperature=1.0, top_k=50")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=50,
            temperature=1.0,
            top_k=50,
            do_sample=True
        )
    print(f"  - Generated shape: {generated.shape}")
    print(f"  - First 20 tokens: {generated[0, :20].tolist()}")
    
    print("\nGenerating with temperature=0.7, top_p=0.9")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    print(f"  - Generated shape: {generated.shape}")
    print(f"  - First 20 tokens: {generated[0, :20].tolist()}")
    
    print("\nGreedy decoding (do_sample=False)")
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_length=50,
            do_sample=False
        )
    print(f"  - Generated shape: {generated.shape}")
    print(f"  - First 20 tokens: {generated[0, :20].tolist()}")
    print()


def example_3_save_and_load():
    """Example 3: Saving and loading models"""
    print("=" * 60)
    print("Example 3: Save and Load")
    print("=" * 60)
    
    config = FlashSTUConfig(
        n_embd=256,
        n_layers=4,
        seq_len=512,
        vocab_size=1000,
        torch_dtype=torch.bfloat16
    )
    
    # Create and save model
    model = FlashSTU(config).cuda()
    save_path = "/tmp/flash_stu_test"
    
    print(f"Saving model to {save_path}")
    model.save_pretrained(save_path)
    print(f"  - Config saved")
    print(f"  - Weights saved")
    print(f"  - Phi saved (as part of state dict)")
    
    # Load model
    print(f"\nLoading model from {save_path}")
    loaded_model = FlashSTU.from_pretrained(save_path).cuda()
    
    # Verify phi is loaded correctly
    print(f"  - Phi device: {loaded_model.phi.device}")
    print(f"  - Phi shape: {loaded_model.phi.shape}")
    
    # Test that loaded model works
    input_ids = torch.randint(0, 1000, (1, 64), device='cuda')
    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids)
    
    print(f"Loaded model works!")
    print(f"  - Output logits shape: {outputs.logits.shape}")
    print()


def example_4_standalone_blocks():
    """Example 4: Using FlashSTUBlock standalone"""
    print("=" * 60)
    print("Example 4: Standalone FlashSTUBlock")
    print("=" * 60)
    
    # Create standalone STU block
    stu_block = FlashSTUBlock(
        d_model=256,
        sequence_length=512,
        num_filters=16,
        use_attention=False,
        use_flash_fft=True
    ).cuda()
    
    print(f"STU Block created")
    print(f"  - Is attention block: {stu_block.is_attention_block}")
    print(f"  - Num parameters: {stu_block.get_num_params():,}")
    
    # Forward pass
    x = torch.randn(2, 128, 256, device='cuda', dtype=torch.bfloat16)
    out = stu_block(x)
    
    print(f"Forward pass: {x.shape} -> {out.shape}")
    
    # Create standalone attention block
    attn_block = FlashSTUBlock(
        d_model=256,
        sequence_length=512,
        use_attention=True,
        attention_config={'n_heads': 8, 'softcap': 50.0}
    ).cuda()
    
    print(f"\nAttention Block created")
    print(f"  - Is attention block: {attn_block.is_attention_block}")
    print(f"  - Num parameters: {attn_block.get_num_params():,}")
    
    out = attn_block(x)
    print(f"Forward pass: {x.shape} -> {out.shape}")
    print()


def example_5_huggingface_compatibility():
    """Example 5: HuggingFace compatibility features"""
    print("=" * 60)
    print("Example 5: HuggingFace Compatibility")
    print("=" * 60)
    
    config = FlashSTUConfig(
        n_embd=256,
        n_layers=4,
        seq_len=512,
        vocab_size=1000,
        torch_dtype=torch.bfloat16
    )
    
    model = FlashSTU(config).cuda()
    
    # Feature 1: Return dict vs tuple
    input_ids = torch.randint(0, 1000, (1, 64), device='cuda')
    
    print("Return dict (default):")
    outputs = model(input_ids=input_ids, return_dict=True)
    print(f"  - Type: {type(outputs)}")
    print(f"  - Keys: {outputs.keys() if hasattr(outputs, 'keys') else 'Has attributes: logits, loss, etc.'}")
    
    # Feature 2: Output hidden states
    print("\nOutput hidden states:")
    outputs = model(input_ids=input_ids, output_hidden_states=True)
    if outputs.hidden_states is not None:
        print(f"  - Number of hidden states: {len(outputs.hidden_states)}")
        print(f"  - First hidden state shape: {outputs.hidden_states[0].shape}")
    
    # Feature 3: Resize token embeddings
    print("\nResize token embeddings:")
    print(f"  - Original vocab size: {model.config.vocab_size}")
    model.resize_token_embeddings(1500)
    print(f"  - New vocab size: {model.config.vocab_size}")
    print(f"  - New embedding shape: {model.tok_emb.weight.shape}")
    
    # Test it still works
    input_ids = torch.randint(0, 1500, (1, 64), device='cuda')
    outputs = model(input_ids=input_ids)
    print(f"  - Output logits shape: {outputs.logits.shape}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("FlashSTU Usage Examples")
    print("=" * 60 + "\n")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Please run on a GPU.")
        return
    
    print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")
    
    try:
        example_1_basic_forward()
        example_2_generation()
        example_3_save_and_load()
        example_4_standalone_blocks()
        example_5_huggingface_compatibility()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

