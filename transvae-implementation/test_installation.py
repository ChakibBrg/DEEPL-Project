"""
Test script to verify TransVAE installation and basic functionality
"""

import torch
import sys
from transvae import TransVAE


def test_model_creation():
    """Test that models can be created"""
    print("Test 1: Model Creation")
    print("-" * 60)
    
    variants = ['tiny', 'base', 'large']
    
    for variant in variants:
        try:
            model = TransVAE(variant=variant, compression_ratio=16, latent_dim=32)
            params = model.get_num_params()
            print(f"✓ {variant.upper()}: {params['total']:,} parameters")
        except Exception as e:
            print(f"✗ {variant.upper()}: Failed - {e}")
            return False
    
    print()
    return True


def test_forward_pass():
    """Test forward pass"""
    print("Test 2: Forward Pass")
    print("-" * 60)
    
    model = TransVAE(variant='tiny', compression_ratio=16, latent_dim=32)
    
    try:
        x = torch.randn(2, 3, 256, 256)
        reconstruction, mu, logvar = model(x)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Reconstruction shape: {reconstruction.shape}")
        print(f"✓ Latent mu shape: {mu.shape}")
        print(f"✓ Latent logvar shape: {logvar.shape}")
        
        # Verify shapes
        assert reconstruction.shape == x.shape, "Reconstruction shape mismatch"
        assert mu.shape[2:] == (16, 16), "Latent spatial size incorrect"
        assert mu.shape[1] == 32, "Latent dimension incorrect"
        
        print("✓ All shapes correct!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print()
        return False


def test_encode_decode():
    """Test separate encode/decode"""
    print("Test 3: Encode/Decode Separately")
    print("-" * 60)
    
    model = TransVAE(variant='tiny', compression_ratio=16, latent_dim=32)
    
    try:
        x = torch.randn(2, 3, 256, 256)
        
        # Encode
        mu, logvar = model.encode(x)
        print(f"✓ Encoded to latent: {mu.shape}")
        
        # Decode
        reconstruction = model.decode(mu)
        print(f"✓ Decoded to image: {reconstruction.shape}")
        
        assert reconstruction.shape == x.shape, "Shape mismatch"
        print("✓ Encode/decode works correctly!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Encode/decode failed: {e}")
        print()
        return False


def test_resolution_flexibility():
    """Test that model works with different resolutions"""
    print("Test 4: Resolution Flexibility (RoPE)")
    print("-" * 60)
    
    model = TransVAE(variant='tiny', compression_ratio=16, latent_dim=32)
    
    resolutions = [128, 256, 512]
    
    try:
        for res in resolutions:
            x = torch.randn(1, 3, res, res)
            reconstruction, _, _ = model(x)
            print(f"✓ {res}×{res}: {x.shape} -> {reconstruction.shape}")
            assert reconstruction.shape == x.shape
        
        print("✓ Model handles arbitrary resolutions!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Resolution flexibility failed: {e}")
        print()
        return False


def test_gradient_checkpointing():
    """Test gradient checkpointing"""
    print("Test 5: Gradient Checkpointing")
    print("-" * 60)
    
    model = TransVAE(variant='tiny', compression_ratio=16, latent_dim=32)
    
    try:
        model.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled")
        
        x = torch.randn(2, 3, 256, 256)
        reconstruction, mu, logvar = model(x)
        
        # Compute loss and backward
        loss = torch.nn.functional.mse_loss(reconstruction, x)
        loss.backward()
        
        print("✓ Backward pass with checkpointing works!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Gradient checkpointing failed: {e}")
        print()
        return False


def test_compression_ratios():
    """Test different compression ratios"""
    print("Test 6: Compression Ratios")
    print("-" * 60)
    
    ratios = [8, 16]
    
    try:
        for ratio in ratios:
            model = TransVAE(
                variant='tiny',
                compression_ratio=ratio,
                latent_dim=16 if ratio == 8 else 32
            )
            
            x = torch.randn(1, 3, 256, 256)
            reconstruction, mu, logvar = model(x)
            
            expected_spatial = 256 // ratio
            actual_spatial = mu.shape[2]
            
            print(f"✓ f{ratio}: Latent {mu.shape}, spatial compression {actual_spatial}×{actual_spatial}")
            assert actual_spatial == expected_spatial, f"Expected {expected_spatial}, got {actual_spatial}"
        
        print("✓ All compression ratios work!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Compression ratios test failed: {e}")
        print()
        return False


def main():
    print("\n" + "="*60)
    print("TransVAE Installation Test")
    print("="*60)
    print()
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_encode_decode,
        test_resolution_flexibility,
        test_gradient_checkpointing,
        test_compression_ratios,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! TransVAE is ready to use.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check your installation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
