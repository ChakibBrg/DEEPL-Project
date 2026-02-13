"""
Simple inference example for TransVAE
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from transvae import TransVAE


def load_image(image_path, size=256):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def save_image(tensor, save_path):
    """Save tensor as image"""
    # Remove batch dimension and convert to PIL
    image = tensor.squeeze(0).cpu()
    image = transforms.ToPILImage()(image)
    image.save(save_path)
    print(f"Saved to {save_path}")


def main():
    # Configuration
    checkpoint_path = "checkpoints/transvae_large/best.pth"
    input_image = "example_input.jpg"
    output_image = "example_reconstruction.jpg"
    
    # Load model
    print("Loading TransVAE model...")
    model = TransVAE(
        variant='large',
        compression_ratio=16,
        latent_dim=32,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    print(f"Model loaded. Parameters: {model.get_num_params()}")
    
    # Load input image
    print(f"Loading image from {input_image}...")
    image = load_image(input_image)
    image = image.cuda()
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # Encode
        mu, logvar = model.encode(image)
        print(f"Encoded to latent: {mu.shape}")
        
        # Optionally use mean (no sampling)
        z = mu
        
        # Decode
        reconstruction = model.decode(z)
        print(f"Reconstructed: {reconstruction.shape}")
    
    # Save output
    save_image(reconstruction, output_image)
    
    print("\nDone!")
    print(f"Compression ratio: {image.shape[-1] / mu.shape[-1]:.1f}x")
    print(f"Latent size: {mu.shape[2]}x{mu.shape[3]}x{mu.shape[1]}")


if __name__ == '__main__':
    main()
