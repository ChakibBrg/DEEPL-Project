"""
Reproduce Figure 1: Early Training Visualization
Compares CNN-VAE, ViT-VAE, and TransVAE during early training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from transvae import TransVAE
# Note: You would need to implement CNN-VAE and ViT-VAE baselines
# from transvae.models.baseline import CNNVAE, ViTVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps', nargs='+', type=int, default=[512, 1500, 6000])
    parser.add_argument('--output_dir', type=str, default='results/early_training')
    return parser.parse_args()


def visualize_reconstruction(model, image, step, model_name):
    """Visualize reconstruction at a given training step"""
    model.eval()
    
    with torch.no_grad():
        reconstruction, _, _ = model(image)
    
    # Convert to numpy
    img_orig = image[0].cpu().numpy().transpose(1, 2, 0)
    img_recon = reconstruction[0].cpu().numpy().transpose(1, 2, 0)
    
    return img_orig, img_recon


def main():
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Get a test image
    test_image = next(iter(dataloader))[0][:1].cuda()  # Single image
    
    # Initialize models
    print("Initializing models...")
    models = {
        'TransVAE': TransVAE(variant='large', compression_ratio=16, latent_dim=32).cuda(),
        # 'CNN-VAE': CNNVAE(...).cuda(),  # Would need implementation
        # 'ViT-VAE': ViTVAE(...).cuda(),  # Would need implementation
    }
    
    # Training loop (simplified)
    print("\nSimulating early training...")
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss()
        
        step = 0
        visualizations = []
        
        for epoch in range(10):  # Few epochs for early training
            for images, _ in dataloader:
                images = images.cuda()
                
                # Forward
                reconstruction, mu, logvar = model(images)
                loss = criterion(reconstruction, images)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Visualize at specific steps
                if step in args.steps:
                    print(f"  Step {step}: Loss = {loss.item():.4f}")
                    img_orig, img_recon = visualize_reconstruction(
                        model, test_image, step, model_name
                    )
                    visualizations.append((step, img_orig, img_recon))
                
                step += 1
                
                if step > max(args.steps):
                    break
            
            if step > max(args.steps):
                break
        
        # Save visualizations
        fig, axes = plt.subplots(2, len(args.steps) + 1, figsize=(4*(len(args.steps)+1), 8))
        
        # Ground truth
        axes[0, 0].imshow(visualizations[0][1])
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Reconstructions at different steps
        for i, (step, _, img_recon) in enumerate(visualizations):
            axes[0, i+1].imshow(img_recon)
            axes[0, i+1].set_title(f'Step {step}')
            axes[0, i+1].axis('off')
            
            # Difference map
            diff = abs(visualizations[0][1] - img_recon)
            axes[1, i+1].imshow(diff)
            axes[1, i+1].set_title(f'Difference')
            axes[1, i+1].axis('off')
        
        plt.suptitle(f'{model_name} Early Training', fontsize=16)
        plt.tight_layout()
        
        save_path = f"{args.output_dir}/{model_name}_early_training.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    
    print("\n" + "="*60)
    print("Key Observations (from paper):")
    print("="*60)
    print("CNN-VAE:")
    print("  ✓ Learns local details (pixel colors) quickly")
    print("  ✗ Struggles with global structure initially")
    print("\nViT-VAE:")
    print("  ✓ Captures global structure early")
    print("  ✗ Blurry details, wrong colors, slow convergence")
    print("\nTransVAE (Ours):")
    print("  ✓ BOTH local details AND global structure")
    print("  ✓ Faster convergence than both baselines")
    print("="*60)


if __name__ == '__main__':
    main()
