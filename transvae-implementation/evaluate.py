"""
Evaluation script for TransVAE
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips

from transvae import TransVAE


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TransVAE')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'coco'])
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--resolution', type=int, default=256, help='Evaluation resolution')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips'], 
                       help='Metrics to compute')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate')
    
    return parser.parse_args()


def create_dataloader(args):
    """Create evaluation dataloader"""
    if args.dataset == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(
            f"{args.data_dir}/{args.split}",
            transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return dataloader


@torch.no_grad()
def evaluate(model, dataloader, metrics, device='cuda'):
    """
    Evaluate model on dataset
    
    Args:
        model: TransVAE model
        dataloader: Data loader
        metrics: List of metrics to compute
        device: Device to use
        
    Returns:
        Dictionary of metric values
    """
    model.eval()
    
    # Initialize metrics
    metric_values = {m: [] for m in metrics}
    
    # Initialize LPIPS if needed
    if 'lpips' in metrics:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        lpips_fn.eval()
    
    for images, _ in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        
        # Forward pass
        reconstruction, _, _ = model(images)
        
        # Move to CPU for metric computation
        images_np = images.cpu().numpy()
        reconstruction_np = reconstruction.cpu().numpy()
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            img_orig = np.transpose(images_np[i], (1, 2, 0))
            img_recon = np.transpose(reconstruction_np[i], (1, 2, 0))
            
            # Clip to [0, 1]
            img_orig = np.clip(img_orig, 0, 1)
            img_recon = np.clip(img_recon, 0, 1)
            
            # PSNR
            if 'psnr' in metrics:
                psnr_val = psnr_metric(img_orig, img_recon, data_range=1.0)
                metric_values['psnr'].append(psnr_val)
            
            # SSIM
            if 'ssim' in metrics:
                ssim_val = ssim_metric(
                    img_orig, img_recon,
                    data_range=1.0,
                    channel_axis=2,
                )
                metric_values['ssim'].append(ssim_val)
        
        # LPIPS (batch computation)
        if 'lpips' in metrics:
            # Normalize to [-1, 1]
            img_orig_norm = images * 2 - 1
            img_recon_norm = reconstruction * 2 - 1
            
            lpips_val = lpips_fn(img_orig_norm, img_recon_norm)
            metric_values['lpips'].extend(lpips_val.squeeze().cpu().numpy().tolist())
    
    # Compute mean values
    results = {}
    for metric_name, values in metric_values.items():
        results[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
        }
    
    return results


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    
    # Extract model configuration from checkpoint
    model_args = checkpoint.get('args', {})
    
    model = TransVAE(
        variant=model_args.get('variant', 'large'),
        compression_ratio=model_args.get('compression_ratio', 16),
        latent_dim=model_args.get('latent_dim', 32),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    print(f"Model loaded. Parameters: {model.get_num_params()}")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_dataloader(args)
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Evaluate
    print(f"Evaluating on {args.dataset} {args.split} set at {args.resolution}x{args.resolution}...")
    results = evaluate(model, dataloader, args.metrics)
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results ({args.dataset} {args.split}, {args.resolution}x{args.resolution})")
    print("="*60)
    
    for metric_name, values in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:   {values['mean']:.4f}")
        print(f"  Std:    {values['std']:.4f}")
        print(f"  Median: {values['median']:.4f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
