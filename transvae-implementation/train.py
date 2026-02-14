"""
Main training script for TransVAE
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


from transvae import TransVAE, TransVAELoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train TransVAE')
    
    # Model arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--variant', type=str, default='large', choices=['tiny', 'base', 'large', 'huge', 'giant'])
    parser.add_argument('--compression_ratio', type=int, default=16, choices=[8, 16])
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=False, help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'coco'])
    parser.add_argument('--resolution', type=int, default=256, help='Training resolution')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder (for stage 2)')
    
    # Loss weights
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--lpips_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=1e-8)
    parser.add_argument('--vf_weight', type=float, default=0.1)
    parser.add_argument('--gan_weight', type=float, default=0.0)
    parser.add_argument('--use_gan', action='store_true', help='Use GAN loss')
    
    # Checkpoint arguments
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=5000, help='Save frequency (steps)')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Evaluation frequency (steps)')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    
    # Optimization
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(args, config):
    """Create TransVAE model"""
    model_config = config.get('model', {})
    
    # model = TransVAE(
    #     variant=args.variant,
    #     compression_ratio=args.compression_ratio,
    #     latent_dim=args.latent_dim,
    #     use_rope=model_config.get('use_rope', True),
    #     use_conv_ffn=model_config.get('use_conv_ffn', True),
    #     use_dc_path=model_config.get('use_dc_path', True),
    # )

    model = TransVAE(
        config=model_config,
        variant=model_config.get("variant", args.variant),
        compression_ratio=model_config.get("compression_ratio", args.compression_ratio),
        latent_dim=model_config.get("latent_dim", args.latent_dim),
        use_rope=model_config.get('use_rope', True),
        use_conv_ffn=model_config.get('use_conv_ffn', True),
        use_dc_path=model_config.get('use_dc_path', True),
    )
    
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    return model


def create_dataloader_old(args, rank, world_size):
    """Create data loader"""
    # from datasets import load_dataset

    # ds = load_dataset("evanarlian/imagenet_1k_resized_256")
    # Import dataset here to avoid circular imports
    if args.dataset == 'imagenet':
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    # Distributed sampler
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = None
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader, sampler


def create_dataloader(args, rank, world_size):
    """Create dataloader with optional streaming support"""

    # -----------------------
    # Transform
    # -----------------------
    transform_list = [
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        # Add Normalization here if needed, e.g.:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # We create a specific transform for local files that handles RGB conversion
    # inside the Compose (ImageFolder usually loads RGB, but to be safe):
    local_transform = transforms.Compose(transform_list)

    # -----------------------
    # Load dataset
    # -----------------------
    if args.dataset == "imagenet":

        if getattr(args, "hf_dataset", True):
            # Use HuggingFace dataset
            ds = load_dataset(
                "evanarlian/imagenet_1k_resized_256",
                split="train",
                streaming=getattr(args, "streaming", False)
            )

            # Apply transform
            def transform_fn(examples):
                # 'examples' is a dict of lists: {'image': [PIL.Image, ...], 'label': [int, ...]}
                
                pixel_values = []
                for image in examples["image"]:
                    # 1. Convert to RGB to avoid channel errors
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    # 2. Apply transforms
                    # We apply the list manually or use a Compose without the RGB check 
                    # since we did it above.
                    trans = transforms.Compose(transform_list)
                    pixel_values.append(trans(image))
                
                # Return dictionary with transformed tensors
                # Do NOT torch.stack() here; the DataLoader collate_fn handles stacking.
                return {"image": pixel_values, "label": examples["label"]}

            ds = ds.with_transform(transform_fn)

            train_dataset = ds

        else:
            # Use local ImageFolder
            from torchvision import datasets

            train_dataset = datasets.ImageFolder(
                os.path.join(args.data_dir, "train"),
                transform=local_transform
            )

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    # -----------------------
    # Distributed Sampler
    # -----------------------
    if world_size > 1 and not getattr(args, "streaming", False):
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # -----------------------
    # DataLoader
    # -----------------------
    # size = int(len(train_dataset) * 0.7)
    size = 840000
    dataloader = DataLoader(
        train_dataset[:size],
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler

def train_epoch(
    model, 
    dataloader, 
    optimizer, 
    loss_fn, 
    scaler,
    epoch, 
    args,
    rank,
    writer=None,
):
    """Train for one epoch"""
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
    total_loss = 0
    step = 0
    
    # CHANGE 1: Don't unpack (images, _) in the loop header. Get the whole batch.
    for batch_idx, batch in enumerate(pbar):
        
        # CHANGE 2: Check if batch is a Dict (HuggingFace) or Tuple (ImageFolder)
        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images, _ = batch

        # Now images is a Tensor, so .cuda() (or .to(device)) works
        images = images.to(args.device, non_blocking=True)
        
        # Forward pass
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                reconstruction, mu, logvar = model(images)
                losses = loss_fn(reconstruction, images, mu, logvar)
                loss = losses['total']
        else:
            reconstruction, mu, logvar = model(images)
            losses = loss_fn(reconstruction, images, mu, logvar)
            loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        
        if args.mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Logging
        total_loss += loss.item()
        step += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l1': f'{losses["l1"].item():.4f}',
                'lpips': f'{losses["lpips"].item():.4f}',
            })
            
            if writer is not None and batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                for name, value in losses.items():
                    writer.add_scalar(f'train/{name}', value.item(), global_step)
    
    avg_loss = total_loss / step
    return avg_loss


def save_checkpoint(model, optimizer, epoch, step, save_path, args):
    """Save model checkpoint"""
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print(f"Creating TransVAE-{args.variant} model...")
    model = create_model(args, config)
    model = model.cuda()
    
    if rank == 0:
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Freeze encoder if specified (for stage 2 training)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for stage 2 training")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    
    # Create loss function
    loss_fn = TransVAELoss(
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        kl_weight=args.kl_weight,
        vf_weight=args.vf_weight,
        gan_weight=args.gan_weight,
        use_gan=args.use_gan,
    ).cuda()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader, sampler = create_dataloader(args, rank, world_size)
    
    # Tensorboard writer
    writer = SummaryWriter(args.output_dir) if rank == 0 else None
    
    # Training loop
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    print(f"Starting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(
            model, dataloader, optimizer, loss_fn, scaler,
            epoch, args, rank, writer
        )
        
        if rank == 0:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
                save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
                save_checkpoint(model, optimizer, epoch, 0, save_path, args)
    
    if rank == 0 and writer is not None:
        writer.close()
    
    print("Training complete!")


if __name__ == '__main__':
    import torch, gc

    gc.collect()
    torch.cuda.empty_cache()

    main()
