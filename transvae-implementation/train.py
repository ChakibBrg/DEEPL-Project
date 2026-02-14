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
import torch, gc



from transvae import TransVAE, TransVAELoss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--streaming', type=bool, default=False)
    
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


# def create_dataloader(args, rank, world_size):
#     """Create dataloader with optional streaming support"""

#     # -----------------------
#     # Transform
#     # -----------------------
#     transform_list = [
#         transforms.Resize(args.resolution),
#         transforms.CenterCrop(args.resolution),
#         transforms.ToTensor(),
#         # Add Normalization here if needed, e.g.:
#         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ]
    
#     # We create a specific transform for local files that handles RGB conversion
#     # inside the Compose (ImageFolder usually loads RGB, but to be safe):
#     local_transform = transforms.Compose(transform_list)

#     # -----------------------
#     # Load dataset
#     # -----------------------
#     if args.dataset == "imagenet":

#         if getattr(args, "hf_dataset", True):
#             # Use HuggingFace dataset
#             ds = load_dataset(
#                 "evanarlian/imagenet_1k_resized_256",
#                 split="train",
#                 streaming=getattr(args, "streaming", False)
#             )

#             # # Apply transform
#             # def transform_fn(examples):
#             #     # 'examples' is a dict of lists: {'image': [PIL.Image, ...], 'label': [int, ...]}
                
#             #     pixel_values = []
#             #     for image in examples["image"]:
#             #         # 1. Convert to RGB to avoid channel errors
#             #         if image.mode != "RGB":
#             #             image = image.convert("RGB")
                    
#             #         # 2. Apply transforms
#             #         # We apply the list manually or use a Compose without the RGB check 
#             #         # since we did it above.
#             #         trans = transforms.Compose(transform_list)
#             #         pixel_values.append(trans(image))
                
#             #     # Return dictionary with transformed tensors
#             #     # Do NOT torch.stack() here; the DataLoader collate_fn handles stacking.
#             #     return {"image": pixel_values, "label": examples["label"]}

#             # ds = ds.with_transform(transform_fn)

#             # train_dataset = ds

#             # --- 1. Distributed Sharding for Streaming ---
#             if world_size > 1:
#                 ds = ds.shard(num_shards=world_size, index=rank)
            
#             # --- 2. Shuffle for Streaming (CRITICAL) ---
#             # DataLoader shuffle=True doesn't work for streams. 
#             # We must shuffle the stream buffer.
#             ds = ds.shuffle(seed=42, buffer_size=10_000)

#             # --- 3. Transform Function ---
#             def transform_fn(examples):
#                 pixel_values = []
#                 for image in examples["image"]:
#                     if image.mode != "RGB":
#                         image = image.convert("RGB")
#                     # Apply transforms
#                     trans = transforms.Compose(transform_list)
#                     pixel_values.append(trans(image))
#                 return {"image": pixel_values, "label": examples["label"]}

#             # --- 4. FIX: Use .map() instead of .with_transform() ---
#             # batched=True makes it efficient (processes batch_size items at once)
#             # but it still yields 1 item at a time to the DataLoader
#             ds = ds.map(transform_fn, batched=True, batch_size=args.batch_size)

#             train_dataset = ds

#         else:
#             # Use local ImageFolder
#             from torchvision import datasets

#             train_dataset = datasets.ImageFolder(
#                 os.path.join(args.data_dir, "train"),
#                 transform=local_transform
#             )

#     else:
#         raise NotImplementedError(f"Dataset {args.dataset} not implemented")

#     # -----------------------
#     # Distributed Sampler
#     # -----------------------
#     # if world_size > 1 and not getattr(args, "streaming", False):
#     #     sampler = DistributedSampler(
#     #         train_dataset,
#     #         num_replicas=world_size,
#     #         rank=rank,
#     #         shuffle=True,
#     #     )
#     #     shuffle = False
#     # else:
#     #     sampler = None
#     #     shuffle = True
#     # Samplers (only for non-streaming)
#     is_streaming = True if getattr(args, "hf_dataset", True) else False
    
#     if world_size > 1 and not is_streaming:
#         sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#         shuffle = False
#     else:
#         sampler = None
#         # Disable DataLoader shuffling if streaming (we did it manually above)
#         shuffle = False if is_streaming else True

#     # -----------------------
#     # DataLoader
#     # -----------------------
#     # size = int(len(train_dataset) * 0.7)
#     size = 840000
#     dataloader = DataLoader(
#         # train_dataset[:size],
#         train_dataset,
#         batch_size=args.batch_size,
#         sampler=sampler,
#         shuffle=shuffle,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#     )

#     return dataloader, sampler

# def train_epoch(
#     model, 
#     dataloader, 
#     optimizer, 
#     loss_fn, 
#     scaler,
#     epoch, 
#     args,
#     rank,
#     writer=None,
# ):
#     """Train for one epoch"""
#     model.train()
    
#     pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
#     total_loss = 0
#     step = 0
    
#     # CHANGE 1: Don't unpack (images, _) in the loop header. Get the whole batch.
#     for batch_idx, batch in enumerate(pbar):
        
#         # CHANGE 2: Check if batch is a Dict (HuggingFace) or Tuple (ImageFolder)
#         if isinstance(batch, dict):
#             images = batch["image"]
#         else:
#             images, _ = batch

#         # Now images is a Tensor, so .cuda() (or .to(device)) works
#         images = images.to(args.device, non_blocking=True)
        
#         # Forward pass
#         if args.mixed_precision:
#             with torch.cuda.amp.autocast():
#                 reconstruction, mu, logvar = model(images)
#                 losses = loss_fn(reconstruction, images, mu, logvar)
#                 loss = losses['total']
#         else:
#             reconstruction, mu, logvar = model(images)
#             losses = loss_fn(reconstruction, images, mu, logvar)
#             loss = losses['total']
        
#         # Backward pass
#         optimizer.zero_grad()
        
#         if args.mixed_precision:
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             if args.grad_clip > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             if args.grad_clip > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optimizer.step()
        
#         # Logging
#         total_loss += loss.item()
#         step += 1
        
#         if rank == 0:
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'l1': f'{losses["l1"].item():.4f}',
#                 'lpips': f'{losses["lpips"].item():.4f}',
#             })
            
#             if writer is not None and batch_idx % 100 == 0:
#                 global_step = epoch * len(dataloader) + batch_idx
#                 for name, value in losses.items():
#                     writer.add_scalar(f'train/{name}', value.item(), global_step)
    
#     avg_loss = total_loss / step
#     return avg_loss


def create_dataloader(args, rank, world_size):
    """Create dataloader with proper memory management"""
    
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])
    
    if args.dataset == "imagenet":
        if getattr(args, "hf_dataset", True):
            ds = load_dataset(
                "evanarlian/imagenet_1k_resized_256",
                split="train",
                streaming=True  # CRITICAL: Always stream for large datasets
            )
            
            if world_size > 1:
                ds = ds.shard(num_shards=world_size, index=rank)
            
            ds = ds.shuffle(seed=42, buffer_size=10_000)
            
            # FIX 1: Use IterableDataset wrapper to transform on-the-fly
            # DON'T use .map() - it caches everything!
            class TransformDataset(torch.utils.data.IterableDataset):
                def __init__(self, hf_dataset, transform):
                    self.dataset = hf_dataset
                    self.transform = transform
                
                def __iter__(self):
                    for sample in self.dataset:
                        image = sample["image"]
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image = self.transform(image)
                        yield image, sample["label"]
            
            train_dataset = TransformDataset(ds, transform)
        else:
            from torchvision import datasets
            train_dataset = datasets.ImageFolder(
                os.path.join(args.data_dir, "train"),
                transform=transform
            )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    is_streaming = getattr(args, "hf_dataset", True)
    
    if world_size > 1 and not is_streaming:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = False if is_streaming else True
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,  # Reduce prefetching
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    return dataloader, sampler


# def train_epoch(
#     model, 
#     dataloader, 
#     optimizer, 
#     loss_fn, 
#     scaler,
#     epoch, 
#     args,
#     rank,
#     writer=None,
# ):
#     """Train for one epoch with proper memory management"""
#     model.train()
    
#     pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
#     total_loss = 0
#     step = 0
    
#     # FIX 2: Add gradient accumulation
#     accumulation_steps = getattr(args, 'accumulation_steps', 1)
    
#     for batch_idx, batch in enumerate(pbar):
        
#         if isinstance(batch, dict):
#             images = batch["image"]
#         else:
#             images, _ = batch
        
#         images = images.to(args.device, non_blocking=True)
        
#         # Forward pass
#         if args.mixed_precision:
#             with torch.cuda.amp.autocast():
#                 reconstruction, mu, logvar = model(images)
#                 losses = loss_fn(reconstruction, images, mu, logvar)
#                 loss = losses['total'] / accumulation_steps  # Scale loss
#         else:
#             reconstruction, mu, logvar = model(images)
#             losses = loss_fn(reconstruction, images, mu, logvar)
#             loss = losses['total'] / accumulation_steps
        
#         # Backward pass
#         if args.mixed_precision:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()
        
#         # FIX 3: Clear intermediate tensors immediately
#         del reconstruction, mu, logvar
        
#         # Optimizer step with accumulation
#         if (batch_idx + 1) % accumulation_steps == 0:
#             if args.mixed_precision:
#                 scaler.unscale_(optimizer)
#                 if args.grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 if args.grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#                 optimizer.step()
            
#             optimizer.zero_grad(set_to_none=True)  # FIX 4: set_to_none=True saves memory
        
#         # Logging
#         total_loss += loss.item() * accumulation_steps
#         step += 1
        
#         # FIX 5: Detach losses before logging to prevent graph accumulation
#         if rank == 0:
#             pbar.set_postfix({
#                 'loss': f'{loss.item() * accumulation_steps:.4f}',
#                 'l1': f'{losses["l1"].item():.4f}',
#                 'lpips': f'{losses["lpips"].item():.4f}',
#             })
            
#             if writer is not None and batch_idx % 100 == 0:
#                 global_step = epoch * len(dataloader) + batch_idx
#                 for name, value in losses.items():
#                     writer.add_scalar(f'train/{name}', value.detach().item(), global_step)
        
#         # FIX 6: Periodic memory cleanup
#         if batch_idx % 50 == 0:
#             torch.cuda.empty_cache()
        
#         # FIX 7: Delete loss dict to free computation graph
#         del losses, loss
    
#     avg_loss = total_loss / step
#     return avg_loss

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
    global_step_offset=0,  # Add this parameter
):
    """Train for one epoch with proper memory management"""
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
    total_loss = 0
    step = 0
    
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    
    for batch_idx, batch in enumerate(pbar):
        
        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images, _ = batch
        
        images = images.to(args.device, non_blocking=True)
        
        # Forward pass
        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                reconstruction, mu, logvar = model(images)
                losses = loss_fn(reconstruction, images, mu, logvar)
                loss = losses['total'] / accumulation_steps
        else:
            reconstruction, mu, logvar = model(images)
            losses = loss_fn(reconstruction, images, mu, logvar)
            loss = losses['total'] / accumulation_steps
        
        # Backward pass
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        del reconstruction, mu, logvar
        
        # Optimizer step with accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if args.mixed_precision:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Logging
        total_loss += loss.item() * accumulation_steps
        step += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'l1': f'{losses["l1"].item():.4f}',
                'lpips': f'{losses["lpips"].item():.4f}',
            })
            
            # FIX: Use global_step_offset instead of len(dataloader)
            if writer is not None and batch_idx % 100 == 0:
                global_step = global_step_offset + batch_idx
                for name, value in losses.items():
                    writer.add_scalar(f'train/{name}', value.detach().item(), global_step)
        
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        del losses, loss
    
    avg_loss = total_loss / step
    # Return both avg_loss and the number of steps for global step tracking
    return avg_loss, step


def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    args = parse_args()
    
    args.accumulation_steps = 4
    args.hf_dataset = True
    args.streaming = True
    
    rank, world_size, local_rank = setup_distributed()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating TransVAE model...")
    model = create_model(args, config)
    model = model.cuda()
    
    if rank == 0:
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], 
                   find_unused_parameters=False)
    
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for stage 2 training")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True if torch.cuda.is_available() else False,
    )
    
    loss_fn = TransVAELoss(
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        kl_weight=args.kl_weight,
        vf_weight=args.vf_weight,
        gan_weight=args.gan_weight,
        use_gan=args.use_gan,
    ).cuda()
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    print("Creating dataloader...")
    dataloader, sampler = create_dataloader(args, rank, world_size)
    
    writer = SummaryWriter(args.output_dir) if rank == 0 else None
    
    start_epoch = 0
    global_step_offset = 0  # Track global steps across epochs
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step_offset = checkpoint.get('global_step', 0)  # Load if available
        del checkpoint
        torch.cuda.empty_cache()
    
    print(f"Starting training from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss, num_steps = train_epoch(
            model, dataloader, optimizer, loss_fn, scaler,
            epoch, args, rank, writer, global_step_offset
        )
        
        # Update global step offset for next epoch
        global_step_offset += num_steps
        
        if rank == 0:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
                save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
                # FIX: Save global_step in checkpoint
                save_checkpoint(model, optimizer, epoch, global_step_offset, save_path, args)
        
        gc.collect()
        torch.cuda.empty_cache()
    
    if rank == 0 and writer is not None:
        writer.close()
    
    print("Training complete!")


def save_checkpoint(model, optimizer, epoch, global_step, save_path, args):
    """Save model checkpoint"""
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,  # Save global step
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


# def save_checkpoint(model, optimizer, epoch, step, save_path, args):
#     """Save model checkpoint"""
#     if isinstance(model, DDP):
#         model_state = model.module.state_dict()
#     else:
#         model_state = model.state_dict()
    
#     checkpoint = {
#         'epoch': epoch,
#         'step': step,
#         'model_state_dict': model_state,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'args': vars(args),
#     }
    
#     torch.save(checkpoint, save_path)
#     print(f"Checkpoint saved to {save_path}")


# def main():
#     gc.collect()
#     torch.cuda.empty_cache()
#     # torch.cuda.reset_peak_memory_stats()

#     args = parse_args()
    
#     # Setup distributed training
#     rank, world_size, local_rank = setup_distributed()
    
#     # Load config
#     config = load_config(args.config)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Create model
#     print(f"Creating TransVAE-{config.get('variant', args.variant)}-f{config.get('compression_ratio', args.compression_ratio)}d{config.get('latent_dim', args.latent_dim)} model...")
#     model = create_model(args, config)
#     model = model.cuda()
    
#     if rank == 0:
#         num_params = model.get_num_params()
#         print(f"Model parameters: {num_params}")
    
#     # Wrap with DDP
#     if world_size > 1:
#         model = DDP(model, device_ids=[local_rank])
    
#     # Freeze encoder if specified (for stage 2 training)
#     if args.freeze_encoder:
#         for param in model.encoder.parameters():
#             param.requires_grad = False
#         print("Encoder frozen for stage 2 training")
    
#     # Create optimizer
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.learning_rate,
#         betas=(0.9, 0.95),
#         weight_decay=0.0,
#     )
    
#     # Create loss function
#     loss_fn = TransVAELoss(
#         l1_weight=args.l1_weight,
#         lpips_weight=args.lpips_weight,
#         kl_weight=args.kl_weight,
#         vf_weight=args.vf_weight,
#         gan_weight=args.gan_weight,
#         use_gan=args.use_gan,
#     ).cuda()
    
#     # Mixed precision scaler
#     scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
#     # Create dataloader
#     print("Creating dataloader...")
#     dataloader, sampler = create_dataloader(args, rank, world_size)
    
#     # Tensorboard writer
#     writer = SummaryWriter(args.output_dir) if rank == 0 else None
    
#     # Training loop
#     start_epoch = 0
#     if args.checkpoint:
#         print(f"Loading checkpoint from {args.checkpoint}")
#         checkpoint = torch.load(args.checkpoint, map_location='cuda')
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
    
#     print(f"Starting training from epoch {start_epoch}...")
    
#     for epoch in range(start_epoch, args.num_epochs):
#         if sampler is not None:
#             sampler.set_epoch(epoch)
        
#         avg_loss = train_epoch(
#             model, dataloader, optimizer, loss_fn, scaler,
#             epoch, args, rank, writer
#         )
        
#         if rank == 0:
#             print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            
#             # Save checkpoint
#             if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
#                 save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
#                 save_checkpoint(model, optimizer, epoch, 0, save_path, args)
        
#         # Periodic memory clean
#         gc.collect()
#         torch.cuda.empty_cache()
    
#     if rank == 0 and writer is not None:
#         writer.close()
    
#     print("Training complete!")


# def main():
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     args = parse_args()
    
#     # FIX 8: Add these arguments
#     args.accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
#     args.hf_dataset = True
#     args.streaming = True
    
#     rank, world_size, local_rank = setup_distributed()
#     config = load_config(args.config)
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     print(f"Creating TransVAE model...")
#     model = create_model(args, config)
#     model = model.cuda()
    
#     if rank == 0:
#         num_params = model.get_num_params()
#         print(f"Model parameters: {num_params}")
#         # FIX 9: Print initial memory
#         print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
#     if world_size > 1:
#         model = DDP(model, device_ids=[local_rank], 
#                    find_unused_parameters=False)  # FIX 10: Set to False if all params used
    
#     if args.freeze_encoder:
#         for param in model.encoder.parameters():
#             param.requires_grad = False
#         print("Encoder frozen for stage 2 training")
    
#     # FIX 11: Use fused optimizer for better memory efficiency
#     optimizer = torch.optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.learning_rate,
#         betas=(0.9, 0.95),
#         weight_decay=0.0,
#         fused=True if torch.cuda.is_available() else False,  # Faster, less memory
#     )
    
#     loss_fn = TransVAELoss(
#         l1_weight=args.l1_weight,
#         lpips_weight=args.lpips_weight,
#         kl_weight=args.kl_weight,
#         vf_weight=args.vf_weight,
#         gan_weight=args.gan_weight,
#         use_gan=args.use_gan,
#     ).cuda()
    
#     scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
#     print("Creating dataloader...")
#     dataloader, sampler = create_dataloader(args, rank, world_size)
    
#     writer = SummaryWriter(args.output_dir) if rank == 0 else None
    
#     start_epoch = 0
#     if args.checkpoint:
#         print(f"Loading checkpoint from {args.checkpoint}")
#         checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
#         if isinstance(model, DDP):
#             model.module.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         del checkpoint  # FIX 12: Free checkpoint memory
#         torch.cuda.empty_cache()
    
#     print(f"Starting training from epoch {start_epoch}...")
    
#     for epoch in range(start_epoch, args.num_epochs):
#         if sampler is not None:
#             sampler.set_epoch(epoch)
        
#         avg_loss = train_epoch(
#             model, dataloader, optimizer, loss_fn, scaler,
#             epoch, args, rank, writer
#         )
        
#         if rank == 0:
#             print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
#             print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
#             if (epoch + 1) % 5 == 0 or (epoch + 1) == args.num_epochs:
#                 save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
#                 save_checkpoint(model, optimizer, epoch, 0, save_path, args)
        
#         gc.collect()
#         torch.cuda.empty_cache()
    
#     if rank == 0 and writer is not None:
#         writer.close()
    
#     print("Training complete!")

if __name__ == '__main__':
    main()
