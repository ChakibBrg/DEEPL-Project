"""
Main training script for TransVAE (NaN-safe + GPU memory friendly)

Key fixes:
- Set PYTORCH_CUDA_ALLOC_CONF BEFORE importing torch
- Proper boolean flags in argparse
- HF streaming IterableDataset with correct sharding across DDP ranks + DataLoader workers
- Normalize images to [-1, 1] (typical for VAE/LPIPS setups)
- Mixed precision: autocast ONLY for model forward, loss computed in FP32 with autocast disabled
- Clamp logvar for stable KL
- Skip non-finite loss steps to avoid corrupting optimizer state
- No torch.cuda.empty_cache() inside the training loop (only between epochs)
- Warmup scheduler (linear) actually used
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import yaml
from datasets import load_dataset

from transvae import TransVAE, TransVAELoss


# ---------------------------
# Args / Config
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train TransVAE")

    # Model arguments
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--variant", type=str, default="large",
                        choices=["tiny", "base", "large", "huge", "giant"])
    parser.add_argument("--compression_ratio", type=int, default=16, choices=[8, 16])
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=False, help="Path to dataset (local)")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "coco"])
    parser.add_argument("--resolution", type=int, default=256, help="Training resolution")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    # Use HF dataset streaming
    parser.add_argument("--hf_dataset", action="store_true", help="Use HuggingFace dataset")
    parser.add_argument("--streaming", action="store_true", help="Stream HF dataset (recommended)")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)  # per paper tokenizer warmup ~1000
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder (for stage 2)")
    parser.add_argument("--accumulation_steps", type=int, default=4)

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-8)
    parser.add_argument("--vf_weight", type=float, default=0.1)
    parser.add_argument("--gan_weight", type=float, default=0.0)
    parser.add_argument("--use_gan", action="store_true", help="Use GAN loss")

    # Checkpoint arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save_every_epochs", type=int, default=5)

    # Distributed training
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=0)

    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")

    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(args, config):
    model_config = config.get("model", {})

    model = TransVAE(
        config=model_config,
        variant=model_config.get("variant", args.variant),
        compression_ratio=model_config.get("compression_ratio", args.compression_ratio),
        latent_dim=model_config.get("latent_dim", args.latent_dim),
        use_rope=model_config.get("use_rope", True),
        use_conv_ffn=model_config.get("use_conv_ffn", True),
        use_dc_path=model_config.get("use_dc_path", True),
    )

    if args.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    return model


# ---------------------------
# Data
# ---------------------------
class HFTransformIterableDataset(torch.utils.data.IterableDataset):
    """
    Streaming HF dataset wrapper that:
    - shards across DDP ranks upstream (before wrapping)
    - shards again across DataLoader workers to avoid duplicate samples per worker
    - applies PIL->Tensor transforms on the fly
    """
    def __init__(self, hf_dataset, transform):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        ds = self.dataset
        if worker is not None:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)

        for sample in ds:
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.transform(img)
            yield img, sample.get("label", 0)


def create_dataloader(args, rank, world_size):
    # Normalize to [-1, 1] (often expected for VAE + LPIPS pipelines)
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [0,1] -> [-1,1]
    ])

    if args.dataset != "imagenet":
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    if args.hf_dataset:
        ds = load_dataset(
            "evanarlian/imagenet_1k_resized_256",
            split="train",
            streaming=args.streaming or True,  # strongly recommended
        )

        # DDP rank sharding for streaming
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)

        # shuffle stream with buffer
        ds = ds.shuffle(seed=42, buffer_size=10_000)

        train_dataset = HFTransformIterableDataset(ds, transform)

        # For iterable datasets, do NOT use DistributedSampler / shuffle in DataLoader
        sampler = None
        shuffle = False

        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=False,  # safer for iterable streams
        )
        return dataloader, sampler

    else:
        # Local ImageFolder
        from torchvision import datasets

        if not args.data_dir:
            raise ValueError("--data_dir is required when not using --hf_dataset")

        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, "train"),
            transform=transform,
        )

        if world_size > 1:
            sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        return dataloader, sampler


# ---------------------------
# Checkpointing
# ---------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, save_path, args):
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved to {save_path}")


# ---------------------------
# Training
# ---------------------------
def make_scheduler(args, optimizer):
    # Linear warmup then constant
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    scaler,
    epoch,
    args,
    rank,
    writer=None,
    global_step_offset=0,
):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))

    total_loss = 0.0
    step = 0
    accum = max(1, args.accumulation_steps)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        images, _ = batch
        images = images.to(args.device, non_blocking=True)

        # Forward in mixed precision (model only)
        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                reconstruction, mu, logvar = model(images)
        else:
            reconstruction, mu, logvar = model(images)

        # Loss in FP32 with autocast disabled (LPIPS/KL stability)
        with torch.cuda.amp.autocast(enabled=False):
            rec32 = reconstruction.float()
            img32 = images.float()
            mu32 = mu.float()
            logvar32 = logvar.float().clamp(-30.0, 20.0)

            # keep ranges stable (useful if loss expects [-1,1])
            rec32 = rec32.clamp(-1.0, 1.0)
            img32 = img32.clamp(-1.0, 1.0)

            losses = loss_fn(rec32, img32, mu32, logvar32)
            loss = losses["total"] / accum

        # Skip non-finite
        if not torch.isfinite(loss):
            if rank == 0:
                print("⚠️ Non-finite loss detected. Skipping step.")
                print("finite(recon/mu/logvar):",
                      torch.isfinite(reconstruction).float().mean().item(),
                      torch.isfinite(mu).float().mean().item(),
                      torch.isfinite(logvar).float().mean().item())
            optimizer.zero_grad(set_to_none=True)
            del reconstruction, mu, logvar, losses, loss
            continue

        # Backward
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Free big tensors ASAP
        del reconstruction, mu, logvar

        # Step on accumulation boundary
        do_step = ((batch_idx + 1) % accum == 0)
        if do_step:
            if args.mixed_precision:
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        # Logging
        total_loss += (loss.detach().item() * accum)
        step += 1

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.detach().item() * accum:.4f}",
                "l1": f"{losses['l1'].detach().item():.4f}" if "l1" in losses else "n/a",
                "lpips": f"{losses['lpips'].detach().item():.4f}" if "lpips" in losses else "n/a",
            })

            if writer is not None and (batch_idx % 100 == 0):
                global_step = global_step_offset + batch_idx
                for name, value in losses.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"train/{name}", value.detach().item(), global_step)

        del losses, loss

    # If epoch ends mid-accumulation, flush remaining gradients
    if (batch_idx + 1) % accum != 0:
        if args.mixed_precision:
            scaler.unscale_(optimizer)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / max(1, step)
    return avg_loss, step


# ---------------------------
# Main
# ---------------------------
def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = parse_args()

    rank, world_size, local_rank = setup_distributed()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        print(f"Creating TransVAE-{config.get('variant', args.variant)} "
              f"-f{config.get('compression_ratio', args.compression_ratio)}"
              f"d{config.get('latent_dim', args.latent_dim)} model...")

    model = create_model(args, config).to(args.device)

    if rank == 0 and hasattr(model, "get_num_params"):
        print(f"Model parameters: {model.get_num_params()}")
        if args.device == "cuda":
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if args.freeze_encoder:
        enc = model.module.encoder if isinstance(model, DDP) else model.encoder
        for p in enc.parameters():
            p.requires_grad = False
        if rank == 0:
            print("Encoder frozen for stage 2 training")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True if (args.device == "cuda" and hasattr(torch.optim.AdamW, "fused")) else False,
    )

    scheduler = make_scheduler(args, optimizer)

    loss_fn = TransVAELoss(
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        kl_weight=args.kl_weight,
        vf_weight=args.vf_weight,
        gan_weight=args.gan_weight,
        use_gan=args.use_gan,
    ).to(args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and args.device == "cuda")

    if rank == 0:
        print("Creating dataloader...")
    dataloader, sampler = create_dataloader(args, rank, world_size)

    writer = SummaryWriter(args.output_dir) if rank == 0 else None

    start_epoch = 0
    global_step_offset = 0

    if args.checkpoint:
        if rank == 0:
            print(f"Loading checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)

        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        global_step_offset = ckpt.get("global_step", 0)

        del ckpt
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        print(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss, num_steps = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            epoch=epoch,
            args=args,
            rank=rank,
            writer=writer,
            global_step_offset=global_step_offset,
        )

        global_step_offset += num_steps

        if rank == 0:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            if args.device == "cuda":
                print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

            if ((epoch + 1) % args.save_every_epochs == 0) or ((epoch + 1) == args.num_epochs):
                save_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step_offset, save_path, args)

        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()

    if rank == 0:
        print("Training complete!")


if __name__ == "__main__":
    main()
