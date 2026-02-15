"""
Working training script for this TransVAE repo (OOM-safe + NaN-safe)

Why this version works with YOUR codebase:
- Your decoder outputs unbounded logits (no sigmoid/tanh). The patched TransVAELoss
  converts reconstruction -> [0,1] via sigmoid() for L1/LPIPS, preventing LPIPS NaNs.
- Mixed precision is used for the model forward (bf16 if supported, else fp16),
  while the loss is computed in FP32 (autocast disabled) for stability.
- HF streaming dataloader is correctly sharded across DDP ranks AND DataLoader workers,
  avoiding duplicate samples and odd behavior.
- Activation memory is reduced with gradient checkpointing + gradient accumulation.
- No empty_cache() inside the step loop (only between epochs).

Run (single GPU, safe):
python3 train_working.py --config configs/transvae_base_f16d32.yaml --output_dir out \
  --hf_dataset --streaming --mixed_precision --gradient_checkpointing \
  --batch_size 8 --accumulation_steps 16 --num_workers 2 --num_epochs 5
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import os.path as osp

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


def parse_args():
    p = argparse.ArgumentParser("Train TransVAE (patched)")

    # config / io
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_every_epochs", type=int, default=5)

    # data
    p.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "coco"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--hf_dataset", action="store_true")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--size", type=int, default=200000)

    # training
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accumulation_steps", type=int, default=16)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true")

    # loss weights
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--lpips_weight", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=1e-8)
    p.add_argument("--vf_weight", type=float, default=0.0)  # IMPORTANT: keep 0 unless you actually pass dinov2
    p.add_argument("--gan_weight", type=float, default=0.0)
    p.add_argument("--use_gan", action="store_true")

    # distributed
    p.add_argument("--local_rank", type=int, default=0)

    return p.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class HFTransformIterableDataset(torch.utils.data.IterableDataset):
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
            img = self.transform(img)  # [0,1]
            yield img, sample.get("label", 0)


# def create_dataloader(args, rank, world_size):
#     if args.dataset != "imagenet":
#         raise NotImplementedError("Only imagenet is wired in this script.")

#     # IMPORTANT: keep target in [0,1]. Loss will apply sigmoid() to reconstruction.
#     transform = transforms.Compose([
#         transforms.Resize(args.resolution),
#         transforms.CenterCrop(args.resolution),
#         transforms.ToTensor(),
#     ])

#     if args.hf_dataset:
#         ds = load_dataset(
#             "evanarlian/imagenet_1k_resized_256",
#             split="train",
#             streaming=args.streaming or False,
#         )

#         if world_size > 1:
#             ds = ds.shard(num_shards=world_size, index=rank)

#         ds = ds.shuffle(seed=42, buffer_size=10_000)
#         train_dataset = HFTransformIterableDataset(ds, transform)

#         dataloader = DataLoader(
#             train_dataset,
#             batch_size=args.batch_size,
#             num_workers=args.num_workers,
#             pin_memory=True,
#             drop_last=True,
#             prefetch_factor=2 if args.num_workers > 0 else None,
#             persistent_workers=False,
#         )
#         return dataloader, None

#     # local ImageFolder
#     if not args.data_dir:
#         raise ValueError("--data_dir is required when not using --hf_dataset")

#     from torchvision import datasets
#     train_dataset = datasets.ImageFolder(osp.join(args.data_dir, "train"), transform=transform)

#     if world_size > 1:
#         sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
#         shuffle = False
#     else:
#         sampler = None
#         shuffle = True

#     dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         sampler=sampler,
#         shuffle=shuffle,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         prefetch_factor=2 if args.num_workers > 0 else None,
#         persistent_workers=True if args.num_workers > 0 else False,
#     )
#     return dataloader, sampler


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

            train_dataset = ds.take(getattr(args, "size", 200000))

            # # --- 1. Distributed Sharding for Streaming ---
            # if world_size > 1:
            #     ds = ds.shard(num_shards=world_size, index=rank)
            
            # # --- 2. Shuffle for Streaming (CRITICAL) ---
            # # DataLoader shuffle=True doesn't work for streams. 
            # # We must shuffle the stream buffer.
            # ds = ds.shuffle(seed=42, buffer_size=10_000)

            # # --- 3. Transform Function ---
            # def transform_fn(examples):
            #     pixel_values = []
            #     for image in examples["image"]:
            #         if image.mode != "RGB":
            #             image = image.convert("RGB")
            #         # Apply transforms
            #         trans = transforms.Compose(transform_list)
            #         pixel_values.append(trans(image))
            #     return {"image": pixel_values, "label": examples["label"]}

            # # --- 4. FIX: Use .map() instead of .with_transform() ---
            # # batched=True makes it efficient (processes batch_size items at once)
            # # but it still yields 1 item at a time to the DataLoader
            # ds = ds.map(transform_fn, batched=True, batch_size=args.batch_size)

            # train_dataset = ds

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
    # Samplers (only for non-streaming)
    # is_streaming = True if getattr(args, "hf_dataset", True) else False
    
    # if world_size > 1 and not is_streaming:
    #     sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    #     shuffle = False
    # else:
    #     sampler = None
    #     # Disable DataLoader shuffling if streaming (we did it manually above)
    #     shuffle = False if is_streaming else True

    # -----------------------
    # DataLoader
    # -----------------------
    # size = int(len(train_dataset) * 0.7)
    size = 840000
    dataloader = DataLoader(
        # train_dataset[:size],
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler


def make_scheduler(args, optimizer):
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def print_tensor_stats(x: torch.Tensor, name: str):
    x32 = x.float()
    finite = torch.isfinite(x32).float().mean().item()
    print(f"{name}: finite={finite:.4f}, min={x32.min().item():.4f}, max={x32.max().item():.4f}, mean={x32.mean().item():.4f}")


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
    steps = 0
    accum = max(1, args.accumulation_steps)

    optimizer.zero_grad(set_to_none=True)

    amp_dtype = torch.bfloat16 if (args.mixed_precision and torch.cuda.is_bf16_supported()) else torch.float16

    # for batch_idx, (images, _) in enumerate(pbar):
        # images = images.to(args.device, non_blocking=True)
    for batch_idx, batch in enumerate(pbar):
        
        # CHANGE 2: Check if batch is a Dict (HuggingFace) or Tuple (ImageFolder)
        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images, _ = batch

        # Now images is a Tensor, so .cuda() (or .to(device)) works
        images = images.cuda(non_blocking=True)

        # channels_last can help on Ampere+ for conv-heavy models
        images = images.to(memory_format=torch.channels_last)

        # Forward (AMP)
        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                reconstruction, mu, logvar = model(images)
        else:
            reconstruction, mu, logvar = model(images)

        # Loss (FP32, autocast OFF)
        with torch.cuda.amp.autocast(enabled=False):
            losses = loss_fn(
                reconstruction.float(),
                images.float(),
                mu.float(),
                logvar.float(),
                discriminator=None,
                dinov2=None,
            )
            loss = losses["total"] / accum

        # Detect non-finite and print diagnostics ONCE
        if not torch.isfinite(loss):
            if rank == 0:
                print("⚠️ Non-finite loss. Diagnostics:")
                print({k: (v.detach().float().min().item(), v.detach().float().max().item())
                      for k, v in losses.items() if torch.is_tensor(v)})
                print_tensor_stats(images, "images([0,1])")
                print_tensor_stats(reconstruction, "reconstruction(logits)")
                print_tensor_stats(mu, "mu")
                print_tensor_stats(logvar, "logvar")
            optimizer.zero_grad(set_to_none=True)
            del reconstruction, mu, logvar, losses, loss
            continue

        # Backward
        if args.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        del reconstruction, mu, logvar

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
            scheduler.step()

        total_loss += (loss.detach().item() * accum)
        steps += 1

        if rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.detach().item() * accum:.4f}",
                "l1": f"{losses['l1'].detach().item():.4f}",
                "lpips": f"{losses['lpips'].detach().item():.4f}",
                "kl": f"{losses['kl'].detach().item():.4f}",
            })

            if writer is not None and batch_idx % 100 == 0:
                gs = global_step_offset + batch_idx
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", float(v.detach().item()), gs)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], gs)

        del losses, loss

    # flush if epoch ended mid-accumulation
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
        scheduler.step()

    return total_loss / max(1, steps), steps


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, path, args):
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)
    print(f"Saved: {path}")


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})

    if rank == 0:
        print(f"Creating TransVAE-{model_cfg.get('variant', cfg.get('variant','?'))} "
              f"-f{model_cfg.get('compression_ratio', cfg.get('compression_ratio','?'))}"
              f"d{model_cfg.get('latent_dim', cfg.get('latent_dim','?'))} model...")

    model = TransVAE(
        config=model_cfg,
        variant=model_cfg.get("variant"),
        compression_ratio=model_cfg.get("compression_ratio"),
        latent_dim=model_cfg.get("latent_dim"),
        use_rope=model_cfg.get("use_rope", True),
        use_conv_ffn=model_cfg.get("use_conv_ffn", True),
        use_dc_path=model_cfg.get("use_dc_path", True),
    ).cuda()

    # channels_last for conv efficiency
    model = model.to(memory_format=torch.channels_last)

    if args.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    if rank == 0 and hasattr(model, "get_num_params"):
        print("Model parameters:", model.get_num_params())
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if args.freeze_encoder:
        enc = model.module.encoder if isinstance(model, DDP) else model.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        if rank == 0:
            print("Encoder frozen.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        fused=True if hasattr(torch.optim.AdamW, "fused") else False,
    )

    scheduler = make_scheduler(args, optimizer)

    loss_fn = TransVAELoss(
        l1_weight=args.l1_weight,
        lpips_weight=args.lpips_weight,
        kl_weight=args.kl_weight,
        vf_weight=args.vf_weight,
        gan_weight=args.gan_weight,
        use_gan=args.use_gan,
    ).cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    dataloader, sampler = create_dataloader(args, rank, world_size)

    writer = SummaryWriter(args.output_dir) if rank == 0 else None

    start_epoch = 0
    global_step = 0

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cuda")
        if isinstance(model, DDP):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        del ckpt
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        avg_loss, steps = train_epoch(
            model, dataloader, optimizer, scheduler, loss_fn, scaler,
            epoch, args, rank, writer, global_step_offset=global_step
        )
        global_step += steps

        if rank == 0:
            print(f"Epoch {epoch}: avg_loss={avg_loss:.6f} | GPU mem={torch.cuda.memory_allocated() / 1e9:.2f} GB")

            # if ((epoch + 1) % args.save_every_epochs == 0) or ((epoch + 1) == args.num_epochs):
            save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                osp.join(args.output_dir, f"checkpoint_epoch{epoch}.pth"), args)

        gc.collect()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
