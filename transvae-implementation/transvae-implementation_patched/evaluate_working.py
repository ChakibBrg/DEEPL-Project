"""
Evaluation script for TransVAE (validation set + checkpoint sweep)

Key fixes vs original:
- Supports --checkpoint being a single file OR a directory containing many .pth
- Reconstruction from this repo is logits -> apply sigmoid() before metrics
- LPIPS expects inputs in [-1, 1] with images in [0, 1] -> do correct conversion
- Robust checkpoint loading for different saved formats
"""

import argparse
import os
from pathlib import Path
import re
import json
import yaml

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from datasets import load_dataset

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

from transvae import TransVAE


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TransVAE on validation set")

    # checkpoint can be file or directory
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint (.pth) OR a directory containing checkpoints")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save results JSON (for checkpoint sweep)")
    p.add_argument("--config", type=str, required=True)

    # dataset
    p.add_argument("--data_dir", type=str, help="Path to dataset root")
    p.add_argument("--hf_dataset", action="store_true")
    p.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "coco"])
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Evaluate on a random subset of N samples (for quick checks)")
    p.add_argument("--size", type=int, default=20000)


    # metrics
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "fid"], choices=["psnr", "ssim", "lpips", "fid"], help="Metrics to compute")

    # device/precision
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp", action="store_true",
                   help="Use autocast during model forward (not needed usually for eval)")
    
    # distributed
    p.add_argument("--local_rank", type=int, default=0)

    return p.parse_args()


def natural_key(s: str):
    # for nicer sorting: checkpoint_epoch9 < checkpoint_epoch10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_checkpoints(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        ckpts = sorted(list(p.glob("*.pth")) + list(p.glob("*.pt")), key=lambda x: natural_key(x.name))
        if not ckpts:
            raise FileNotFoundError(f"No .pth/.pt checkpoints found in directory: {p}")
        return ckpts
    raise FileNotFoundError(f"Checkpoint path not found: {p}")

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

# def create_dataloader(args):
#     if args.dataset != "imagenet":
#         raise NotImplementedError("Only imagenet ImageFolder is implemented here.")

#     transform = transforms.Compose([
#         transforms.Resize(args.resolution),
#         transforms.CenterCrop(args.resolution),
#         transforms.ToTensor(),  # yields [0,1]
#     ])

#     dataset = datasets.ImageFolder(
#         os.path.join(args.data_dir, args.split),
#         transform=transform,
#     )

#     if args.num_samples is not None:
#         rng = np.random.default_rng(42)
#         indices = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
#         dataset = torch.utils.data.Subset(dataset, indices)

#     loader = DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return loader


def create_dataloader(args, rank=0, world_size=1):
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
                split="val",
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
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def load_model_from_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # common formats:
    # 1) {"model_state_dict": ..., "args": {...}}
    # 2) {"state_dict": ...}
    # 3) raw state_dict
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", None)
        if state is None:
            state = ckpt.get("state_dict", None)
        if state is None and any(k.startswith("encoder.") or k.startswith("decoder.") for k in ckpt.keys()):
            state = ckpt  # raw state dict
    else:
        raise ValueError("Checkpoint is not a dict; unsupported format.")

    # Build model config from checkpoint args if present
    model_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    # Your training code saved args as vars(args), so keys exist.

    # IMPORTANT: Your TransVAE constructor in training used (config=..., variant=..., etc.)
    # But many repos allow variant/compression_ratio/latent_dim only. We keep it compatible:
    print("model args: ", model_args)
    cfg = load_config(model_args.config)
    model_cfg = cfg.get("model", {})

    model = TransVAE(
        config=model_cfg,
        variant=model_args.get("variant", "base"),
        compression_ratio=model_args.get("compression_ratio", 16),
        latent_dim=model_args.get("latent_dim", 32),
        use_rope=model_cfg.get("use_rope", True),
        use_conv_ffn=model_cfg.get("use_conv_ffn", True),
        use_dc_path=model_cfg.get("use_dc_path", True),
    ).to(device)

    # If keys are prefixed with "module." from DDP, strip them
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    model.eval()

    return model, missing, unexpected


@torch.no_grad()
def evaluate_one_checkpoint(model, dataloader, metrics, device="cuda", use_amp=False):
    metric_values = {m: [] for m in metrics}

    lpips_fn = None
    if "lpips" in metrics:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    fid = None
    if "fid" in metrics:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        # normalize=True expects inputs in [0,1] float


    amp_dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16

    for images, _ in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)

        # forward
        if use_amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                reconstruction, _, _ = model(images)
        else:
            reconstruction, _, _ = model(images)

        # IMPORTANT: in your repo, reconstruction is logits -> convert to image [0,1]
        recon_img = torch.sigmoid(reconstruction)

        # --- LPIPS (batched GPU) ---
        if "lpips" in metrics:
            # LPIPS wants [-1,1]
            img_orig_norm = images * 2.0 - 1.0
            img_recon_norm = recon_img * 2.0 - 1.0
            # clamp for safety
            img_orig_norm = img_orig_norm.clamp(-1, 1)
            img_recon_norm = img_recon_norm.clamp(-1, 1)

            lp = lpips_fn(img_orig_norm, img_recon_norm)  # [B,1,1,1] or [B,1]
            lp = lp.view(lp.shape[0]).detach().cpu().numpy()
            metric_values["lpips"].extend(lp.tolist())

        # --- PSNR/SSIM (CPU per-image) ---
        if ("psnr" in metrics) or ("ssim" in metrics):
            images_np = images.detach().cpu().numpy()
            recon_np = recon_img.detach().cpu().numpy()

            b = images_np.shape[0]
            for i in range(b):
                img_orig = np.transpose(images_np[i], (1, 2, 0))
                img_recon = np.transpose(recon_np[i], (1, 2, 0))

                img_orig = np.clip(img_orig, 0, 1)
                img_recon = np.clip(img_recon, 0, 1)

                if "psnr" in metrics:
                    metric_values["psnr"].append(psnr_metric(img_orig, img_recon, data_range=1.0))

                if "ssim" in metrics:
                    metric_values["ssim"].append(
                        ssim_metric(img_orig, img_recon, data_range=1.0, channel_axis=2)
                    )

        if "fid" in metrics:
            # FID expects 3-channel images. If your data is RGB, you're good.
            # It also expects [0,1] floats if normalize=True.
            fid.update(images, real=True)
            fid.update(recon_img, real=False)


    results = {}
    for name, vals in metric_values.items():
        vals = np.asarray(vals, dtype=np.float64)
        results[name] = {
            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "std": float(np.std(vals)) if len(vals) else float("nan"),
            "median": float(np.median(vals)) if len(vals) else float("nan"),
            "n": int(len(vals)),
        }
    if "fid" in metrics:
        fid_value = fid.compute().item()
        results["fid"] = {"mean": fid_value, "std": 0.0, "median": fid_value, "n": len(dataloader.dataset)}
    return results


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    ckpts = list_checkpoints(args.checkpoint)

    print("Creating dataloader...")
    dataloader, _ = create_dataloader(args)
    ds_size = len(dataloader.dataset) if hasattr(dataloader.dataset, "__len__") else None
    print(f"Dataset: {args.dataset} split={args.split}  resolution={args.resolution}  size={ds_size}")

    all_results = []

    for ckpt_path in ckpts:
        print("\n" + "=" * 80)
        print(f"Checkpoint: {ckpt_path}")

        model, missing, unexpected = load_model_from_checkpoint(ckpt_path, device=device)
        if missing:
            print(f"⚠️ Missing keys (showing up to 10): {missing[:10]}")
        if unexpected:
            print(f"⚠️ Unexpected keys (showing up to 10): {unexpected[:10]}")

        results = evaluate_one_checkpoint(
            model=model,
            dataloader=dataloader,
            metrics=args.metrics,
            device=device,
            use_amp=args.use_amp,
        )

        print(f"Results ({ckpt_path.name}):")
        for m in args.metrics:
            r = results[m]
            print(f"  {m.upper():5s} | mean={r['mean']:.4f}  std={r['std']:.4f}  median={r['median']:.4f}  n={r['n']}")

        all_results.append({
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "results": results,
        })

        # free VRAM between checkpoints
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Optional JSON output
    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {out_path}")

    # Print best checkpoint summary (by LPIPS if available else PSNR)
    key_metric = "lpips" if "lpips" in args.metrics else ("psnr" if "psnr" in args.metrics else args.metrics[0])
    reverse = True if key_metric == "lpips" else False  # lpips lower is better -> sort ascending (reverse=False)
    # Actually: lpips lower is better, so reverse=False; psnr/ssim higher better -> reverse=True
    reverse = False if key_metric == "lpips" else True

    sorted_res = sorted(all_results, key=lambda x: x["results"][key_metric]["mean"], reverse=reverse)
    best = sorted_res[0]
    print("\n" + "=" * 80)
    print(f"Best checkpoint by {key_metric.upper()}: {best['checkpoint_name']}")
    print(best["results"])


if __name__ == "__main__":
    main()
