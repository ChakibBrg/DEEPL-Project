"""
Visualize TransVAE reconstructions (5 examples) from a checkpoint.

Usage (HF val):
python3 viz_transvae_5.py --checkpoint out/checkpoint_epoch9.pth --config configs/transvae_base_f16d32.yaml \
  --hf_dataset --split val --resolution 256 --out_png recon_grid.png

Usage (local ImageFolder):
python3 viz_transvae_5.py --checkpoint out/checkpoint_epoch9.pth --config configs/transvae_base_f16d32.yaml \
  --data_dir /path/to/imagenet --split val --resolution 256 --out_png recon_grid.png
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from datasets import load_dataset

import yaml

from transvae import TransVAE


def parse_args():
    p = argparse.ArgumentParser("Visualize 5 TransVAE reconstructions")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, required=True)

    # data
    p.add_argument("--hf_dataset", action="store_true")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)

    # output
    p.add_argument("--out_png", type=str, default="transvae_recon_grid.png")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),  # [0,1]
    ])

    if args.hf_dataset:
        ds = load_dataset(
            "evanarlian/imagenet_1k_resized_256",
            split=args.split,
            streaming=args.streaming or False
        )

        def tf(examples):
            px = []
            for im in examples["image"]:
                if im.mode != "RGB":
                    im = im.convert("RGB")
                px.append(transform(im))
            return {"image": px, "label": examples.get("label", [0] * len(px))}

        ds = ds.with_transform(tf)

        # Take a small stream slice so DataLoader has something finite
        ds = ds.take(64)

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    # local ImageFolder
    if not args.data_dir:
        raise ValueError("--data_dir is required when not using --hf_dataset")

    from torchvision import datasets
    dataset = datasets.ImageFolder(os.path.join(args.data_dir, args.split), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def load_model(checkpoint_path: str, config_path: str, device: str):
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})

    model = TransVAE(
        config=model_cfg,
        variant=model_cfg.get("variant", "base"),
        compression_ratio=model_cfg.get("compression_ratio", 16),
        latent_dim=model_cfg.get("latent_dim", 32),
        use_rope=model_cfg.get("use_rope", True),
        use_conv_ffn=model_cfg.get("use_conv_ffn", True),
        use_dc_path=model_cfg.get("use_dc_path", True),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"

    print("Loading model...")
    model = load_model(args.checkpoint, args.config, device=device)

    print("Loading data...")
    loader = create_dataloader(args)

    batch = next(iter(loader))
    images = batch["image"] if isinstance(batch, dict) else batch[0]
    images = images.to(device)

    # forward
    recon_logits, _, _ = model(images)
    recon = torch.sigmoid(recon_logits)  # logits -> [0,1]
    recon = recon.clamp(0, 1)

    # Build grid: first row originals, second row reconstructions
    grid = make_grid(
        torch.cat([images.cpu(), recon.cpu()], dim=0),
        nrow=args.batch_size,
        padding=4
    )

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))

    print(f"Saved comparison grid to: {out_path}")
    print("Top row: originals | Bottom row: reconstructions")


if __name__ == "__main__":
    main()
