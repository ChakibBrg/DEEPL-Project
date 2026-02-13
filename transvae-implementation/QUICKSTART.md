# TransVAE Quick Start Guide

This guide will help you get started with TransVAE in under 10 minutes.

## Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/your-repo/transvae-implementation.git
cd transvae-implementation

# Create environment
conda create -n transvae python=3.9
conda activate transvae

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Test (2 minutes)

Test that everything works:

```python
import torch
from transvae import TransVAE

# Create model
model = TransVAE(variant='tiny', compression_ratio=16, latent_dim=32)

# Test forward pass
x = torch.randn(1, 3, 256, 256)
reconstruction, mu, logvar = model(x)

print(f"✓ Input: {x.shape}")
print(f"✓ Reconstruction: {reconstruction.shape}")
print(f"✓ Latent: {mu.shape}")
print("Success! TransVAE is working correctly.")
```

## Train Your First Model (3 minutes setup)

### 1. Prepare Data

For ImageNet:
```bash
# Download ImageNet (you'll need credentials)
cd data/
bash download_imagenet.sh

# Or use a small subset for testing
python scripts/create_tiny_imagenet.py --output_dir data/tiny_imagenet
```

### 2. Start Training

```bash
# Train TransVAE-Tiny (44M params) on 1 GPU
python train.py \
    --config configs/transvae_tiny_f16d32.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/my_first_transvae \
    --batch_size 32 \
    --num_epochs 5
```

### 3. Monitor Training

```bash
# Open tensorboard in another terminal
tensorboard --logdir ./checkpoints/my_first_transvae
```

Visit http://localhost:6006 to see training curves.

## Run Inference

```python
import torch
from transvae import TransVAE
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = TransVAE.from_pretrained('transvae-large-f16d32')
model.eval()

# Load image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image = Image.open('example.jpg')
x = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    reconstruction, mu, logvar = model(x)

# Save
transforms.ToPILImage()(reconstruction[0]).save('output.jpg')
```

## Common Tasks

### Evaluate Reconstruction Quality

```bash
python evaluate.py \
    --checkpoint ./checkpoints/my_first_transvae/best.pth \
    --data_dir ./data/imagenet \
    --resolution 256 \
    --metrics psnr ssim lpips
```

### Test Resolution Extrapolation

```bash
python scripts/reproduce/test_rope_extrapolation.py \
    --checkpoint ./checkpoints/my_first_transvae/best.pth \
    --data_dir ./data/imagenet \
    --test_resolutions 256 512 1024
```

### Scale to Multiple GPUs

```bash
# Distributed training on 8 GPUs
torchrun --nproc_per_node=8 train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/transvae_large \
    --distributed
```

## Next Steps

- 📚 Read the [full README](README.md) for detailed documentation
- 🔬 Reproduce paper results with scripts in `scripts/reproduce/`
- 🎯 Fine-tune on your custom dataset
- 📊 Experiment with different model sizes and configurations

## Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size or enable gradient checkpointing
  ```bash
  python train.py ... --batch_size 16 --gradient_checkpointing
  ```

**Issue**: Slow training
- **Solution**: Use mixed precision training
  ```bash
  python train.py ... --mixed_precision
  ```

**Issue**: Poor reconstruction quality
- **Solution**: Train longer or scale up the model
  ```bash
  python train.py ... --num_epochs 100 --variant large
  ```

## Getting Help

- 📖 Check the [documentation](docs/)
- 💬 Open an issue on GitHub
- 📧 Contact: [email]

Happy training! 🚀
