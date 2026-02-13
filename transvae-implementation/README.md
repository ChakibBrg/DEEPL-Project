# TransVAE: A Hybrid Paradigm for Vision Autoencoders

Official implementation of "A Hybrid Paradigm for Vision Autoencoders: Unifying CNNs and Transformers for Learning Efficiency and Scalability" (ICLR 2026 submission).

## 📋 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Zoo](#model-zoo)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)

## 🔍 Overview

TransVAE is a hybrid VAE architecture that combines CNN front-ends for local feature extraction with Transformer backbones for global context modeling. This design achieves:

- **Superior Learning Efficiency**: Faster convergence than CNN and ViT baselines
- **True Scalability**: Consistent improvements from 44M to 2.3B parameters
- **Enhanced Extrapolation**: Models trained on 256×256 generalize to arbitrary higher resolutions
- **Unified Representation**: Better alignment of pixel-level and semantic-level features

## 🎯 Key Contributions

1. **Hybrid Architecture**: Unifies CNN front-end with Transformer backbone to capture complete visual representations
2. **Scaling Properties**: 
   - Increasing reconstruction fidelity with model size
   - Enhanced extrapolation to unseen resolutions
   - Better alignment to visual foundation models
3. **Key Components**:
   - RoPE (Rotary Position Embeddings) for resolution generalization
   - Multi-stage architecture with CNN stem
   - Convolutional FFN for local prior enhancement
   - DC path for information preservation

## 🏗️ Architecture

```
Input (H×W×3)
    ↓
[CNN Stage 1] ──→ ResBlocks + Downsample
    ↓
[CNN Stage 2] ──→ ResBlocks + Downsample
    ↓
[TransVAE Stage 3] ──→ TransVAE Blocks + Downsample
    ↓
[TransVAE Stage 4] ──→ TransVAE Blocks + Downsample
    ↓
[TransVAE Stage 5] ──→ TransVAE Blocks
    ↓
Latent (H/f × W/f × d)
    ↓
[Symmetric Decoder]
    ↓
Reconstruction (H×W×3)
```

**TransVAE Block**:
- Flash Attention with RoPE and QKV Normalization
- Convolutional FFN with residual path
- RMSNorm for stability

## 🛠️ Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/transvae-implementation.git
cd transvae-implementation

# Create conda environment
conda create -n transvae python=3.9
conda activate transvae

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### Minimal Example

```python
import torch
from transvae import TransVAE

# Initialize model
model = TransVAE(
    variant='large',
    compression_ratio=16,
    latent_dim=32,
    input_resolution=256
)

# Forward pass
x = torch.randn(4, 3, 256, 256)
reconstruction, mu, logvar = model(x)

print(f"Input shape: {x.shape}")
print(f"Reconstruction shape: {reconstruction.shape}")
print(f"Latent shape: {mu.shape}")
```

### Inference Example

```python
from transvae import TransVAE
from transvae.utils import load_checkpoint, preprocess_image
from PIL import Image

# Load pretrained model
model = TransVAE.from_pretrained('transvae-large-f16d32')
model.eval()

# Load and process image
image = Image.open('example.jpg')
x = preprocess_image(image)

# Encode and decode
with torch.no_grad():
    z = model.encode(x)
    reconstruction = model.decode(z)

# Save result
save_image(reconstruction, 'reconstruction.jpg')
```

## 🏋️ Training

### Dataset Preparation

```bash
# Download ImageNet-1k
cd data/
bash download_imagenet.sh

# Prepare dataset
python scripts/prepare_dataset.py \
    --dataset imagenet \
    --data_dir ./data/imagenet \
    --output_dir ./data/imagenet_processed
```

### Training TransVAE

#### Stage 1: Reconstruction Training (100 epochs)

```bash
python train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet_processed \
    --output_dir ./checkpoints/transvae_large_stage1 \
    --batch_size 256 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --losses l1 lpips kl vf \
    --loss_weights 1.0 1.0 1e-8 0.1 \
    --num_gpus 8
```

#### Stage 2: GAN Refinement (10 epochs)

```bash
python train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet_processed \
    --output_dir ./checkpoints/transvae_large_stage2 \
    --checkpoint ./checkpoints/transvae_large_stage1/best.pth \
    --batch_size 256 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --losses l1 lpips kl vf gan \
    --loss_weights 1.0 1.0 1e-8 0.1 0.05 \
    --freeze_encoder \
    --num_gpus 8
```

### Multi-Resolution Training

```bash
python train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet_processed \
    --output_dir ./checkpoints/transvae_large_multires \
    --batch_size 128 \
    --resolutions 256 512 \
    --num_epochs 50 \
    --num_gpus 8
```

### Distributed Training

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/transvae_large_f16d32.yaml \
    --distributed
```

## 📊 Evaluation

### Image Reconstruction

```bash
# Evaluate on ImageNet validation set
python evaluate.py \
    --checkpoint ./checkpoints/transvae_large/best.pth \
    --dataset imagenet \
    --data_dir ./data/imagenet_processed \
    --resolution 256 \
    --metrics psnr ssim lpips rfid

# Evaluate on COCO validation set
python evaluate.py \
    --checkpoint ./checkpoints/transvae_large/best.pth \
    --dataset coco \
    --data_dir ./data/coco2017 \
    --resolution 512 \
    --metrics psnr ssim lpips rfid
```

### Resolution Extrapolation

```bash
# Test extrapolation: train on 256, test on 512 and 1024
python evaluate_extrapolation.py \
    --checkpoint ./checkpoints/transvae_large/best.pth \
    --train_resolution 256 \
    --test_resolutions 256 512 1024 \
    --dataset imagenet \
    --data_dir ./data/imagenet_processed
```

### Latent Space Analysis

```bash
# Linear probing accuracy
python evaluate_latent.py \
    --checkpoint ./checkpoints/transvae_large/best.pth \
    --task linear_probe \
    --dataset imagenet \
    --num_epochs 100

# Latent space metrics (Density CV, Normalized Entropy, Gini)
python evaluate_latent.py \
    --checkpoint ./checkpoints/transvae_large/best.pth \
    --task metrics \
    --dataset imagenet
```

### Downstream Generation

```bash
# Train LightningDiT with TransVAE latents
python train_dit.py \
    --vae_checkpoint ./checkpoints/transvae_large/best.pth \
    --dit_config configs/lightningdit_xl.yaml \
    --data_dir ./data/imagenet_processed \
    --output_dir ./checkpoints/dit_transvae \
    --num_epochs 80 \
    --batch_size 1024

# Evaluate FID
python evaluate_dit.py \
    --vae_checkpoint ./checkpoints/transvae_large/best.pth \
    --dit_checkpoint ./checkpoints/dit_transvae/best.pth \
    --num_samples 10000 \
    --batch_size 128
```

## 🔬 Model Zoo

### Available Models

| Model | Compression | Latent Dim | Params | ImageNet PSNR | COCO PSNR | Download |
|-------|-------------|------------|--------|---------------|-----------|----------|
| TransVAE-T | f16 | 32 | 44M | 28.38 | 28.11 | [link](#) |
| TransVAE-B | f16 | 32 | 140M | - | - | [link](#) |
| TransVAE-L | f16 | 32 | 545M | 28.92 | 28.69 | [link](#) |
| TransVAE-H | f16 | 32 | 1.3B | - | - | [link](#) |
| TransVAE-G | f16 | 32 | 2.3B | - | - | [link](#) |
| TransVAE-L | f8 | 16 | 719M | 32.89 | 32.85 | [link](#) |

### Loading Pretrained Models

```python
from transvae import TransVAE

# Load from checkpoint
model = TransVAE.from_pretrained('transvae-large-f16d32')

# Or load manually
model = TransVAE(variant='large', compression_ratio=16, latent_dim=32)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔬 Reproducing Paper Results

### Section 3.2.1: Resolution Extrapolation

```bash
# Train on 256x256
python train.py --config configs/experiments/rope_ablation.yaml

# Test extrapolation
python scripts/reproduce/test_rope_extrapolation.py \
    --checkpoint ./checkpoints/rope_experiment/best.pth
```

### Section 3.2.2: Multi-Stage Architecture

```bash
# Ablation: SeqConv vs Standard Conv
python scripts/reproduce/ablate_patch_embedding.py

# Ablation: Multi-stage vs Single-stage
python scripts/reproduce/ablate_multistage.py
```

### Section 3.3.1: Convolutional FFN

```bash
# Compare: Standard FFN vs DWConv vs FullConv
python scripts/reproduce/ablate_conv_ffn.py
```

### Section 3.4: Scaling Experiments

```bash
# Train all model sizes
bash scripts/reproduce/train_all_scales.sh

# Generate scaling curves
python scripts/reproduce/plot_scaling_curves.py
```

### Figure 1: Early Training Visualization

```bash
python scripts/reproduce/visualize_early_training.py \
    --models cnn vit transvae \
    --steps 512 1500 6000
```

### Figure 5: Scaling Comparison

```bash
python scripts/reproduce/compare_scaling.py \
    --architectures cnn vit swin transvae \
    --sizes tiny base large huge giant
```

### Table 1: Reconstruction Benchmarks

```bash
bash scripts/reproduce/benchmark_reconstruction.sh
```

### Table 2: VF Loss Analysis

```bash
python scripts/reproduce/analyze_vf_loss.py
```

## 📁 Project Structure

```
transvae-implementation/
├── configs/                    # Configuration files
│   ├── transvae_tiny_f16d32.yaml
│   ├── transvae_large_f16d32.yaml
│   └── experiments/
├── data/                       # Dataset directory
├── transvae/                   # Main package
│   ├── models/
│   │   ├── transvae.py        # Main TransVAE model
│   │   ├── encoder.py         # Encoder architecture
│   │   ├── decoder.py         # Decoder architecture
│   │   ├── blocks.py          # TransVAE blocks
│   │   └── baseline.py        # CNN/ViT baselines
│   ├── modules/
│   │   ├── attention.py       # Attention mechanisms
│   │   ├── conv.py            # Convolutional layers
│   │   ├── positional.py      # RoPE implementation
│   │   └── upsample.py        # Up/downsample modules
│   ├── losses/
│   │   ├── reconstruction.py  # L1, LPIPS
│   │   ├── gan.py             # GAN loss
│   │   ├── vf.py              # VF alignment loss
│   │   └── kl.py              # KL divergence
│   ├── data/
│   │   ├── imagenet.py        # ImageNet dataset
│   │   └── coco.py            # COCO dataset
│   └── utils/
│       ├── metrics.py         # Evaluation metrics
│       ├── visualization.py   # Plotting utilities
│       └── checkpoint.py      # Model saving/loading
├── scripts/
│   ├── reproduce/             # Scripts to reproduce paper results
│   ├── prepare_dataset.py
│   └── download_imagenet.sh
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Model configurations are defined in YAML files. Example:

```yaml
# configs/transvae_large_f16d32.yaml
model:
  variant: large
  compression_ratio: 16
  latent_dim: 32
  depths: [3, 3, 3, 4, 6]
  base_dims: [192, 192, 384, 768, 1536]
  mlp_ratio: 1.0
  head_dim: 64
  use_rope: true
  use_conv_ffn: true
  use_dc_path: true

training:
  batch_size: 256
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000
  optimizer: adamw
  weight_decay: 0.0

losses:
  l1_weight: 1.0
  lpips_weight: 1.0
  kl_weight: 1e-8
  vf_weight: 0.1
  gan_weight: 0.05
```

## 📝 Tips and Best Practices

### Training Stability

1. **Use gradient clipping**: Essential for large models (>1B params)
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **RMSNorm before attention**: Stabilizes training at scale
3. **Warm up learning rate**: Use 10k steps warmup
4. **Mixed precision**: Use BFloat16 for H20 GPUs

### Memory Optimization

1. **Gradient checkpointing**: Reduces memory for deep models
   ```python
   model.enable_gradient_checkpointing()
   ```

2. **Efficient attention**: Use Flash Attention 2
3. **Batch size tuning**: Start with smaller batch, scale up

### Hyperparameter Tuning

- **KL weight**: Start with 1e-8, increase if latent space collapses
- **VF margin**: Adjust based on convergence (typical: 0.3-0.5)
- **GAN weight**: Keep low (0.05) to avoid artifacts

## 🐛 Troubleshooting

### Common Issues

**Q: Training loss becomes NaN**
- Check gradient norms, enable gradient clipping
- Reduce learning rate
- Check for inf/nan in input data

**Q: Poor extrapolation to higher resolutions**
- Ensure RoPE is enabled
- Verify position embeddings are relative, not absolute

**Q: Blurry reconstructions**
- Increase model capacity
- Add more TransVAE stages (reduce CNN stages)
- Tune perceptual loss weight

**Q: Artifacts in reconstructions**
- Reduce GAN loss weight
- Train decoder longer in stage 2
- Check for mode collapse

## 📚 Citation

```bibtex
@inproceedings{transvae2026,
  title={A Hybrid Paradigm for Vision Autoencoders: Unifying CNNs and Transformers for Learning Efficiency and Scalability},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built on PyTorch and Flash Attention
- Inspired by Stable Diffusion VAE, FLUX VAE, and VA-VAE
- DINOv2 for semantic alignment
- ImageNet and COCO datasets

## 📧 Contact

For questions and feedback:
- Open an issue on GitHub
- Email: [contact@example.com]

---

**Note**: This is a research implementation. Models and weights will be released upon paper acceptance.
