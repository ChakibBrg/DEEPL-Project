# Reproducing Paper Experiments

This document provides step-by-step instructions to reproduce all experiments from the paper "A Hybrid Paradigm for Vision Autoencoders: Unifying CNNs and Transformers for Learning Efficiency and Scalability".

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Section 3.1: Preliminaries](#section-31-preliminaries)
3. [Section 3.2: Macro Design](#section-32-macro-design)
4. [Section 3.3: Micro Design](#section-33-micro-design)
5. [Section 3.4: Scaling Properties](#section-34-scaling-properties)
6. [Section 4: Main Results](#section-4-main-results)

---

## Environment Setup

### Hardware Requirements

- **Training**: 8x H20 GPUs (80GB each) for large models
- **Evaluation**: 1x GPU (16GB+) sufficient
- **Storage**: 500GB for ImageNet-1k dataset

### Software Installation

```bash
conda create -n transvae python=3.9
conda activate transvae
pip install -r requirements.txt
pip install -e .
```

---

## Section 3.1: Preliminaries

### Figure 1: Early Training Visualization

**Compares CNN-VAE, ViT-VAE, and TransVAE during early training.**

```bash
# Train all three models and visualize
python scripts/reproduce/visualize_early_training.py \
    --data_dir ./data/imagenet \
    --batch_size 256 \
    --steps 512 1500 6000 \
    --output_dir results/figure1
```

**Expected Results:**
- CNN-VAE: Good local details, poor global structure initially
- ViT-VAE: Good global structure, blurry details, slow convergence
- TransVAE: Both local AND global, fastest convergence

**Runtime:** ~8 hours on 8 GPUs

---

## Section 3.2: Macro Design

### Section 3.2.1: Resolution Extrapolation with RoPE

**Figure 3(a): Tests models trained on 256×256 on higher resolutions.**

#### Step 1: Train Models

Train three variants:
1. ViT-VAE with Absolute Position Embeddings (APE)
2. ViT-VAE with RoPE
3. TransVAE with RoPE

```bash
# 1. ViT-VAE with APE (baseline)
python train.py \
    --config configs/experiments/vit_vae_ape.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/vit_ape \
    --resolution 256 \
    --num_epochs 5

# 2. ViT-VAE with RoPE
python train.py \
    --config configs/experiments/vit_vae_rope.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/vit_rope \
    --resolution 256 \
    --num_epochs 5

# 3. TransVAE with RoPE (ours)
python train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/transvae_rope \
    --resolution 256 \
    --num_epochs 5
```

#### Step 2: Test Extrapolation

```bash
# Test all models on 256, 512, 1024 resolutions
for model in vit_ape vit_rope transvae_rope; do
    python scripts/reproduce/test_rope_extrapolation.py \
        --checkpoint ./checkpoints/${model}/best.pth \
        --data_dir ./data/imagenet \
        --train_resolution 256 \
        --test_resolutions 256 512 1024 \
        --output_dir results/figure3a_${model}
done
```

**Expected Results:**
- ViT-VAE (APE): Severe grid artifacts at 512×512 and 1024×1024
- ViT-VAE (RoPE): Improved, but still some artifacts
- TransVAE (RoPE): Clean extrapolation to all resolutions

**Runtime:** Training: ~40 hours per model on 8 GPUs; Testing: ~1 hour per model

---

### Section 3.2.2: Multi-Stage Architecture

**Figure 3(b,c): Convergence curves for different patch embedding and staging.**

#### Ablation 1: Patch Embedding

Compare:
1. Single non-overlapping conv
2. Sequential overlapping convs (SeqConv)

```bash
python scripts/reproduce/ablate_patch_embedding.py \
    --data_dir ./data/imagenet \
    --output_dir results/figure3b \
    --num_epochs 5 \
    --batch_size 256
```

**Expected Results:**
- SeqConv: Faster convergence, lower loss
- Standard: Slower convergence, higher loss

#### Ablation 2: Multi-Stage Architecture

Compare:
1. Single-stage ViT
2. Multi-stage with CNN stem

```bash
python scripts/reproduce/ablate_multistage.py \
    --data_dir ./data/imagenet \
    --output_dir results/figure3c \
    --num_epochs 5 \
    --batch_size 256
```

**Expected Results:**
- Multi-stage: Matches CNN-VAE convergence speed
- Single-stage: Significantly slower

**Runtime:** ~40 hours per ablation on 8 GPUs

---

## Section 3.3: Micro Design

### Section 3.3.1: Convolutional FFN

**Figure 4(a,b): Impact of Conv-FFN on convergence.**

#### Experiment: Compare FFN Variants

Test three FFN types:
1. Standard point-wise FFN
2. Depthwise convolution FFN (DWConv)
3. Full convolution FFN (FullConv)

```bash
python scripts/reproduce/ablate_conv_ffn.py \
    --data_dir ./data/imagenet \
    --output_dir results/figure4 \
    --ffn_types standard dwconv fullconv \
    --num_epochs 5 \
    --batch_size 256
```

**Expected Results (Convergence Speed):**
1. Standard FFN: Baseline
2. DWConv: ~1.5× faster convergence
3. FullConv: ~2× faster convergence

**Runtime:** ~40 hours per variant on 8 GPUs

---

### Section 3.3.2: Minor Modifications

**Figure 4(c): Impact of DC path and RMSNorm.**

```bash
python scripts/reproduce/ablate_minor_mods.py \
    --data_dir ./data/imagenet \
    --output_dir results/figure4c \
    --num_epochs 10 \
    --batch_size 256
```

Test combinations:
- Baseline (no DC path, LayerNorm)
- +DC path
- +RMSNorm
- +Both

**Expected Results:**
- DC path: Noticeable improvement
- RMSNorm: Marginal improvement
- Both: Best performance

**Runtime:** ~80 hours total on 8 GPUs

---

## Section 3.4: Scaling Properties

### Figure 5: Scaling Comparison

**Train 5 sizes of 4 architectures (CNN, ViT, Swin, TransVAE).**

#### Step 1: Train All Models

This is the most compute-intensive experiment.

```bash
# Train all model variants
bash scripts/reproduce/train_all_scales.sh
```

This script trains:
- CNN-VAE: Tiny, Base, Large, Huge, Giant
- ViT-VAE: Tiny, Base, Large, Huge, Giant
- Swin-VAE: Tiny, Base, Large, Huge, Giant
- TransVAE: Tiny, Base, Large, Huge, Giant

**Total:** 20 models, 3 epochs each

#### Step 2: Generate Plots

```bash
python scripts/reproduce/plot_scaling_curves.py \
    --checkpoint_dir ./checkpoints/scaling \
    --output_dir results/figure5
```

**Expected Results:**
- CNN-VAE: Limited scaling benefit after Large
- ViT-VAE: Poor scaling, high-frequency artifacts at small sizes
- Swin-VAE: Better scaling than ViT, but still limited
- TransVAE: Clear scaling benefits across all sizes

**Runtime:** 
- Training: ~1000 GPU hours (50 days on 8 GPUs if sequential)
- Recommend parallel training on multiple nodes
- Plotting: ~2 hours

---

## Section 4: Main Results

### Table 1: Image Reconstruction Performance

**Evaluate on ImageNet and COCO at multiple resolutions.**

#### Step 1: Train Full Models

Train TransVAE variants:
- TransVAE-T f16d32
- TransVAE-L f16d32
- TransVAE-L f8d16

```bash
# TransVAE-T f16d32 (ImageNet only, 256×256)
python train.py \
    --config configs/transvae_tiny_f16d32.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/transvae_t_f16d32 \
    --num_epochs 110 \
    --batch_size 256 \
    --num_gpus 8

# TransVAE-L f16d32 (ImageNet only, 256×256)
python train.py \
    --config configs/transvae_large_f16d32.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/transvae_l_f16d32 \
    --num_epochs 110 \
    --batch_size 256 \
    --num_gpus 8

# TransVAE-L f8d16 (Large-scale dataset, 256×256)
python train.py \
    --config configs/transvae_large_f8d16.yaml \
    --data_dir ./data/large_scale \
    --output_dir ./checkpoints/transvae_l_f8d16 \
    --num_epochs 110 \
    --batch_size 256 \
    --num_gpus 8
```

**Training Strategy:**
- Stage 1: 100 epochs with L1, LPIPS, KL, VF losses
- Stage 2: 10 epochs with GAN loss, encoder frozen

#### Step 2: Comprehensive Evaluation

```bash
bash scripts/reproduce/benchmark_reconstruction.sh
```

This evaluates:
- ImageNet val: 256×256, 512×512, 1024×1024
- COCO val: 256×256, 512×512, 1024×1024
- Metrics: PSNR, SSIM, LPIPS, rFID

**Expected Results (TransVAE-L f16d32, ImageNet 256×256):**
- PSNR: 28.92 dB
- SSIM: 0.82
- LPIPS: 0.086
- rFID: 0.66

**Runtime:**
- Training: ~220 hours per model on 8 GPUs
- Evaluation: ~4 hours per model

---

### Table 2: VF Loss Analysis

**Compare models with and without VF loss.**

```bash
python scripts/reproduce/analyze_vf_loss.py \
    --data_dir ./data/imagenet \
    --output_dir results/table2
```

Tests:
1. Linear probing accuracy
2. Latent space metrics (Density CV, Normalized Entropy, Gini)
3. Reconstruction quality with/without VF

**Expected Results:**
- With VF: Better linear probing accuracy (58.77% for TransVAE-L)
- With VF: More discriminative latent space (lower Gini)
- Trade-off: Slightly lower PSNR with VF at same resolution

**Runtime:** ~16 hours on 8 GPUs

---

### Figure 6: Downstream Generation

**Train DiT models with different VAE latents.**

#### Step 1: Extract Latents

```bash
# Extract latents from TransVAE, FLUX-VAE, SD3.5-VAE
python scripts/extract_latents.py \
    --vae_checkpoint ./checkpoints/transvae_l_f8d16/best.pth \
    --data_dir ./data/imagenet \
    --output_dir ./data/latents/transvae
```

#### Step 2: Train DiT Models

```bash
# LightningDiT-B/2 with TransVAE latents
python train_dit.py \
    --vae_checkpoint ./checkpoints/transvae_l_f8d16/best.pth \
    --dit_config configs/lightningdit_b2.yaml \
    --data_dir ./data/imagenet \
    --output_dir ./checkpoints/dit_transvae_b2 \
    --num_epochs 160 \
    --batch_size 1024
```

#### Step 3: Evaluate FID

```bash
python evaluate_dit.py \
    --vae_checkpoint ./checkpoints/transvae_l_f8d16/best.pth \
    --dit_checkpoint ./checkpoints/dit_transvae_b2/best.pth \
    --data_dir ./data/imagenet \
    --num_samples 10000
```

**Expected Results (FID-10K, no CFG):**
- TransVAE-L + DiT-B/2: 28.72 (best)
- FLUX-VAE + DiT-B/2: 37.45
- SD3.5-VAE + DiT-B/2: 33.52

**Runtime:**
- DiT training: ~200 hours on 8 GPUs
- FID evaluation: ~2 hours

---

## Quick Verification (For Reviewers)

If you want to quickly verify key claims without full training:

### 1. Architecture Test (~10 minutes)

```bash
python test_architecture.py
```

Verifies:
- Model can be instantiated
- Forward pass works
- RoPE enables resolution extrapolation
- Conv-FFN maintains local biases

### 2. Small-Scale Training (~4 hours on 1 GPU)

```bash
python train.py \
    --config configs/transvae_tiny_f16d32.yaml \
    --data_dir ./data/tiny_imagenet \
    --output_dir ./checkpoints/quick_test \
    --num_epochs 10 \
    --batch_size 32
```

Should show:
- Convergence within 10 epochs
- PSNR > 25 dB on validation set

### 3. Extrapolation Test (~30 minutes)

```bash
python scripts/reproduce/test_rope_extrapolation.py \
    --checkpoint ./checkpoints/quick_test/best.pth \
    --data_dir ./data/tiny_imagenet \
    --test_resolutions 64 128 256
```

Should show:
- No artifacts at higher resolutions
- PSNR degrades gracefully

---

## Computational Budget Summary

| Experiment | GPUs | Time | GPU-Hours |
|------------|------|------|-----------|
| Section 3.1 (Figure 1) | 8 | 8h | 64 |
| Section 3.2.1 (RoPE) | 8 | 120h | 960 |
| Section 3.2.2 (Multi-stage) | 8 | 80h | 640 |
| Section 3.3.1 (Conv-FFN) | 8 | 120h | 960 |
| Section 3.4 (Scaling) | 8 | 1000h | 8000 |
| Table 1 (Main results) | 8 | 660h | 5280 |
| Table 2 (VF loss) | 8 | 16h | 128 |
| Figure 6 (DiT) | 8 | 200h | 1600 |
| **Total** | | | **17,632** |

**Total Cost Estimate:** ~$35,000 at $2/GPU-hour (cloud pricing)

---

## Notes

- All experiments use ImageNet-1k (256×256) unless specified
- Random seed is fixed for reproducibility
- Checkpoints are saved every 5000 steps
- Validation runs every 1000 steps
- Early stopping is NOT used (train full epochs)

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{transvae2026,
  title={A Hybrid Paradigm for Vision Autoencoders: Unifying CNNs and Transformers},
  author={Anonymous},
  booktitle={ICLR},
  year={2026}
}
```
