"""
Loss functions for TransVAE training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class TransVAELoss(nn.Module):
    """
    Combined loss for TransVAE training
    
    Components:
    - L1 reconstruction loss
    - LPIPS perceptual loss
    - KL divergence loss
    - VF (Visual Feature) alignment loss
    - GAN loss (optional, for stage 2)
    
    Args:
        l1_weight: Weight for L1 loss
        lpips_weight: Weight for LPIPS loss
        kl_weight: Weight for KL loss
        vf_weight: Weight for VF loss
        gan_weight: Weight for GAN loss
        use_gan: Whether to use GAN loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 1.0,
        kl_weight: float = 1e-8,
        vf_weight: float = 0.1,
        gan_weight: float = 0.05,
        use_gan: bool = False,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.kl_weight = kl_weight
        self.vf_weight = vf_weight
        self.gan_weight = gan_weight
        self.use_gan = use_gan
        
        # LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        
        # VF loss (if using DINOv2 alignment)
        self.vf_loss = VFLoss() if vf_weight > 0 else None
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        discriminator: nn.Module = None,
        dinov2: nn.Module = None,
    ) -> dict:
        """
        Compute combined loss
        
        Args:
            reconstruction: Reconstructed image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            mu: Latent mean [B, D, H/f, W/f]
            logvar: Latent log variance [B, D, H/f, W/f]
            discriminator: Discriminator model (optional)
            dinov2: DINOv2 model for VF loss (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # L1 reconstruction loss
        l1_loss = F.l1_loss(reconstruction, target)
        losses['l1'] = l1_loss * self.l1_weight
        
        # LPIPS perceptual loss
        # Normalize to [-1, 1] for LPIPS
        recon_norm = reconstruction * 2 - 1
        target_norm = target * 2 - 1
        lpips_loss = self.lpips_loss(recon_norm, target_norm).mean()
        losses['lpips'] = lpips_loss * self.lpips_weight
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[2] * mu.shape[3])  # Normalize
        losses['kl'] = kl_loss * self.kl_weight
        
        # VF alignment loss
        if self.vf_loss is not None and dinov2 is not None:
            vf_loss = self.vf_loss(reconstruction, target, mu, dinov2)
            losses['vf'] = vf_loss * self.vf_weight
        
        # GAN loss
        if self.use_gan and discriminator is not None:
            # Generator loss (fool discriminator)
            fake_pred = discriminator(reconstruction)
            gan_loss = F.binary_cross_entropy_with_logits(
                fake_pred,
                torch.ones_like(fake_pred)
            )
            losses['gan'] = gan_loss * self.gan_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class VFLoss(nn.Module):
    """
    Visual Feature (VF) alignment loss
    
    Aligns VAE latent space with DINOv2 features
    Uses margin-based contrastive loss
    
    Args:
        margin: Margin for contrastive loss
        temperature: Temperature for similarity
    """
    
    def __init__(self, margin: float = 0.4, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        latent: torch.Tensor,
        dinov2: nn.Module,
    ) -> torch.Tensor:
        """
        Compute VF loss
        
        Args:
            reconstruction: Reconstructed image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            latent: Latent representation [B, D, H/f, W/f]
            dinov2: DINOv2 model
            
        Returns:
            VF loss scalar
        """
        # Extract DINOv2 features
        with torch.no_grad():
            # DINOv2 expects images at 224x224
            target_resized = F.interpolate(
                target, size=(224, 224), mode='bilinear', align_corners=False
            )
            dino_features = dinov2(target_resized)  # [B, C_dino, H_dino, W_dino]
        
        # Downsample VAE latent to match DINOv2 spatial size
        B, D, H_lat, W_lat = latent.shape
        _, C_dino, H_dino, W_dino = dino_features.shape
        
        if H_lat != H_dino or W_lat != W_dino:
            latent_resized = F.interpolate(
                latent, size=(H_dino, W_dino), mode='bilinear', align_corners=False
            )
        else:
            latent_resized = latent
        
        # Project latent to same dimension as DINOv2 if needed
        if D != C_dino:
            # Simple linear projection
            if not hasattr(self, 'proj'):
                self.proj = nn.Linear(D, C_dino).to(latent.device)
            
            latent_flat = latent_resized.flatten(2).transpose(1, 2)  # [B, N, D]
            latent_proj = self.proj(latent_flat).transpose(1, 2)  # [B, C_dino, N]
            latent_proj = latent_proj.reshape(B, C_dino, H_dino, W_dino)
        else:
            latent_proj = latent_resized
        
        # Normalize features
        latent_norm = F.normalize(latent_proj, dim=1)
        dino_norm = F.normalize(dino_features, dim=1)
        
        # Compute cosine similarity
        similarity = (latent_norm * dino_norm).sum(dim=1).mean()
        
        # Margin-based loss: encourage similarity to be above margin
        loss = torch.clamp(self.margin - similarity, min=0.0)
        
        return loss


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for adversarial training
    
    Args:
        loss_type: Type of GAN loss ('bce', 'hinge', 'wgan')
    """
    
    def __init__(self, loss_type: str = 'bce'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discriminator loss
        
        Args:
            real_pred: Discriminator predictions for real images
            fake_pred: Discriminator predictions for fake images
            
        Returns:
            Discriminator loss
        """
        if self.loss_type == 'bce':
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            return (real_loss + fake_loss) / 2
        
        elif self.loss_type == 'hinge':
            real_loss = torch.mean(F.relu(1.0 - real_pred))
            fake_loss = torch.mean(F.relu(1.0 + fake_pred))
            return (real_loss + fake_loss) / 2
        
        elif self.loss_type == 'wgan':
            return -torch.mean(real_pred) + torch.mean(fake_pred)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


if __name__ == '__main__':
    # Test loss computation
    print("Testing TransVAELoss...")
    
    loss_fn = TransVAELoss(
        l1_weight=1.0,
        lpips_weight=1.0,
        kl_weight=1e-8,
        vf_weight=0.0,  # Disable VF for testing
        use_gan=False,
    )
    
    # Create dummy data
    B, C, H, W = 4, 3, 256, 256
    reconstruction = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    mu = torch.randn(B, 32, H // 16, W // 16)
    logvar = torch.randn(B, 32, H // 16, W // 16)
    
    # Compute losses
    losses = loss_fn(reconstruction, target, mu, logvar)
    
    print("Computed losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
