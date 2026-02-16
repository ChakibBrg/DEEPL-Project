from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class TransVAELoss(nn.Module):
    """
    Combined loss for TransVAE training.

    IMPORTANT for this repo:
    - The decoder outputs raw logits (no tanh/sigmoid). We convert reconstruction -> [0,1]
      using sigmoid() for all pixel/perceptual/image-based losses.
    - Input `target` is expected to be in [0,1] (torchvision.transforms.ToTensor()).
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 1.0,
        kl_weight: float = 1e-8,
        vf_weight: float = 0.1,
        gan_weight: float = 0.05,
        use_gan: bool = False,
        logvar_clip: tuple[float, float] = (-30.0, 20.0),
    ):
        super().__init__()

        self.l1_weight = float(l1_weight)
        self.lpips_weight = float(lpips_weight)
        self.kl_weight = float(kl_weight)
        self.vf_weight = float(vf_weight)
        self.gan_weight = float(gan_weight)
        self.use_gan = bool(use_gan)
        self.logvar_clip = logvar_clip

        # LPIPS perceptual loss (frozen)
        self.lpips_loss = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips_loss.parameters():
            p.requires_grad_(False)

        # VF loss is only constructed when needed (avoids overhead/bugs when unused)
        self._vf_loss = None

    @property
    def vf_loss(self):
        if self._vf_loss is None:
            self._vf_loss = VFLoss()
        return self._vf_loss

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        discriminator: nn.Module | None = None,
        dinov2: nn.Module | None = None,
    ) -> dict:
        losses: dict[str, torch.Tensor] = {}

        # --------
        # Image-space reconstruction in [0,1]
        # --------
        # decoder outputs unbounded logits -> stabilize with sigmoid
        recon_img = reconstruction.sigmoid()

        # L1 reconstruction loss
        l1 = F.l1_loss(recon_img, target)
        losses["l1"] = l1 * self.l1_weight

        # LPIPS perceptual loss: expects inputs in [-1,1]
        if self.lpips_weight > 0:
            recon_lp = (recon_img * 2.0 - 1.0).clamp(-1.0, 1.0)
            targ_lp = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
            lp = self.lpips_loss(recon_lp, targ_lp).mean()
            losses["lpips"] = lp * self.lpips_weight
        else:
            losses["lpips"] = recon_img.new_zeros(())

        # KL divergence (FP32 + clamped logvar for stability)
        if self.kl_weight > 0:
            mu32 = mu.float()
            logvar32 = logvar.float().clamp(self.logvar_clip[0], self.logvar_clip[1])
            # mean over all dims
            kl = -0.5 * (1.0 + logvar32 - mu32.pow(2) - logvar32.exp())
            kl = kl.mean()
            losses["kl"] = kl * self.kl_weight
        else:
            losses["kl"] = recon_img.new_zeros(())

        # VF alignment loss (optional; requires dinov2)
        if self.vf_weight > 0 and dinov2 is not None:
            vf = self.vf_loss(recon_img, target, mu, dinov2)
            losses["vf"] = vf * self.vf_weight
        else:
            losses["vf"] = recon_img.new_zeros(())

        # GAN loss (optional)
        if self.use_gan and discriminator is not None and self.gan_weight > 0:
            fake_pred = discriminator(recon_img)
            gan = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
            losses["gan"] = gan * self.gan_weight
        else:
            losses["gan"] = recon_img.new_zeros(())

        losses["total"] = losses["l1"] + losses["lpips"] + losses["kl"] + losses["vf"] + losses["gan"]
        return losses


class VFLoss(nn.Module):
    """
    Visual Feature (VF) alignment loss (simple margin-based similarity).
    """

    def __init__(self, margin: float = 0.4, temperature: float = 0.07):
        super().__init__()
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.proj: nn.Linear | None = None

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        latent: torch.Tensor,
        dinov2: nn.Module,
    ) -> torch.Tensor:
        # DINOv2 expects images around 224; use target only (teacher)
        with torch.no_grad():
            target_resized = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
            dino_features = dinov2(target_resized)

        B, D, H_lat, W_lat = latent.shape
        _, C_dino, H_dino, W_dino = dino_features.shape

        if (H_lat, W_lat) != (H_dino, W_dino):
            latent_resized = F.interpolate(latent, size=(H_dino, W_dino), mode="bilinear", align_corners=False)
        else:
            latent_resized = latent

        if D != C_dino:
            if self.proj is None:
                self.proj = nn.Linear(D, C_dino).to(latent.device)
            latent_flat = latent_resized.flatten(2).transpose(1, 2)  # [B, N, D]
            latent_proj = self.proj(latent_flat).transpose(1, 2)     # [B, C_dino, N]
            latent_proj = latent_proj.reshape(B, C_dino, H_dino, W_dino)
        else:
            latent_proj = latent_resized

        latent_norm = F.normalize(latent_proj, dim=1)
        dino_norm = F.normalize(dino_features, dim=1)

        # similarity in [-1,1]
        similarity = (latent_norm * dino_norm).sum(dim=1).mean()

        loss = torch.clamp(self.margin - similarity, min=0.0)
        return loss
