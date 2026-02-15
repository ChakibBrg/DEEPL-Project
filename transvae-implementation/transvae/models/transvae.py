"""
TransVAE: A Hybrid Paradigm for Vision Autoencoders
Main model implementation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from .encoder import TransVAEEncoder
from .decoder import TransVAEDecoder


class TransVAE(nn.Module):
    """
    TransVAE: Hybrid CNN-Transformer Variational Autoencoder
    
    Args:
        variant: Model variant ('tiny', 'base', 'large', 'huge', 'giant')
        compression_ratio: Spatial compression ratio (8 or 16)
        latent_dim: Latent space dimension
        input_channels: Number of input channels (default: 3)
        use_rope: Use Rotary Position Embeddings (default: True)
        use_conv_ffn: Use convolutional FFN (default: True)
        use_dc_path: Use DC path in up/downsample (default: True)
    """
    
    def __init__(
        self,
        config,
        variant: str = 'large',
        compression_ratio: int = 16,
        latent_dim: int = 32,
        input_channels: int = 3,
        use_rope: bool = True,
        use_conv_ffn: bool = True,
        use_dc_path: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.variant = variant
        self.compression_ratio = compression_ratio
        self.latent_dim = latent_dim
        
        # Model configuration based on variant
        # config = self._get_variant_config(variant, compression_ratio, latent_dim)
        
        # Encoder
        # self.encoder = TransVAEEncoder(
        #     input_channels=input_channels,
        #     latent_dim=latent_dim,
        #     depths=config['depths'],
        #     base_dims=config['base_dims'],
        #     compression_ratio=compression_ratio,
        #     mlp_ratio=config['mlp_ratio'],
        #     head_dim=config['head_dim'],
        #     use_rope=use_rope,
        #     use_conv_ffn=use_conv_ffn,
        #     use_dc_path=use_dc_path,
        # )
        self.encoder = TransVAEEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            depths=config.get('depths'),
            base_dims=config.get('base_dims'),
            compression_ratio=compression_ratio,
            mlp_ratio=config.get('mlp_ratio', 1.0),
            head_dim=config.get('head_dim', 64),
            use_rope=use_rope,
            use_conv_ffn=use_conv_ffn,
            use_dc_path=use_dc_path,
        )
        
        # Latent projection
        final_dim = config['base_dims'][-1]
        self.conv_mu = nn.Conv2d(final_dim, latent_dim, 3, padding=1)
        self.conv_logvar = nn.Conv2d(final_dim, latent_dim, 3, padding=1)
        
        # Decoder (symmetric to encoder)
        # self.decoder = TransVAEDecoder(
        #     latent_dim=latent_dim,
        #     output_channels=input_channels,
        #     depths=config['depths'][::-1],
        #     base_dims=config['base_dims'][::-1],
        #     compression_ratio=compression_ratio,
        #     mlp_ratio=config['mlp_ratio'],
        #     head_dim=config['head_dim'],
        #     use_rope=use_rope,
        #     use_conv_ffn=use_conv_ffn,
        #     use_dc_path=use_dc_path,
        # )
        self.decoder = TransVAEDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            depths=config.get('depths')[::-1],
            base_dims=config.get('base_dims')[::-1],
            compression_ratio=compression_ratio,
            mlp_ratio=config.get('mlp_ratio', 1.0),
            head_dim=config.get('head_dim', 64),
            use_rope=use_rope,
            use_conv_ffn=use_conv_ffn,
            use_dc_path=use_dc_path,
        )
        
        self._initialize_weights()
    
    def _get_variant_config(self, variant: str, f: int, d: int) -> dict:
        """Get model configuration for different variants"""
        
        configs = {
            'tiny_f16d32': {
                'depths': [3, 3, 3, 3, 3],
                'base_dims': [128, 128, 256, 256, 512],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
            'base_f16d32': {
                'depths': [3, 3, 3, 3, 3],
                'base_dims': [128, 128, 256, 512, 1024],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
            'large_f16d32': {
                'depths': [3, 3, 3, 4, 6],
                'base_dims': [192, 192, 384, 768, 1536],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
            'huge_f16d32': {
                'depths': [3, 3, 4, 6, 8],
                'base_dims': [256, 256, 512, 1024, 2048],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
            'giant_f16d32': {
                'depths': [3, 3, 4, 8, 10],
                'base_dims': [320, 320, 640, 1280, 2560],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
            'large_f8d16': {
                'depths': [3, 3, 6, 8],
                'base_dims': [192, 384, 768, 1536],
                'mlp_ratio': 1.0,
                'head_dim': 64,
            },
        }
        
        key = f"{variant}_f{f}d{d}"
        if key in configs:
            return configs[key]
        else:
            raise ValueError(f"Unknown variant: {variant} with f{f}d{d}")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            mu: Mean of latent distribution [B, D, H/f, W/f]
            logvar: Log variance of latent distribution [B, D, H/f, W/f]
        """
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # IMPORTANT: do this in FP32 for stability
        mu_f = mu.float()
        logvar_f = logvar.float().clamp(-30.0, 20.0)   # critical clamp

        std = torch.exp(0.5 * logvar_f)
        eps = torch.randn_like(std)
        z = mu_f + eps * std

        # cast back to original dtype for decoder
        return z.to(mu.dtype)

    
    # def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    #     """
    #     Reparameterization trick
        
    #     Args:
    #         mu: Mean [B, D, H, W]
    #         logvar: Log variance [B, D, H, W]
            
    #     Returns:
    #         z: Sampled latent [B, D, H, W]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction
        
        Args:
            z: Latent tensor [B, D, H/f, W/f]
            
        Returns:
            Reconstruction [B, C, H, W]
        """
        return self.decoder(z)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_dict: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            return_dict: Return dictionary instead of tuple
            
        Returns:
            reconstruction: Reconstructed image [B, C, H, W]
            mu: Latent mean [B, D, H/f, W/f]
            logvar: Latent log variance [B, D, H/f, W/f]
        """
        mu, logvar = self.encode(x)
        mu = mu.clamp(-50, 50)
        logvar = logvar.clamp(-30, 20)

        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        if return_dict:
            return {
                'reconstruction': reconstruction,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        
        return reconstruction, mu, logvar
    
    def get_last_layer(self):
        """Get last layer for GAN training"""
        return self.decoder.conv_out.weight
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load pretrained model"""
        # This would load from a checkpoint
        # For now, just initialize
        variant, config = model_name.split('-')[1:3]
        f, d = int(config[1:].split('d')[0]), int(config.split('d')[1])
        
        model = cls(
            variant=variant,
            compression_ratio=f,
            latent_dim=d,
            **kwargs
        )
        
        # TODO: Load actual weights
        # checkpoint = torch.hub.load_state_dict_from_url(url)
        # model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.enable_gradient_checkpointing()
        self.decoder.enable_gradient_checkpointing()
    
    def get_num_params(self) -> dict:
        """Get number of parameters"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params,
        }


def create_transvae(
    variant: str = 'large',
    compression_ratio: int = 16,
    latent_dim: int = 32,
    **kwargs
) -> TransVAE:
    """
    Factory function to create TransVAE model
    
    Args:
        variant: Model size ('tiny', 'base', 'large', 'huge', 'giant')
        compression_ratio: Spatial compression (8 or 16)
        latent_dim: Latent dimension (typically 16 or 32)
        **kwargs: Additional arguments passed to TransVAE
        
    Returns:
        TransVAE model
    """
    return TransVAE(
        variant=variant,
        compression_ratio=compression_ratio,
        latent_dim=latent_dim,
        **kwargs
    )


if __name__ == '__main__':
    # Test model instantiation
    model = create_transvae(variant='large', compression_ratio=16, latent_dim=32)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    reconstruction, mu, logvar = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"\nModel parameters: {model.get_num_params()}")
