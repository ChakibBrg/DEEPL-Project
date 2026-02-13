"""
TransVAE Decoder Implementation
"""

import torch
import torch.nn as nn
from typing import List
from ..modules.blocks import ResBlock, TransVAEBlock
from ..modules.upsample import Upsample


class TransVAEDecoder(nn.Module):
    """
    TransVAE Decoder with multi-stage architecture (symmetric to encoder)
    
    Args:
        latent_dim: Dimension of latent space
        output_channels: Number of output channels
        depths: Number of blocks per stage (reversed from encoder)
        base_dims: Channel dimensions per stage (reversed from encoder)
        compression_ratio: Spatial compression factor
        mlp_ratio: MLP expansion ratio
        head_dim: Attention head dimension
        use_rope: Use RoPE position embeddings
        use_conv_ffn: Use convolutional FFN
        use_dc_path: Use DC path in upsampling
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        output_channels: int = 3,
        depths: List[int] = [6, 4, 3, 3, 3],
        base_dims: List[int] = [1536, 768, 384, 192, 192],
        compression_ratio: int = 16,
        mlp_ratio: float = 1.0,
        head_dim: int = 64,
        use_rope: bool = True,
        use_conv_ffn: bool = True,
        use_dc_path: bool = True,
    ):
        super().__init__()
        
        self.num_stages = len(depths)
        self.depths = depths
        self.base_dims = base_dims
        
        # Initial projection from latent
        self.conv_in = nn.Conv2d(latent_dim, base_dims[0], 3, padding=1)
        
        # Build stages
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Determine which stages use Transformer vs CNN
        # First stages use Transformer, last 2 use CNN (symmetric to encoder)
        num_transformer_stages = self.num_stages - 2
        
        for i in range(self.num_stages):
            in_dim = base_dims[i]
            out_dim = base_dims[i]
            
            # Build blocks for this stage
            if i < num_transformer_stages:
                # Transformer stage with TransVAE blocks
                blocks = nn.ModuleList([
                    TransVAEBlock(
                        dim=out_dim,
                        mlp_ratio=mlp_ratio,
                        head_dim=head_dim,
                        use_rope=use_rope,
                        use_conv_ffn=use_conv_ffn,
                    )
                    for _ in range(depths[i])
                ])
            else:
                # CNN stage with ResBlocks
                blocks = nn.ModuleList([
                    ResBlock(in_dim if j == 0 else out_dim, out_dim)
                    for j in range(depths[i])
                ])
            
            self.stages.append(blocks)
            
            # Upsample (except for last stage)
            if i < self.num_stages - 1:
                next_dim = base_dims[i + 1]
                self.upsamples.append(
                    Upsample(out_dim, next_dim, use_dc_path=use_dc_path)
                )
        
        # Final output convolution
        self.norm_out = nn.GroupNorm(32, base_dims[-1])
        self.conv_out = nn.Conv2d(base_dims[-1], output_channels, 3, padding=1)
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Latent [B, D, H/f, W/f]
            
        Returns:
            x: Reconstruction [B, C, H, W]
        """
        h = self.conv_in(z)
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            # Process blocks in stage
            for block in stage:
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            
            # Upsample (except last stage)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)
        
        # Final output
        h = self.norm_out(h)
        h = nn.functional.silu(h)
        x = self.conv_out(h)
        
        return x


if __name__ == '__main__':
    # Test decoder
    decoder = TransVAEDecoder(
        latent_dim=32,
        output_channels=3,
        depths=[6, 4, 3, 3, 3],
        base_dims=[1536, 768, 384, 192, 192],
        compression_ratio=16,
    )
    
    z = torch.randn(2, 32, 16, 16)
    x = decoder(z)
    
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x.shape}")
    print(f"Upsampling ratio: {x.shape[-1] / z.shape[-1]}")
