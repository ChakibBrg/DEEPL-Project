"""
TransVAE Encoder Implementation
"""

import torch
import torch.nn as nn
from typing import List
from ..modules.blocks import ResBlock, TransVAEBlock
from ..modules.upsample import Downsample


class TransVAEEncoder(nn.Module):
    """
    TransVAE Encoder with multi-stage architecture
    - First stages: CNN ResBlocks for local features
    - Later stages: TransVAE blocks for global context
    
    Args:
        input_channels: Number of input channels
        latent_dim: Dimension of latent space
        depths: Number of blocks per stage
        base_dims: Channel dimensions per stage
        compression_ratio: Spatial compression factor
        mlp_ratio: MLP expansion ratio
        head_dim: Attention head dimension
        use_rope: Use RoPE position embeddings
        use_conv_ffn: Use convolutional FFN
        use_dc_path: Use DC path in downsampling
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        depths: List[int] = [3, 3, 3, 4, 6],
        base_dims: List[int] = [192, 192, 384, 768, 1536],
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
        self.compression_ratio = compression_ratio
        
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channels, base_dims[0], 3, padding=1)
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        # Determine which stages use CNN vs Transformer
        # First 2 stages use CNN, rest use Transformer
        num_cnn_stages = 2
        
        for i in range(self.num_stages):
            in_dim = base_dims[i]
            out_dim = base_dims[i]
            
            # Build blocks for this stage
            if i < num_cnn_stages:
                # CNN stage with ResBlocks
                blocks = nn.ModuleList([
                    ResBlock(in_dim if j == 0 else out_dim, out_dim)
                    for j in range(depths[i])
                ])
            else:
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
            
            self.stages.append(blocks)
            
            # Downsample (except for last stage)
            if i < self.num_stages - 1:
                next_dim = base_dims[i + 1]
                self.downsamples.append(
                    Downsample(out_dim, next_dim, use_dc_path=use_dc_path)
                )
        
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            h: Encoded features [B, D, H/f, W/f]
        """
        h = self.conv_in(x)
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            # Process blocks in stage
            for block in stage:
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            
            # Downsample (except last stage)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        
        return h


if __name__ == '__main__':
    # Test encoder
    encoder = TransVAEEncoder(
        input_channels=3,
        latent_dim=32,
        depths=[3, 3, 3, 4, 6],
        base_dims=[192, 192, 384, 768, 1536],
        compression_ratio=16,
    )
    
    x = torch.randn(2, 3, 256, 256)
    h = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {h.shape}")
    print(f"Compression ratio: {x.shape[-1] / h.shape[-1]}")
