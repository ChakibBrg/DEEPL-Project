"""
Core Building Blocks for TransVAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.attention import FlashAttentionWithRoPE
from ..modules.conv import ConvFFN


class ResBlock(nn.Module):
    """
    Residual Block for CNN stages
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_conv_shortcut: Use convolutional shortcut
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            out: Output [B, C', H, W]
        """
        h = x
        
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class TransVAEBlock(nn.Module):
    """
    TransVAE Block: Attention + Conv FFN
    
    Key features:
    - Flash Attention with RoPE
    - RMSNorm before attention
    - Convolutional FFN with residual path
    
    Args:
        dim: Feature dimension
        mlp_ratio: MLP expansion ratio
        head_dim: Attention head dimension
        use_rope: Use RoPE position embeddings
        use_conv_ffn: Use convolutional FFN (vs standard FFN)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 1.0,
        head_dim: int = 64,
        use_rope: bool = True,
        use_conv_ffn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        # Pre-attention norm
        self.norm1 = RMSNorm(dim)
        
        # Attention
        self.attn = FlashAttentionWithRoPE(
            dim=dim,
            head_dim=head_dim,
            use_rope=use_rope,
            dropout=dropout,
        )
        
        # Pre-FFN norm
        self.norm2 = RMSNorm(dim)
        
        # FFN
        if use_conv_ffn:
            self.ffn = ConvFFN(
                dim=dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        else:
            # Standard FFN
            hidden_dim = int(dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            out: Output [B, C, H, W]
        """
        # Attention block
        x = x + self.attn(self.norm1(x))
        
        # FFN block
        x = x + self.ffn(self.norm2(x))
        
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Args:
        dim: Feature dimension
        eps: Small constant for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W] or [B, N, C]
            
        Returns:
            Normalized output with same shape
        """
        # Get the channel dimension
        if x.dim() == 4:  # [B, C, H, W]
            # Reshape for normalization
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)  # [B, C, H*W]
            
            # Compute RMS
            rms = torch.sqrt(torch.mean(x_flat ** 2, dim=1, keepdim=True) + self.eps)
            
            # Normalize
            x_norm = x_flat / rms  # [B, C, H*W]
            
            # Scale
            x_norm = x_norm * self.weight.view(1, -1, 1)
            
            # Reshape back
            return x_norm.view(B, C, H, W)
        
        elif x.dim() == 3:  # [B, N, C]
            # Compute RMS
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            
            # Normalize and scale
            return (x / rms) * self.weight
        
        else:
            raise ValueError(f"RMSNorm expects 3D or 4D input, got {x.dim()}D")


if __name__ == '__main__':
    # Test ResBlock
    print("Testing ResBlock...")
    res_block = ResBlock(256, 512)
    x = torch.randn(2, 256, 32, 32)
    out = res_block(x)
    print(f"ResBlock: {x.shape} -> {out.shape}")
    
    # Test TransVAEBlock
    print("\nTesting TransVAEBlock...")
    transvae_block = TransVAEBlock(dim=512, mlp_ratio=1.0, head_dim=64)
    x = torch.randn(2, 512, 16, 16)
    out = transvae_block(x)
    print(f"TransVAEBlock: {x.shape} -> {out.shape}")
    
    # Test RMSNorm
    print("\nTesting RMSNorm...")
    norm = RMSNorm(512)
    x = torch.randn(2, 512, 16, 16)
    out = norm(x)
    print(f"RMSNorm: {x.shape} -> {out.shape}")
    print(f"Mean: {out.mean():.6f}, Std: {out.std():.6f}")
