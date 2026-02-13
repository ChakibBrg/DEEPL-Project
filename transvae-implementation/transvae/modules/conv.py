"""
Convolutional Feed-Forward Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN(nn.Module):
    """
    Convolutional FFN with residual path
    
    Key features:
    - Point-wise expansion
    - Spatial mixing via convolution (with residual)
    - Point-wise projection
    
    Args:
        dim: Input/output dimension
        mlp_ratio: Expansion ratio
        conv_type: Type of convolution ('full' or 'depthwise')
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 1.0,
        conv_type: str = 'full',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio * 4)  # 4x expansion for inverted bottleneck
        
        # Input projection (expansion)
        self.proj_in = nn.Linear(dim, hidden_dim)
        
        # Spatial mixing with convolution
        if conv_type == 'depthwise':
            # Depthwise convolution (lightweight)
            self.conv = nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,  # Depthwise
            )
        elif conv_type == 'full':
            # Full convolution (more expressive)
            conv_hidden = int(dim * mlp_ratio)  # 1x expansion to maintain params
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, conv_hidden, 1),
                nn.GELU(),
                nn.Conv2d(conv_hidden, conv_hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(conv_hidden, hidden_dim, 1),
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # Output projection
        self.proj_out = nn.Linear(hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            out: Output [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Input projection (expansion)
        x_expanded = self.proj_in(x_flat)  # [B, N, hidden_dim]
        x_expanded = F.gelu(x_expanded)
        
        # Reshape for convolution: [B, hidden_dim, H, W]
        x_spatial = x_expanded.transpose(1, 2).reshape(B, -1, H, W)
        
        # Spatial mixing with residual
        x_conv = self.conv(x_spatial)
        x_spatial = x_spatial + x_conv  # Residual connection
        
        # Reshape back: [B, N, hidden_dim]
        x_flat = x_spatial.flatten(2).transpose(1, 2)
        
        # Output projection
        out = self.proj_out(x_flat)
        out = self.dropout(out)
        
        # Reshape back to [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class StandardFFN(nn.Module):
    """
    Standard point-wise FFN (without convolution)
    Used for ablation comparisons
    
    Args:
        dim: Input/output dimension
        mlp_ratio: Expansion ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            out: Output [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Reshape to [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        
        # FFN
        x_flat = self.fc1(x_flat)
        x_flat = self.act(x_flat)
        x_flat = self.dropout(x_flat)
        x_flat = self.fc2(x_flat)
        x_flat = self.dropout(x_flat)
        
        # Reshape back
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return out


if __name__ == '__main__':
    # Test ConvFFN
    print("Testing ConvFFN (full)...")
    ffn_full = ConvFFN(dim=512, mlp_ratio=1.0, conv_type='full')
    x = torch.randn(2, 512, 16, 16)
    out = ffn_full(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    # Count parameters
    params_full = sum(p.numel() for p in ffn_full.parameters())
    print(f"Parameters (full conv): {params_full:,}")
    
    # Test ConvFFN depthwise
    print("\nTesting ConvFFN (depthwise)...")
    ffn_dw = ConvFFN(dim=512, mlp_ratio=1.0, conv_type='depthwise')
    out = ffn_dw(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    params_dw = sum(p.numel() for p in ffn_dw.parameters())
    print(f"Parameters (depthwise): {params_dw:,}")
    
    # Test Standard FFN
    print("\nTesting StandardFFN...")
    ffn_std = StandardFFN(dim=512, mlp_ratio=4.0)
    out = ffn_std(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    
    params_std = sum(p.numel() for p in ffn_std.parameters())
    print(f"Parameters (standard): {params_std:,}")
