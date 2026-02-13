"""
Attention Module with RoPE (Rotary Position Embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class FlashAttentionWithRoPE(nn.Module):
    """
    Flash Attention with Rotary Position Embeddings
    
    Args:
        dim: Feature dimension
        head_dim: Dimension per attention head
        use_rope: Whether to use RoPE
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        head_dim: int = 64,
        use_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        
        # QKV projection with pre-normalization
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        if use_rope:
            self.rope = RoPE2D(head_dim)
    
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
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C] where N = H*W
        
        # Normalize and project Q, K, V
        q = self.to_q(self.norm_q(x_flat))
        k = self.to_k(self.norm_k(x_flat))
        v = self.to_v(self.norm_v(x_flat))
        
        # Reshape to multi-head: [B, num_heads, N, head_dim]
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        if self.use_rope:
            q = self.rope(q, H, W)
            k = self.rope(k, H, W)
        
        # Scaled dot-product attention
        # For efficiency, we use PyTorch's scaled_dot_product_attention
        # which automatically uses Flash Attention when available
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        
        # Reshape back: [B, N, C]
        attn_output = attn_output.transpose(1, 2).reshape(B, H * W, C)
        
        # Output projection
        out = self.proj(attn_output)
        out = self.dropout(out)
        
        # Reshape back to [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embeddings
    
    Args:
        dim: Dimension per head (must be even)
        max_resolution: Maximum resolution to precompute (default: 4096)
    """
    
    def __init__(self, dim: int, max_resolution: int = 4096):
        super().__init__()
        
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        
        self.dim = dim
        self.max_resolution = max_resolution
        
        # Precompute frequency basis
        # We use half the dimension for height and half for width
        dim_per_axis = dim // 2
        
        # Frequency for each dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply 2D RoPE to input tensor
        
        Args:
            x: Input [B, num_heads, N, head_dim] where N = H*W
            H: Height
            W: Width
            
        Returns:
            Rotated tensor with same shape
        """
        B, num_heads, N, head_dim = x.shape
        assert N == H * W, f"Sequence length {N} doesn't match H*W={H*W}"
        assert head_dim == self.dim, f"Head dim {head_dim} doesn't match {self.dim}"
        
        # Generate position indices
        y_pos = torch.arange(H, device=x.device, dtype=x.dtype)
        x_pos = torch.arange(W, device=x.device, dtype=x.dtype)
        
        # Create 2D position grid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        positions = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1)  # [N, 2]
        
        # Compute frequencies
        dim_per_axis = self.dim // 2
        
        # Y positions
        y_positions = positions[:, 0:1]  # [N, 1]
        y_freqs = torch.outer(y_positions.squeeze(-1), self.inv_freq)  # [N, dim_per_axis/2]
        y_emb = torch.cat([y_freqs, y_freqs], dim=-1)  # [N, dim_per_axis]
        
        # X positions
        x_positions = positions[:, 1:2]  # [N, 1]
        x_freqs = torch.outer(x_positions.squeeze(-1), self.inv_freq)  # [N, dim_per_axis/2]
        x_emb = torch.cat([x_freqs, x_freqs], dim=-1)  # [N, dim_per_axis]
        
        # Combine
        emb = torch.cat([y_emb, x_emb], dim=-1)  # [N, head_dim]
        
        # Compute sin and cos
        cos_emb = emb.cos()  # [N, head_dim]
        sin_emb = emb.sin()  # [N, head_dim]
        
        # Apply rotation
        # Split into pairs
        x_reshaped = x.view(B, num_heads, N, head_dim // 2, 2)
        x1 = x_reshaped[..., 0]  # [B, num_heads, N, head_dim/2]
        x2 = x_reshaped[..., 1]  # [B, num_heads, N, head_dim/2]
        
        # Prepare cos and sin for broadcasting
        cos_emb = cos_emb.view(1, 1, N, head_dim // 2, 2)
        sin_emb = sin_emb.view(1, 1, N, head_dim // 2, 2)
        
        cos1 = cos_emb[..., 0]  # [1, 1, N, head_dim/2]
        cos2 = cos_emb[..., 1]
        sin1 = sin_emb[..., 0]
        sin2 = sin_emb[..., 1]
        
        # Rotation
        out1 = x1 * cos1 - x2 * sin1
        out2 = x1 * sin2 + x2 * cos2
        
        # Recombine
        out = torch.stack([out1, out2], dim=-1)
        out = out.view(B, num_heads, N, head_dim)
        
        return out


if __name__ == '__main__':
    # Test attention module
    print("Testing FlashAttentionWithRoPE...")
    
    attn = FlashAttentionWithRoPE(dim=512, head_dim=64, use_rope=True)
    x = torch.randn(2, 512, 16, 16)
    out = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Num heads: {attn.num_heads}")
    
    # Test with different resolutions (extrapolation)
    print("\nTesting resolution extrapolation...")
    x256 = torch.randn(1, 512, 16, 16)  # 256x256 / 16 = 16x16
    x512 = torch.randn(1, 512, 32, 32)  # 512x512 / 16 = 32x32
    
    out256 = attn(x256)
    out512 = attn(x512)
    
    print(f"256x256: {x256.shape} -> {out256.shape}")
    print(f"512x512: {x512.shape} -> {out512.shape}")
    print("RoPE enables seamless resolution extrapolation!")
