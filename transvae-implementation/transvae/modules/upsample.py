"""
Upsample and Downsample modules with DC path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """
    Downsampling module with optional DC (Direct Connect) path
    
    The DC path preserves information through pixel unshuffle
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_dc_path: Use DC path for information preservation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dc_path: bool = True,
    ):
        super().__init__()
        
        self.use_dc_path = use_dc_path
        
        # Main path: Sequential overlapping convolutions
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        )
        
        # DC path: Pixel unshuffle (lossless downsampling)
        if use_dc_path:
            # After pixel unshuffle, channels are multiplied by 4
            self.dc_conv = nn.Conv2d(in_channels * 4, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C_in, H, W]
            
        Returns:
            out: Output [B, C_out, H/2, W/2]
        """
        # Main path
        out = self.main_path(x)
        
        # DC path
        if self.use_dc_path:
            # Pixel unshuffle: [B, C, H, W] -> [B, C*4, H/2, W/2]
            x_dc = F.pixel_unshuffle(x, downscale_factor=2)
            x_dc = self.dc_conv(x_dc)
            
            # Combine paths
            out = out + x_dc
        
        return out


class Upsample(nn.Module):
    """
    Upsampling module with optional DC (Direct Connect) path
    
    The DC path preserves information through pixel shuffle
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_dc_path: Use DC path for information preservation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dc_path: bool = True,
    ):
        super().__init__()
        
        self.use_dc_path = use_dc_path
        
        # Main path: Sequential overlapping convolutions
        # First upsample then convolve
        self.main_path = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        
        # DC path: Pixel shuffle (lossless upsampling)
        if use_dc_path:
            # Before pixel shuffle, we need to expand channels
            self.dc_conv = nn.Conv2d(in_channels, out_channels * 4, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input [B, C_in, H, W]
            
        Returns:
            out: Output [B, C_out, H*2, W*2]
        """
        # Main path
        out = self.main_path(x)
        
        # DC path
        if self.use_dc_path:
            # Expand channels then pixel shuffle
            x_dc = self.dc_conv(x)
            # Pixel shuffle: [B, C*4, H, W] -> [B, C, H*2, W*2]
            x_dc = F.pixel_shuffle(x_dc, upscale_factor=2)
            
            # Combine paths
            out = out + x_dc
        
        return out


if __name__ == '__main__':
    # Test Downsample
    print("Testing Downsample...")
    down = Downsample(256, 512, use_dc_path=True)
    x = torch.randn(2, 256, 64, 64)
    out = down(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Expected: [2, 512, 32, 32]")
    
    # Test without DC path
    down_no_dc = Downsample(256, 512, use_dc_path=False)
    out_no_dc = down_no_dc(x)
    print(f"Without DC path: {out_no_dc.shape}")
    
    # Test Upsample
    print("\nTesting Upsample...")
    up = Upsample(512, 256, use_dc_path=True)
    x = torch.randn(2, 512, 32, 32)
    out = up(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Expected: [2, 256, 64, 64]")
    
    # Test without DC path
    up_no_dc = Upsample(512, 256, use_dc_path=False)
    out_no_dc = up_no_dc(x)
    print(f"Without DC path: {out_no_dc.shape}")
    
    # Test round-trip
    print("\nTesting round-trip (down then up)...")
    x_orig = torch.randn(2, 256, 64, 64)
    x_down = down(x_orig)
    x_up = up(x_down)
    print(f"Original: {x_orig.shape} -> Down: {x_down.shape} -> Up: {x_up.shape}")
    print(f"Shape preserved: {x_orig.shape == x_up.shape}")
