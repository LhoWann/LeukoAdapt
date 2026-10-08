"""Attention-guided Generator Architecture.

Implements the generator architecture described in Section 4.1 and Algorithm 3
of Baydilli (2025).
"""

import torch
import torch.nn as nn

from src.models.attention import SpatialAttention


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and Instance Normalization."""

    def __init__(self, channels: int) -> None:
        """Initialize residual block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip addition."""
        return x + self.block(x)


class AttentionGenerator(nn.Module):
    """Attention-guided CycleGAN Generator with skip connections.

    Consists of 3 encoder blocks with spatial attention, 6 residual blocks,
    and 2 transposed convolution decoder blocks with spatial attention and
    U-Net style skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_res_blocks: int = 6) -> None:
        """Initialize attention generator.

        Args:
            in_channels: Number of input image channels (default 3).
            out_channels: Number of output image channels (default 3).
            num_res_blocks: Number of residual blocks in bottleneck (default 6).
        """
        super().__init__()

        # Encoder blocks
        # Block 0: 7x7 conv, stride 1, 64 channels
        self.enc0_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
        )
        self.enc0_attn = SpatialAttention()

        # Block 1: 3x3 conv, stride 2, 128 channels
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
        )
        self.enc1_attn = SpatialAttention()

        # Block 2: 3x3 conv, stride 2, 256 channels
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(inplace=True),
        )
        self.enc2_attn = SpatialAttention()

        # Transformer / Bottleneck: 6 Residual Blocks
        res_blocks = [ResidualBlock(256) for _ in range(num_res_blocks)]
        self.bottleneck = nn.Sequential(*res_blocks)

        # Decoder blocks with skip connection concatenation
        # Dec 0: Transposed conv 256 -> 128, stride 2
        self.dec0_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(inplace=True),
        )
        self.dec0_attn = SpatialAttention()

        # Dec 1: Transposed conv (128 + 128 = 256) -> 64, stride 2
        self.dec1_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
        )
        self.dec1_attn = SpatialAttention()

        # Output block: 7x7 conv (64 + 64 = 128) -> 3, stride 1, Tanh
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, out_channels, kernel_size=7, stride=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of generator.

        Args:
            x: Input tensor of shape (B, 3, 128, 128).

        Returns:
            Generated content mask G(x) of shape (B, 3, 128, 128).
        """
        # Encoder with attention and skip recording
        x0, _ = self.enc0_attn(self.enc0_conv(x))  # (B, 64, 128, 128)
        x1, _ = self.enc1_attn(self.enc1_conv(x0))  # (B, 128, 64, 64)
        x2, _ = self.enc2_attn(self.enc2_conv(x1))  # (B, 256, 32, 32)

        # Bottleneck
        b = self.bottleneck(x2)  # (B, 256, 32, 32)

        # Decoder with attention and skip concatenation
        d0, _ = self.dec0_attn(self.dec0_deconv(b))  # (B, 128, 64, 64)
        d0 = torch.cat([d0, x1], dim=1)  # (B, 256, 64, 64)

        d1, _ = self.dec1_attn(self.dec1_deconv(d0))  # (B, 64, 128, 128)
        d1 = torch.cat([d1, x0], dim=1)  # (B, 128, 128, 128)

        out = self.out_conv(d1)  # (B, 3, 128, 128)
        return out
