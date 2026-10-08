"""PatchGAN Discriminator Architecture.

Implements the 70x70 PatchGAN discriminator described in Section 4.1 of
Baydilli (2025).
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator.

    Produces a 14x14 patch-level real/fake decision map from a 128x128 input.
    Supports attention masking during training.
    """

    def __init__(self, in_channels: int = 3) -> None:
        """Initialize PatchGAN discriminator.

        Args:
            in_channels: Number of input channels (default 3).
        """
        super().__init__()
        # Layer 1: 4x4 conv, stride 2, pad 1 -> (B, 64, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Layer 2: 4x4 conv, stride 2, pad 1 -> (B, 128, 32, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Layer 3: 4x4 conv, stride 2, pad 1 -> (B, 256, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Layer 4: 4x4 conv, stride 1, pad 1 -> (B, 512, 15, 15)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Final output layer: 4x4 conv, stride 1, pad 1 -> (B, 1, 14, 14)
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image of shape (B, 3, 128, 128).
            mask: Optional attention mask of shape (B, 1, 128, 128) or (B, 3, 128, 128).

        Returns:
            Output logits of shape (B, 1, 14, 14).
        """
        if mask is not None:
            x = x * mask
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.final_conv(x)
