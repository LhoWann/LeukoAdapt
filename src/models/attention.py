"""Spatial Attention Module for Attention-guided CycleGAN.

Based on Algorithm 1 and Algorithm 2 from Baydilli (2025).
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Channel-pooling spatial attention mechanism (CBAM-style).

    Computes spatial attention map using average and max pooling along
    channel dimension followed by 7x7 convolution and sigmoid activation.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize spatial attention module.

        Args:
            kernel_size: Size of convolution kernel (default 7).
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial attention to feature map.

        Args:
            feature_map: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple of (attended_feature_map, attention_weights).
        """
        avg_pool = torch.mean(feature_map, dim=1, keepdim=True)
        max_pool, _ = torch.max(feature_map, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.sigmoid(self.conv(concat))
        attended_features = attention_map * feature_map
        return attended_features, attention_map


class AttentionFusionModule(nn.Module):
    """Module As / At that fuses generator output with background.

    Computes foreground and background attention masks according to Algorithm 1:
        s_a = sigma(f_{7x7}([AvgPool(G(s)), MaxPool(G(s))]))
        s_b = (1 - s_a) * s
        s_f = s_a * G(s)
        s' = s_b + s_f
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize attention fusion module."""
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        original_image: torch.Tensor,
        generated_content: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute translated image via attention-guided blending.

        Args:
            original_image: Original input image s of shape (B, 3, H, W).
            generated_content: Content mask G(s) of shape (B, 3, H, W).

        Returns:
            Tuple of (translated_image s', attention_mask s_a).
        """
        avg_pool = torch.mean(generated_content, dim=1, keepdim=True)
        max_pool, _ = torch.max(generated_content, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_mask = self.sigmoid(self.conv(concat))

        foreground = attention_mask * generated_content
        background = (1.0 - attention_mask) * original_image
        translated_image = foreground + background
        return translated_image, attention_mask
