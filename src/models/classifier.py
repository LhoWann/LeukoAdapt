"""Classifier Architecture based on ResNet34.

Implements the feature extractor and binary classifier specified in
Section 5.1.2 of Baydilli (2025).
"""

import torch
import torch.nn as nn
import torchvision.models as models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeukemiaClassifier(nn.Module):
    """ResNet34 binary classifier for Acute Lymphocytic Leukemia.

    Inputs are expected in [-1, 1] (the pipeline-wide normalisation) and are converted internally to the
    ImageNet statistics that the pre-trained weights were fitted with.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        """Initialize ResNet34 classifier.

        Args:
            num_classes: Number of output classes (2 for ALL vs Normal).
            pretrained: Whether to load ImageNet pre-trained weights.
        """
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _to_imagenet_scale(self, x: torch.Tensor) -> torch.Tensor:
        return ((x * 0.5 + 0.5) - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 128, 128) in [-1, 1].

        Returns:
            Class logits of shape (B, num_classes).
        """
        return self.backbone(self._to_imagenet_scale(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings for t-SNE / analysis.

        Args:
            x: Input tensor of shape (B, 3, 128, 128) in [-1, 1].

        Returns:
            Feature embedding of shape (B, 512).
        """
        x = self._to_imagenet_scale(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)
