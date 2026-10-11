"""Unit tests for Attention-Guided CycleGAN and Classifier models.

Follows GEMINI.md rules: strict assertion handling, type annotations, and clean tests.
"""

import unittest

import torch

from src.models.attention import AttentionFusionModule, SpatialAttention
from src.models.classifier import LeukemiaClassifier
from src.models.cyclegan import AttentionCycleGAN
from src.models.discriminator import PatchDiscriminator
from src.models.generator import AttentionGenerator


class TestAttentionCycleGANModels(unittest.TestCase):
    """Test suite for neural network architectures."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 2
        self.img_size = 128
        self.device = torch.device("cpu")

    def test_spatial_attention_shape(self) -> None:
        """Verify SpatialAttention maintains feature map dimensions."""
        module = SpatialAttention(kernel_size=7)
        x = torch.randn(self.batch_size, 64, 32, 32)
        out, attn = module(x)

        if out.shape != (self.batch_size, 64, 32, 32):
            raise AssertionError(f"Expected shape (2, 64, 32, 32), got {out.shape}")
        if attn.shape != (self.batch_size, 1, 32, 32):
            raise AssertionError(f"Expected attention shape (2, 1, 32, 32), got {attn.shape}")
        if not (torch.all(attn >= 0.0) and torch.all(attn <= 1.0)):
            raise AssertionError("Attention weights must be within [0, 1]")

    def test_attention_fusion_module(self) -> None:
        """Verify AttentionFusionModule blending and mask shape."""
        module = AttentionFusionModule(kernel_size=7)
        s = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        g_s = torch.randn(self.batch_size, 3, self.img_size, self.img_size)

        s_prime, s_a = module(s, g_s)
        if s_prime.shape != (self.batch_size, 3, self.img_size, self.img_size):
            raise AssertionError(f"Expected translated shape (2, 3, 128, 128), got {s_prime.shape}")
        if s_a.shape != (self.batch_size, 1, self.img_size, self.img_size):
            raise AssertionError(f"Expected mask shape (2, 1, 128, 128), got {s_a.shape}")

    def test_generator_architecture(self) -> None:
        """Verify AttentionGenerator produces expected output dimensions."""
        gen = AttentionGenerator(in_channels=3, out_channels=3, num_res_blocks=6)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        out = gen(x)

        if out.shape != (self.batch_size, 3, self.img_size, self.img_size):
            raise AssertionError(f"Expected generator output (2, 3, 128, 128), got {out.shape}")

    def test_discriminator_architecture(self) -> None:
        """Verify PatchGAN Discriminator produces 14x14 patch predictions."""
        disc = PatchDiscriminator(in_channels=3)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        mask = torch.rand(self.batch_size, 1, self.img_size, self.img_size)

        out_unmasked = disc(x)
        out_masked = disc(x, mask=mask)

        if out_unmasked.shape != (self.batch_size, 1, 14, 14):
            raise AssertionError(f"Expected patch logits (2, 1, 14, 14), got {out_unmasked.shape}")
        if out_masked.shape != (self.batch_size, 1, 14, 14):
            raise AssertionError(f"Expected masked logits (2, 1, 14, 14), got {out_masked.shape}")

    def test_cyclegan_losses(self) -> None:
        """Verify AttentionCycleGAN composite loss calculations for every real-mask reading."""
        s = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        t = torch.randn(self.batch_size, 3, self.img_size, self.img_size)

        for real_mask in ("target", "source", "none"):
            model = AttentionCycleGAN(real_mask=real_mask, lambda_identity=5.0)
            loss_g, g_dict, (disc_fake, fake_t, s_a) = model.compute_generator_loss(s, t)
            loss_d, _ = model.compute_discriminator_loss(model.mask_real_target(t, s_a), disc_fake.detach())

            if loss_g.item() <= 0 or loss_d.item() <= 0:
                raise AssertionError(f"{real_mask}: losses must be positive, got G={loss_g.item()} D={loss_d.item()}")
            if not {"loss_gan", "loss_cycle", "loss_pixel", "loss_identity"} <= g_dict.keys():
                raise AssertionError(f"{real_mask}: missing loss terms in {sorted(g_dict)}")
            expected_fake = fake_t if real_mask == "none" else s_a * fake_t
            if not torch.allclose(disc_fake, expected_fake):
                raise AssertionError(f"{real_mask}: unexpected discriminator fake input")
            if real_mask == "none" and not torch.equal(model.mask_real_target(t, s_a), t):
                raise AssertionError("real_mask 'none' must feed the full target image")

    def test_closed_mask_is_identity(self) -> None:
        """A fully closed attention mask returns the source unchanged (the collapse of the literal Eq. 3)."""
        module = AttentionFusionModule(kernel_size=7)
        torch.nn.init.zeros_(module.conv.weight)
        module.conv.weight.data[:, 0] = -1e3
        s = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        g_s = torch.rand(self.batch_size, 3, self.img_size, self.img_size) + 0.5

        s_prime, s_a = module(s, g_s)
        if s_a.max().item() > 1e-6 or not torch.allclose(s_prime, s):
            raise AssertionError("A closed mask must give s' = s")

    def test_classifier_forward(self) -> None:
        """Verify ResNet34 LeukemiaClassifier output logits and feature extraction."""
        clf = LeukemiaClassifier(num_classes=2, pretrained=False)
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)

        logits = clf(x)
        features = clf.extract_features(x)

        if logits.shape != (self.batch_size, 2):
            raise AssertionError(f"Expected logits (2, 2), got {logits.shape}")
        if features.shape != (self.batch_size, 512):
            raise AssertionError(f"Expected features (2, 512), got {features.shape}")


if __name__ == "__main__":
    unittest.main()
