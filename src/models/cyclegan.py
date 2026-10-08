"""Full Attention-Guided CycleGAN model assembly.

Integrates G_{S->T}, F_{T->S}, A_S, A_T, and D_T according to Baydilli (2025).
"""

import torch
import torch.nn as nn

from src.models.attention import AttentionFusionModule
from src.models.discriminator import PatchDiscriminator
from src.models.generator import AttentionGenerator


class AttentionCycleGAN(nn.Module):
    """Attention-guided CycleGAN model for unpaired domain adaptation.

    Translates segmented source images (C-NMC) to target style (ALL-IDB)
    with attention guidance preserving cellular morphology.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_res_blocks: int = 6,
        lambda_gan: float = 0.5,
        lambda_cycle: float = 10.0,
        lambda_pixel: float = 1.0,
    ) -> None:
        """Initialize Attention-Guided CycleGAN.

        Args:
            in_channels: Number of image channels (default 3).
            num_res_blocks: Number of residual blocks in generator bottleneck (default 6).
            lambda_gan: Weight for adversarial loss (default 0.5).
            lambda_cycle: Weight for cycle-consistency loss (default 10.0).
            lambda_pixel: Weight for pixel identity loss (default 1.0).
        """
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_cycle = lambda_cycle
        self.lambda_pixel = lambda_pixel

        # Generators
        self.gen_s2t = AttentionGenerator(in_channels, in_channels, num_res_blocks)
        self.gen_t2s = AttentionGenerator(in_channels, in_channels, num_res_blocks)

        # Attention Modules
        self.attn_s = AttentionFusionModule(kernel_size=7)
        self.attn_t = AttentionFusionModule(kernel_size=7)

        # Discriminator for target domain
        self.disc_t = PatchDiscriminator(in_channels)

        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_pixel = nn.L1Loss()

    def forward_source(self, real_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Translate source image to target domain and recover it.

        Args:
            real_s: Source image of shape (B, 3, 128, 128).

        Returns:
            Tuple of (fake_t, s_a, recovered_s).
        """
        g_s = self.gen_s2t(real_s)
        fake_t, s_a = self.attn_s(real_s, g_s)
        f_fake_t = self.gen_t2s(fake_t)
        recovered_s, _ = self.attn_t(fake_t, f_fake_t)
        return fake_t, s_a, recovered_s

    def compute_generator_loss(
        self,
        real_s: torch.Tensor,
        _real_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute generator total loss according to Eq. (6) and Eq. (8).

        Args:
            real_s: Real source image batch.
            real_t: Real target image batch.

        Returns:
            Tuple of (total_loss, loss_dict, (fake_t, s_a, recovered_s)).
        """
        fake_t, s_a, recovered_s = self.forward_source(real_s)

        # Adversarial loss: generator wants D_T(s_a * fake_t) to be classified as 1 (real)
        pred_fake = self.disc_t(fake_t, mask=s_a)
        target_ones = torch.ones_like(pred_fake)
        loss_gan = self.criterion_gan(pred_fake, target_ones)

        # Cycle-consistency loss: || s - s'' ||_1
        loss_cycle = self.criterion_cycle(recovered_s, real_s)

        # Pixel loss: || s - s' ||_1
        loss_pixel = self.criterion_pixel(fake_t, real_s)

        total_loss = self.lambda_gan * loss_gan + self.lambda_cycle * loss_cycle + self.lambda_pixel * loss_pixel

        loss_dict = {
            "loss_g_total": total_loss.item(),
            "loss_gan": loss_gan.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_pixel": loss_pixel.item(),
        }
        return total_loss, loss_dict, (fake_t, s_a, recovered_s)

    def compute_discriminator_loss(
        self,
        real_t: torch.Tensor,
        fake_t_buffered: torch.Tensor,
        s_a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute discriminator LSGAN loss according to Eq. (8).

        Args:
            real_t: Real target image batch.
            fake_t_buffered: Fake target image batch (possibly sampled from history buffer).
            s_a: Attention mask from source image.

        Returns:
            Tuple of (loss_d, loss_dict).
        """
        pred_real = self.disc_t(real_t, mask=s_a)
        target_ones = torch.ones_like(pred_real)
        loss_real = self.criterion_gan(pred_real, target_ones)

        pred_fake = self.disc_t(fake_t_buffered.detach(), mask=s_a.detach())
        target_zeros = torch.zeros_like(pred_fake)
        loss_fake = self.criterion_gan(pred_fake, target_zeros)

        # Per CycleGAN convention and paper Section 4.2, divided by 2
        loss_d = 0.5 * (loss_real + loss_fake)

        loss_dict = {
            "loss_d": loss_d.item(),
            "loss_d_real": loss_real.item(),
            "loss_d_fake": loss_fake.item(),
        }
        return loss_d, loss_dict
