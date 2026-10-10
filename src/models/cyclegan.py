"""Full Attention-Guided CycleGAN model assembly.

Integrates G_{S->T}, F_{T->S}, A_S, A_T, and D_T according to Baydilli (2025), Algorithm 4.
"""

from typing import Literal

import torch
import torch.nn as nn

from src.models.attention import AttentionFusionModule
from src.models.discriminator import PatchDiscriminator
from src.models.generator import AttentionGenerator

DARK_THRESHOLD = -0.84
RealMask = Literal["source", "target", "none"]


class AttentionCycleGAN(nn.Module):
    """Attention-guided CycleGAN model for unpaired domain adaptation.

    Translates segmented source images (C-NMC) to target style (ALL-IDB). Only the S -> T -> S cycle and the target
    discriminator D_T are trained; the paper leaves the T -> S direction out of scope.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_res_blocks: int = 6,
        lambda_gan: float = 0.5,
        lambda_cycle: float = 10.0,
        lambda_pixel: float = 1.0,
        real_mask: RealMask = "source",
    ) -> None:
        """Initialize Attention-Guided CycleGAN.

        Args:
            in_channels: Number of image channels.
            num_res_blocks: Number of residual blocks in generator bottleneck.
            lambda_gan: Weight of the adversarial loss, Eq. (8).
            lambda_cycle: Weight of the cycle-consistency loss ||s - s''||_1, Eq. (4).
            lambda_pixel: Weight of the pixel loss ||s - s'||_1, Eq. (5).
            real_mask: Mask applied to the real target image fed to D_T; the fake is always s_a * s'. "source" is the
                literal Eq. (3), s_a * t. It has a degenerate optimum: s_a = 0 makes both discriminator inputs zero and
                gives s' = s, and on C-NMC -> ALL-IDB it collapses there within one epoch even with lambda_pixel = 0.
                "target" masks t with its own attention map t_a = A_T(t, F(t)) as in UAIT [21]; "none" feeds the
                full t. Both are documented deviations from the paper.
        """
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_cycle = lambda_cycle
        self.lambda_pixel = lambda_pixel
        self.real_mask = real_mask

        self.gen_s2t = AttentionGenerator(in_channels, in_channels, num_res_blocks)
        self.gen_t2s = AttentionGenerator(in_channels, in_channels, num_res_blocks)
        self.attn_s = AttentionFusionModule(kernel_size=7)
        self.attn_t = AttentionFusionModule(kernel_size=7)
        self.disc_t = PatchDiscriminator(in_channels)

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_pixel = nn.L1Loss()

    def forward_source(self, real_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Translate source image to target domain and recover it (Algorithm 4, lines 2-5).

        Args:
            real_s: Source image of shape (B, 3, 128, 128).

        Returns:
            Tuple of (fake_t s', s_a, recovered_s s'').
        """
        fake_t, s_a = self.attn_s(real_s, self.gen_s2t(real_s))
        recovered_s, _ = self.attn_t(fake_t, self.gen_t2s(fake_t))
        return fake_t, s_a, recovered_s

    def compute_generator_loss(
        self,
        real_s: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute the generator objective of Eq. (6) with the least-squares adversarial term of Eq. (8).

        Args:
            real_s: Real source image batch.

        Returns:
            Tuple of (total_loss, loss_dict, (masked_fake s_a * s', fake_t s', s_a)).
        """
        fake_t, s_a, recovered_s = self.forward_source(real_s)
        masked_fake = s_a * fake_t

        pred_fake = self.disc_t(masked_fake)
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        loss_cycle = self.criterion_cycle(recovered_s, real_s)
        loss_pixel = self.criterion_pixel(fake_t, real_s)

        total_loss = self.lambda_gan * loss_gan + self.lambda_cycle * loss_cycle + self.lambda_pixel * loss_pixel

        with torch.no_grad():
            loss_dict = {
                "loss_g_total": total_loss.item(),
                "loss_gan": loss_gan.item(),
                "loss_cycle": loss_cycle.item(),
                "loss_pixel": loss_pixel.item(),
                "mask_mean": s_a.mean().item(),
                "translation_delta": (fake_t - real_s).abs().mean().item(),
                "fake_dark_fraction": (fake_t.max(dim=1).values < DARK_THRESHOLD).float().mean().item(),
            }
        return total_loss, loss_dict, (masked_fake, fake_t, s_a)

    def mask_real_target(self, real_t: torch.Tensor, s_a: torch.Tensor) -> torch.Tensor:
        """Build the real discriminator input according to real_mask (no gradient flows to the generators).

        Args:
            real_t: Real target image batch.
            s_a: Source attention mask of the current step.

        Returns:
            Masked real target batch.
        """
        if self.real_mask == "source":
            return s_a.detach() * real_t
        if self.real_mask == "target":
            with torch.no_grad():
                _, t_a = self.attn_t(real_t, self.gen_t2s(real_t))
            return t_a * real_t
        return real_t

    def compute_discriminator_loss(
        self,
        masked_real_t: torch.Tensor,
        masked_fake_buffered: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the discriminator LSGAN loss of Eq. (8), halved as in Section 4.2.

        Args:
            masked_real_t: Masked real target batch from mask_real_target.
            masked_fake_buffered: Masked fake batch s_a * s', sampled from the history buffer. The buffer stores masked
                images so every fake keeps the mask it was generated with.

        Returns:
            Tuple of (loss_d, loss_dict).
        """
        pred_real = self.disc_t(masked_real_t)
        loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))

        pred_fake = self.disc_t(masked_fake_buffered.detach())
        loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))

        loss_d = 0.5 * (loss_real + loss_fake)
        loss_dict = {
            "loss_d": loss_d.item(),
            "loss_d_real": loss_real.item(),
            "loss_d_fake": loss_fake.item(),
        }
        return loss_d, loss_dict
