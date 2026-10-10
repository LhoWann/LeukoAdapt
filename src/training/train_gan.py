"""Training procedure for Attention-Guided CycleGAN.

Implements Algorithm 4 from Baydilli (2025) with mixed precision and memory optimization.
Terminal output matches PyTorch Lightning aesthetics.
"""

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import UnpairedLeukemiaDataset, get_default_transform
from src.models.cyclegan import AttentionCycleGAN, RealMask
from src.utils.image_pool import ImagePool
from src.utils.lightning_logger import (
    print_lightning_header,
    print_metrics_table,
    print_model_summary,
)

COLLAPSE_CHECK_EPOCH = 3
MIN_TRANSLATION_DELTA = 0.02
NUM_PREVIEW_IMAGES = 8


class LinearDecayLR:
    """Learning rate scheduler with linear decay after decay_epoch."""

    def __init__(self, total_epochs: int, decay_epoch: int) -> None:
        self.total_epochs = total_epochs
        self.decay_epoch = decay_epoch

    def step(self, epoch: int) -> float:
        """Compute learning rate multiplier for given epoch."""
        if epoch < self.decay_epoch:
            return 1.0
        return 1.0 - (epoch - self.decay_epoch) / (self.total_epochs - self.decay_epoch + 1)


def train_attention_cyclegan(
    source_dir: str,
    target_dir: str,
    output_dir: str = "checkpoints/cyclegan",
    epochs: int = 200,
    decay_epoch: int = 100,
    batch_size: int = 1,
    lr: float = 0.0001,
    lambda_gan: float = 0.5,
    lambda_cycle: float = 10.0,
    lambda_pixel: float = 1.0,
    real_mask: RealMask = "source",
    balance_target_classes: bool = True,
    buffer_size: int = 50,
    save_interval: int = 5,
    image_size: int = 128,
    use_amp: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> AttentionCycleGAN:
    """Train Attention-guided CycleGAN.

    Args:
        source_dir: Directory containing source domain images (C-NMC).
        target_dir: Directory containing target domain images (ALL-IDB train).
        output_dir: Directory to save generator and discriminator weights.
        epochs: Total training epochs (default 200).
        decay_epoch: Epoch at which linear learning rate decay begins (default 100).
        batch_size: Batch size per step (default 1).
        lr: Initial learning rate (default 1e-4).
        lambda_gan: Adversarial loss multiplier (default 0.5).
        lambda_cycle: Cycle consistency loss multiplier (default 10.0).
        lambda_pixel: Pixel loss multiplier (default 1.0).
        real_mask: Mask of the real discriminator input, see AttentionCycleGAN.
        balance_target_classes: Pad the minority target class with flipped copies (paper Section 5.1.2).
        buffer_size: History image pool capacity (default 50).
        save_interval: Save model weights and a preview grid every N epochs (default 5). The paper keeps the
            visually best of these checkpoints.
        image_size: Image resolution (default 128).
        use_amp: Whether to use FP16 automatic mixed precision.
        device: Target execution device.

    Returns:
        Trained AttentionCycleGAN model.

    """
    target_device = torch.device(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print_lightning_header(target_device)

    # Dataset & DataLoader
    transform = get_default_transform(image_size=image_size, is_train=True)
    dataset = UnpairedLeukemiaDataset(source_dir, target_dir, transform, balance_target_classes=balance_target_classes)
    print(f"GAN data: {len(dataset.source_paths)} source images, {len(dataset.target_items)} target images")
    eval_transform = get_default_transform(image_size=image_size, is_train=False)
    preview_step = max(1, len(dataset.source_paths) // NUM_PREVIEW_IMAGES)
    preview_paths = dataset.source_paths[::preview_step][:NUM_PREVIEW_IMAGES]
    preview_s = torch.stack([eval_transform(Image.open(p).convert("RGB")) for p in preview_paths]).to(target_device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # Initialize model
    model = AttentionCycleGAN(
        in_channels=3,
        num_res_blocks=6,
        lambda_gan=lambda_gan,
        lambda_cycle=lambda_cycle,
        lambda_pixel=lambda_pixel,
        real_mask=real_mask,
    ).to(target_device)

    print_model_summary(model, model_name="AttentionCycleGAN")

    # Optimizers
    gen_params = (
        list(model.gen_s2t.parameters())
        + list(model.gen_t2s.parameters())
        + list(model.attn_s.parameters())
        + list(model.attn_t.parameters())
    )
    optimizer_g = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(model.disc_t.parameters(), lr=lr, betas=(0.5, 0.999))

    # Schedulers
    lr_lambda = LinearDecayLR(total_epochs=epochs, decay_epoch=decay_epoch)
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda.step)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda.step)

    # Buffer & AMP Scaler
    masked_fake_pool = ImagePool(pool_size=buffer_size)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_stats: dict[str, float] = defaultdict(float)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:3d}/{epochs}",
            bar_format="{l_bar}{bar:25}{r_bar}",
            dynamic_ncols=True,
        )

        for step, (real_s, real_t) in enumerate(pbar, start=1):
            real_s = real_s.to(target_device)
            real_t = real_t.to(target_device)

            # ---------------------
            # Train Generator (G)
            # ---------------------
            optimizer_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                loss_g, g_stats, (masked_fake, _, s_a) = model.compute_generator_loss(real_s)

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)

            # -------------------------
            # Train Discriminator (D_T)
            # -------------------------
            optimizer_d.zero_grad()
            buffered_fake = masked_fake_pool.query(masked_fake.detach())

            with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
                masked_real = model.mask_real_target(real_t, s_a)
                loss_d, d_stats = model.compute_discriminator_loss(masked_real, buffered_fake)

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            for key, value in {**g_stats, **d_stats}.items():
                epoch_stats[key] += value

            pbar.set_postfix(
                {
                    "loss_g": f"{epoch_stats['loss_g_total'] / step:.3f}",
                    "loss_d": f"{epoch_stats['loss_d'] / step:.3f}",
                    "mask": f"{epoch_stats['mask_mean'] / step:.2f}",
                    "delta": f"{epoch_stats['translation_delta'] / step:.2f}",
                    "v_num": 0,
                }
            )

        scheduler_g.step()
        scheduler_d.step()

        avg_stats = {key: value / len(dataloader) for key, value in epoch_stats.items()}

        collapsed = epoch >= COLLAPSE_CHECK_EPOCH and avg_stats["translation_delta"] < MIN_TRANSLATION_DELTA
        if collapsed:
            delta, mask_mean = avg_stats["translation_delta"], avg_stats["mask_mean"]
            print(
                f"WARNING: translation collapsed at epoch {epoch} (real_mask='{real_mask}'): mean |s' - s| = {delta:.4f} "
                f"< {MIN_TRANSLATION_DELTA}, mask mean {mask_mean:.4f}. The attention mask closed, so s' = s."
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            epoch_metrics = {
                "loss_generator": avg_stats["loss_g_total"],
                "loss_discriminator": avg_stats["loss_d"],
                "loss_cycle": avg_stats["loss_cycle"],
                "loss_pixel": avg_stats["loss_pixel"],
                "mask_mean": avg_stats["mask_mean"],
                "translation_delta": avg_stats["translation_delta"],
                "fake_dark_fraction": avg_stats["fake_dark_fraction"],
                "learning_rate": optimizer_g.param_groups[0]["lr"],
            }
            print_metrics_table(epoch_metrics, title=f"Epoch {epoch} Progress Summary")

        if epoch % save_interval == 0 or epoch == epochs:
            save_file = out_path / f"attention_cyclegan_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "real_mask": real_mask,
                    "lambda_pixel": lambda_pixel,
                    "mask_mean": avg_stats["mask_mean"],
                    "translation_delta": avg_stats["translation_delta"],
                    "collapsed": collapsed,
                    "gen_s2t_state": model.gen_s2t.state_dict(),
                    "gen_t2s_state": model.gen_t2s.state_dict(),
                    "attn_s_state": model.attn_s.state_dict(),
                    "attn_t_state": model.attn_t.state_dict(),
                    "disc_t_state": model.disc_t.state_dict(),
                },
                save_file,
            )
            model.eval()
            with torch.no_grad():
                preview_fake, preview_mask = model.attn_s(preview_s, model.gen_s2t(preview_s))
            grid = torch.cat([preview_s, preview_fake, preview_mask.expand(-1, 3, -1, -1) * 2 - 1])
            save_image(grid * 0.5 + 0.5, out_path / f"preview_epoch_{epoch:03d}.png", nrow=len(preview_s))

    print("Attention-CycleGAN training finished successfully.")
    return model
