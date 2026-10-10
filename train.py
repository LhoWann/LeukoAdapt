"""Main CLI Pipeline Runner for ALL-IDB Generalization Research.

Provides commands to train Attention-CycleGAN, translate source images, train/evaluate the ResNet34 classifier
across evaluation scenarios, and run the comparison baselines (source-only, Reinhard, target-supervised).
All defaults follow the Baydilli (2025) protocol used for Scenario 1 (cell_level).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.data.stain_norm import reinhard_normalize_dir
from src.training.train_classifier import train_classifier
from src.training.train_gan import train_attention_cyclegan
from src.training.translate import translate_source_dataset


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Attention-guided CycleGAN and Classifier Pipeline for Leukemia Diagnosis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["gan", "translate", "classifier", "baseline", "all"],
        help="Stage to execute: 'gan', 'translate', 'classifier', 'baseline', or 'all' (gan + translate + classifier)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="cell_level",
        choices=["cell_level", "slide_level"],
        help="Evaluation scenario: 'cell_level' (random cell split) or 'slide_level' (overlap groups of slides in test)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["source_only", "reinhard", "target_supervised"],
        help="Baseline for --stage baseline",
    )
    parser.add_argument("--epochs_gan", type=int, default=200, help="Number of epochs for Attention-CycleGAN training")
    parser.add_argument("--decay_epoch_gan", type=int, default=100, help="Epoch to start linear LR decay for GAN")
    parser.add_argument("--epochs_clf", type=int, default=50, help="Number of epochs for ResNet34 classifier training")
    parser.add_argument("--batch_size_gan", type=int, default=1, help="Batch size for GAN training (1 for 4GB VRAM)")
    parser.add_argument("--batch_size_clf", type=int, default=32, help="Batch size for ResNet34 training")
    parser.add_argument("--lr_gan", type=float, default=0.0001, help="Initial learning rate for GAN")
    parser.add_argument("--lr_clf", type=float, default=0.001, help="Initial learning rate for Classifier")
    parser.add_argument("--lambda_gan", type=float, default=0.5, help="Adversarial loss weight")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle-consistency loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Pixel loss ||s - s'||_1 weight")
    parser.add_argument(
        "--real_mask",
        type=str,
        default="source",
        choices=["source", "target", "none"],
        help="Mask of the real D_T input: 'source' = s_a*t (paper Eq. 3), 'target' = t_a*t (UAIT), 'none' = full t",
    )
    parser.add_argument(
        "--no_target_balance",
        action="store_true",
        help="Do not pad the minority target class with flipped copies (paper: 410 ALL + 410 Normal)",
    )
    parser.add_argument(
        "--source_val",
        action="store_true",
        help="Select the classifier epoch on the C-NMC fold_2 validation set instead of using the last epoch (paper)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint_gan",
        type=str,
        default=None,
        help="Path to specific GAN checkpoint for translation (defaults to latest in checkpoints/cyclegan_<scenario>)",
    )
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    args = parser.parse_args()
    if args.stage == "baseline" and args.baseline is None:
        parser.error("--stage baseline requires --baseline")
    return args


def run_baseline(
    args: argparse.Namespace, source_dir: str, source_val_dir: str, target_train_dir: str, target_test_dir: str
) -> None:
    """Train the selected baseline classifier and evaluate it once on the target test set."""
    out_dir = f"checkpoints/baseline_{args.baseline}_{args.scenario}"
    train_dir, val_dir = source_dir, (source_val_dir if args.source_val else None)

    if args.baseline == "reinhard":
        train_dir = f"data/processed/reinhard_source/{args.scenario}/train"
        reinhard_normalize_dir(source_dir, train_dir, target_train_dir)
        if val_dir:
            val_dir = f"data/processed/reinhard_source/{args.scenario}/val"
            reinhard_normalize_dir(source_val_dir, val_dir, target_train_dir)
    elif args.baseline == "target_supervised":
        train_dir, val_dir = target_train_dir, None

    print(f"\n=== BASELINE: {args.baseline} | train={train_dir} | val={val_dir} ===")
    train_classifier(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=target_test_dir,
        output_dir=out_dir,
        epochs=args.epochs_clf,
        batch_size=args.batch_size_clf,
        lr=args.lr_clf,
        device=args.device,
    )


def main() -> None:
    """Execute selected pipeline stages."""
    args = parse_args()
    device = args.device
    use_amp = not args.no_amp

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve directories according to scenario
    source_dir = "data/processed/source_cnmc/train"
    source_val_dir = "data/processed/source_cnmc/val"
    target_train_dir = f"data/processed/target_all_idb/{args.scenario}/train"
    target_test_dir = f"data/processed/target_all_idb/{args.scenario}/test"
    translated_dir = f"data/processed/translated_source/{args.scenario}/train"
    translated_val_dir = f"data/processed/translated_source/{args.scenario}/val"
    ckpt_gan_dir = f"checkpoints/cyclegan_{args.scenario}"
    ckpt_clf_dir = f"checkpoints/classifier_{args.scenario}"

    print(f"\n[ALL-IDB Research Pipeline] Selected Scenario: {args.scenario} | Target Stage: {args.stage}")
    print(f"Device: {device} | Mixed Precision (AMP): {use_amp}\n")

    if args.stage == "baseline":
        run_baseline(args, source_dir, source_val_dir, target_train_dir, target_test_dir)
        return

    # Stage 1: Train Attention-CycleGAN
    if args.stage in ["gan", "all"]:
        print("\n=== STEP 1: Training Attention-Guided CycleGAN ===")
        train_attention_cyclegan(
            source_dir=source_dir,
            target_dir=target_train_dir,
            output_dir=ckpt_gan_dir,
            epochs=args.epochs_gan,
            decay_epoch=args.decay_epoch_gan,
            batch_size=args.batch_size_gan,
            lr=args.lr_gan,
            lambda_gan=args.lambda_gan,
            lambda_cycle=args.lambda_cycle,
            lambda_pixel=args.lambda_pixel,
            real_mask=args.real_mask,
            balance_target_classes=not args.no_target_balance,
            use_amp=use_amp,
            device=device,
        )

    # Stage 2: Translate source train and validation sets using the trained generator
    if args.stage in ["translate", "all"]:
        print("\n=== STEP 2: Translating C-NMC Source Images to Target Style ===")
        ckpt_path = args.checkpoint_gan
        if ckpt_path is None:
            ckpt_files = sorted(Path(ckpt_gan_dir).glob("*.pth"))
            if not ckpt_files:
                raise FileNotFoundError(f"No GAN checkpoint found in {ckpt_gan_dir}. Run '--stage gan' first.")
            ckpt_path = str(ckpt_files[-1])

        print(f"Using GAN Checkpoint: {ckpt_path} (inspect preview_epoch_*.png to pick another with --checkpoint_gan)")
        pairs = [(source_dir, translated_dir)] + ([(source_val_dir, translated_val_dir)] if args.source_val else [])
        for src, dst in pairs:
            translate_source_dataset(
                checkpoint_path=ckpt_path, source_dir=src, output_dir=dst, image_size=128, device=device
            )

    # Stage 3: Train Classifier on translated images (last epoch, or best translated source-val epoch), test once
    if args.stage in ["classifier", "all"]:
        print("\n=== STEP 3: Training ResNet34 Classifier on Translated Images ===")
        train_classifier(
            train_dir=translated_dir,
            val_dir=translated_val_dir if args.source_val else None,
            test_dir=target_test_dir,
            output_dir=ckpt_clf_dir,
            epochs=args.epochs_clf,
            batch_size=args.batch_size_clf,
            lr=args.lr_clf,
            device=device,
        )


if __name__ == "__main__":
    main()
