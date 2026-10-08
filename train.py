"""Main CLI Pipeline Runner for ALL-IDB Generalization Research.

Provides commands to train Attention-CycleGAN, translate source images,
and train/evaluate the ResNet34 classifier across evaluation scenarios.
"""

import argparse
from pathlib import Path

import torch

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
        choices=["gan", "translate", "classifier", "all"],
        help="Stage to execute: 'gan', 'translate', 'classifier', or 'all'",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="cell_level",
        choices=["cell_level", "patient_level"],
        help="Evaluation scenario: 'cell_level' (Baydilli baseline) or 'patient_level' (leak-free test)",
    )
    parser.add_argument(
        "--epochs_gan",
        type=int,
        default=200,
        help="Number of epochs for Attention-CycleGAN training",
    )
    parser.add_argument(
        "--decay_epoch_gan",
        type=int,
        default=100,
        help="Epoch to start linear LR decay for GAN",
    )
    parser.add_argument(
        "--epochs_clf",
        type=int,
        default=50,
        help="Number of epochs for ResNet34 classifier training",
    )
    parser.add_argument(
        "--batch_size_gan",
        type=int,
        default=1,
        help="Batch size for GAN training (1 recommended for 4GB VRAM)",
    )
    parser.add_argument(
        "--batch_size_clf",
        type=int,
        default=32,
        help="Batch size for ResNet34 training",
    )
    parser.add_argument(
        "--lr_gan",
        type=float,
        default=0.0001,
        help="Initial learning rate for GAN",
    )
    parser.add_argument(
        "--lr_clf",
        type=float,
        default=0.001,
        help="Initial learning rate for Classifier",
    )
    parser.add_argument(
        "--checkpoint_gan",
        type=str,
        default=None,
        help="Path to specific GAN checkpoint for translation (defaults to latest in checkpoints/cyclegan)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    """Execute selected pipeline stages."""
    args = parse_args()
    device = args.device
    use_amp = not args.no_amp

    # Resolve directories according to scenario
    source_dir = "data/processed/source_cnmc/train"
    target_train_dir = f"data/processed/target_all_idb/{args.scenario}/train"
    target_test_dir = f"data/processed/target_all_idb/{args.scenario}/test"
    translated_dir = f"data/processed/translated_source/{args.scenario}/train"
    ckpt_gan_dir = f"checkpoints/cyclegan_{args.scenario}"
    ckpt_clf_dir = f"checkpoints/classifier_{args.scenario}"

    print(f"\n[ALL-IDB Research Pipeline] Selected Scenario: {args.scenario} | Target Stage: {args.stage}")
    print(f"Device: {device} | Mixed Precision (AMP): {use_amp}\n")

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
            use_amp=use_amp,
            device=device,
        )

    # Stage 2: Translate Source Dataset using trained Generator
    if args.stage in ["translate", "all"]:
        print("\n=== STEP 2: Translating C-NMC Source Images to Target Style ===")
        ckpt_path = args.checkpoint_gan
        if ckpt_path is None:
            # Find latest checkpoint in ckpt_gan_dir
            ckpt_files = sorted(Path(ckpt_gan_dir).glob("*.pth"))
            if not ckpt_files:
                raise FileNotFoundError(f"No GAN checkpoint found in {ckpt_gan_dir}. Run '--stage gan' first.")
            ckpt_path = str(ckpt_files[-1])

        print(f"Using GAN Checkpoint: {ckpt_path}")
        translate_source_dataset(
            checkpoint_path=ckpt_path,
            source_dir=source_dir,
            output_dir=translated_dir,
            image_size=128,
            device=device,
        )

    # Stage 3: Train Classifier & Evaluate on Target Test Set
    if args.stage in ["classifier", "all"]:
        print("\n=== STEP 3: Training ResNet34 Classifier on Translated Images ===")
        train_classifier(
            train_dir=translated_dir,
            test_dir=target_test_dir,
            output_dir=ckpt_clf_dir,
            epochs=args.epochs_clf,
            batch_size=args.batch_size_clf,
            lr=args.lr_clf,
            device=device,
        )


if __name__ == "__main__":
    main()
