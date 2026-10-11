"""PyTorch Dataset implementations for Unpaired Domain Adaptation and Classification."""

import os
import random
from collections.abc import Callable
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.data.backgrounds import composite_on_background, load_background_paths


def get_default_transform(image_size: int = 128, is_train: bool = True) -> Callable:
    """Return standard image transformations for training/testing.

    Args:
        image_size: Target square image dimensions (default 128).
        is_train: Whether transformations include random horizontal flip.

    Returns:
        Compose transform pipeline mapping PIL Image to normalized Tensor [-1, 1].
    """
    transforms_list = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
    ]
    if is_train:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))
    transforms_list.extend(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return T.Compose(transforms_list)


class UnpairedLeukemiaDataset(Dataset):
    """Unpaired dataset loader for CycleGAN training across domains.

    Loads images from source domain (C-NMC) and target domain (ALL-IDB).
    """

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        transform: Callable | None = None,
        balance_target_classes: bool = False,
        background_dir: str | None = None,
    ) -> None:
        """Initialize Unpaired Dataset.

        Args:
            source_dir: Directory containing source domain images.
            target_dir: Directory containing target domain images.
            transform: Optional transform applied to both source and target images.
            balance_target_classes: Pad the minority target class ('all'/'hem' subfolders) with horizontally flipped
                copies up to the majority count, as in Baydilli (2025) Section 5.1.2 (paper: 410 ALL + 249 Normal ->
                410 ALL + 410 Normal; here 410 ALL + 25 Normal -> 410 + 410).
            background_dir: Optional bank of target background patches; when given, every source cell is pasted onto
                a randomly drawn patch before the transform.
        """
        self.source_paths = self._load_image_paths(source_dir)
        target_paths = self._load_image_paths(target_dir)

        if not self.source_paths:
            raise ValueError(f"No valid images found in source_dir: {source_dir}")
        if not target_paths:
            raise ValueError(f"No valid images found in target_dir: {target_dir}")

        self.target_items: list[tuple[Path, bool]] = [(p, False) for p in target_paths]
        if balance_target_classes:
            by_class = [self._load_image_paths(str(Path(target_dir) / cls)) for cls in ("all", "hem")]
            minority, majority = sorted(by_class, key=len)
            if not minority:
                raise ValueError(f"balance_target_classes needs non-empty 'all' and 'hem' folders in {target_dir}")
            deficit = len(majority) - len(minority)
            self.target_items += [(minority[i % len(minority)], True) for i in range(deficit)]

        self.backgrounds = load_background_paths(background_dir) if background_dir else []
        self.transform = transform or get_default_transform(image_size=128, is_train=True)

    @staticmethod
    def _load_image_paths(directory: str) -> list[Path]:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in valid_extensions:
                    paths.append(Path(root) / f)
        return sorted(paths)

    def __len__(self) -> int:
        """Length is maximum between source and target sets."""
        return max(len(self.source_paths), len(self.target_items))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get unpaired sample (s, t)."""
        idx_s = index % len(self.source_paths)
        target_path, flip_target = self.target_items[random.randint(0, len(self.target_items) - 1)]

        img_s = Image.open(self.source_paths[idx_s]).convert("RGB")
        if self.backgrounds:
            img_s = composite_on_background(img_s, Image.open(random.choice(self.backgrounds)))
        img_t = Image.open(target_path).convert("RGB")
        if flip_target:
            img_t = img_t.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        tensor_s = self.transform(img_s)
        tensor_t = self.transform(img_t)

        return tensor_s, tensor_t


class LeukemiaClassificationDataset(Dataset):
    """Labeled dataset loader for ResNet34 classifier training and evaluation.

    Expects directory structure:
        dataset_dir/
            all/   (Class 1: ALL blast cells)
            hem/   (Class 0: Normal leukocytes)
    """

    def __init__(self, root_dir: str, transform: Callable | None = None) -> None:
        """Initialize Classification Dataset.

        Args:
            root_dir: Root directory containing 'all' and 'hem' subdirectories.
            transform: Optional transform applied to images.
        """
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform or get_default_transform(image_size=128, is_train=False)

        all_dir = Path(root_dir) / "all"
        hem_dir = Path(root_dir) / "hem"

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        if all_dir.exists():
            for p in all_dir.glob("*"):
                if p.suffix.lower() in valid_extensions:
                    self.samples.append((p, 1))

        if hem_dir.exists():
            for p in hem_dir.glob("*"):
                if p.suffix.lower() in valid_extensions:
                    self.samples.append((p, 0))

        if not self.samples:
            raise ValueError(f"No labeled images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(image)
        return tensor_img, label
