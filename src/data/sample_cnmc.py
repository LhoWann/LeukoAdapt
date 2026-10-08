"""Sampling and preprocessing pipeline for C-NMC 2019 source dataset.

Selects 1000 ALL and 1000 HEM (Normal) images with patient-level tracking,
resizes them to 128x128, and organizes them in data/processed/source_cnmc.
"""

import json
from pathlib import Path

from PIL import Image


def sample_cnmc_dataset(
    raw_dir: str = "data/raw/C-NMC_2019",
    output_dir: str = "data/processed/source_cnmc/train",
    target_all_count: int = 1000,
    target_hem_count: int = 1000,
    preferred_fold: str = "fold_0",
    image_size: int = 128,
) -> tuple[int, int]:
    """Sample and resize balanced subset from C-NMC 2019 dataset.

    Args:
        raw_dir: Directory containing C-NMC folds.
        output_dir: Destination processed directory.
        target_all_count: Number of ALL images to sample (default 1000).
        target_hem_count: Number of Normal (HEM) images to sample (default 1000).
        preferred_fold: Which fold to prioritize for patient consistency (default fold_0).
        image_size: Output image dimension (default 128x128).

    Returns:
        Tuple of (num_all_saved, num_hem_saved).
    """
    raw_path = Path(raw_dir)
    out_all_dir = Path(output_dir) / "all"
    out_hem_dir = Path(output_dir) / "hem"
    out_all_dir.mkdir(parents=True, exist_ok=True)
    out_hem_dir.mkdir(parents=True, exist_ok=True)

    # Collect ALL images starting from preferred fold
    all_files: list[Path] = []
    preferred_all_dir = raw_path / preferred_fold / "all"
    if preferred_all_dir.exists():
        all_files.extend(sorted(preferred_all_dir.glob("*.bmp")))

    # Collect HEM images starting from preferred fold
    hem_files: list[Path] = []
    preferred_hem_dir = raw_path / preferred_fold / "hem"
    if preferred_hem_dir.exists():
        hem_files.extend(sorted(preferred_hem_dir.glob("*.bmp")))

    # Fallback to other folds if necessary
    for fold in ["fold_0", "fold_1", "fold_2"]:
        if fold == preferred_fold:
            continue
        if len(all_files) < target_all_count:
            all_files.extend(sorted((raw_path / fold / "all").glob("*.bmp")))
        if len(hem_files) < target_hem_count:
            hem_files.extend(sorted((raw_path / fold / "hem").glob("*.bmp")))

    selected_all = all_files[:target_all_count]
    selected_hem = hem_files[:target_hem_count]

    metadata = {"all_samples": [], "hem_samples": []}

    print(f"Sampling and resizing {len(selected_all)} ALL images...")
    for idx, file_path in enumerate(selected_all):
        out_name = f"cnmc_all_{idx:04d}.png"
        out_file = out_all_dir / out_name
        with Image.open(file_path) as img:
            img_resized = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            img_resized.save(out_file)
        metadata["all_samples"].append({"id": out_name, "source": str(file_path)})

    print(f"Sampling and resizing {len(selected_hem)} HEM images...")
    for idx, file_path in enumerate(selected_hem):
        out_name = f"cnmc_hem_{idx:04d}.png"
        out_file = out_hem_dir / out_name
        with Image.open(file_path) as img:
            img_resized = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            img_resized.save(out_file)
        metadata["hem_samples"].append({"id": out_name, "source": str(file_path)})

    meta_file = Path(output_dir).parent / "metadata_cnmc.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Sampling completed: {len(selected_all)} ALL, {len(selected_hem)} HEM saved to {output_dir}")
    return len(selected_all), len(selected_hem)


if __name__ == "__main__":
    sample_cnmc_dataset()
