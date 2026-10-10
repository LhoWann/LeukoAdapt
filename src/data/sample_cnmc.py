"""Sampling and preprocessing pipeline for C-NMC 2019 source dataset.

Draws subject-balanced, class-balanced subsets (round-robin over the subject IDs encoded in the file names),
resizes them to 128x128 and writes:
- data/processed/source_cnmc/train: 1000 ALL + 1000 HEM from fold_0
- data/processed/source_cnmc/val:   500 ALL + 500 HEM from fold_2 (subject-disjoint from train)
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def subject_id(path: Path) -> str:
    """Return the subject ID encoded in a C-NMC file name (UID_<subject>_<cell>_<image>_<class>.bmp)."""
    return path.name.split("_")[1].upper()


def sample_round_robin(files: list[Path], count: int, rng: np.random.Generator) -> list[Path]:
    """Pick files by cycling through subjects so no subject dominates the sample.

    Args:
        files: Candidate image paths.
        count: Number of paths to select.
        rng: Random generator used to shuffle the cells inside each subject.

    Returns:
        Selected paths (fewer than count only if not enough candidates exist).
    """
    by_subject: dict[str, list[Path]] = {}
    for path in files:
        by_subject.setdefault(subject_id(path), []).append(path)
    queues = {subject: [paths[i] for i in rng.permutation(len(paths))] for subject, paths in sorted(by_subject.items())}

    selected: list[Path] = []
    while len(selected) < count and any(queues.values()):
        for queue in queues.values():
            if queue and len(selected) < count:
                selected.append(queue.pop())
    return selected


def sample_cnmc_split(
    raw_dir: str,
    fold: str,
    output_dir: str,
    count_all: int,
    count_hem: int,
    image_size: int = 128,
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """Sample, resize and save one subject-balanced split of C-NMC.

    Args:
        raw_dir: Directory containing C-NMC folds.
        fold: Fold to draw from (e.g. fold_0).
        output_dir: Destination directory with 'all' and 'hem' subfolders (recreated on every call).
        count_all: Number of ALL images.
        count_hem: Number of HEM (Normal) images.
        image_size: Output image dimension (default 128x128).
        seed: Random seed.

    Returns:
        Metadata dictionary with the source path and subject of every saved image.
    """
    rng = np.random.default_rng(seed)
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)

    metadata: dict[str, list[dict[str, str]]] = {"all_samples": [], "hem_samples": []}
    for cls, count in (("all", count_all), ("hem", count_hem)):
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted((Path(raw_dir) / fold / cls).glob("*.bmp"))
        selected = sample_round_robin(candidates, count, rng)
        print(
            f"Sampling {len(selected)} {cls.upper()} images from {fold} ({len({subject_id(p) for p in selected})} subjects)"
        )
        for idx, file_path in enumerate(selected):
            out_name = f"cnmc_{cls}_{idx:04d}.png"
            with Image.open(file_path) as img:
                img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC).save(out_dir / out_name)
            metadata[f"{cls}_samples"].append(
                {"id": out_name, "source": str(file_path), "subject": subject_id(file_path)}
            )
    return metadata


def sample_cnmc_dataset(
    raw_dir: str = "data/raw/C-NMC_2019",
    output_root: str = "data/processed/source_cnmc",
    train_fold: str = "fold_0",
    val_fold: str = "fold_2",
    train_count: int = 1000,
    val_count: int = 500,
    image_size: int = 128,
    seed: int = 42,
) -> None:
    """Create the subject-disjoint source train and validation sets.

    Args:
        raw_dir: Directory containing C-NMC folds.
        output_root: Root directory receiving train/ and val/ plus metadata files.
        train_fold: Fold used for the 2000-image training cohort.
        val_fold: Fold used for source validation (chosen for its subject diversity).
        train_count: Images per class in the training split.
        val_count: Images per class in the validation split.
        image_size: Output image dimension.
        seed: Random seed.

    Raises:
        ValueError: If a subject appears in both the training and validation splits.
    """
    train_meta = sample_cnmc_split(
        raw_dir, train_fold, f"{output_root}/train", train_count, train_count, image_size, seed
    )
    val_meta = sample_cnmc_split(raw_dir, val_fold, f"{output_root}/val", val_count, val_count, image_size, seed)

    for key in ("all_samples", "hem_samples"):
        shared = {x["subject"] for x in train_meta[key]} & {x["subject"] for x in val_meta[key]}
        if shared:
            raise ValueError(f"Subjects shared between train and val for {key}: {sorted(shared)}")

    for name, meta in (("metadata_cnmc.json", train_meta), ("metadata_cnmc_val.json", val_meta)):
        with open(Path(output_root) / name, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    sample_cnmc_dataset()
