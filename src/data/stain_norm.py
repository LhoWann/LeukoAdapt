"""Reinhard colour normalisation baseline for segmented C-NMC cells.

Matches the per-channel LAB mean and standard deviation of every source cell (foreground pixels only, the black
segmentation background is left untouched) to the statistics of the target training set.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

FOREGROUND_MIN_VALUE = 20
MIN_FOREGROUND_PIXELS = 50


def _load_lab(path: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32), bgr


def target_lab_statistics(reference_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel LAB mean and std over every pixel of the reference images.

    Args:
        reference_dir: Directory with 'all' and 'hem' subfolders of target training images.

    Returns:
        Tuple of (mean, std), each of shape (3,).
    """
    pixels = []
    for path in sorted(Path(reference_dir).rglob("*.png")):
        lab, _ = _load_lab(path)
        pixels.append(lab.reshape(-1, 3))
    stacked = np.concatenate(pixels)
    return stacked.mean(axis=0), stacked.std(axis=0)


def reinhard_normalize_dir(source_dir: str, output_dir: str, reference_dir: str) -> int:
    """Colour-normalise every source image towards the reference statistics.

    Args:
        source_dir: Directory with 'all' and 'hem' subfolders of source images.
        output_dir: Destination directory (recreated on every call).
        reference_dir: Directory with target training images that define the reference statistics.

    Returns:
        Number of images written.
    """
    ref_mean, ref_std = target_lab_statistics(reference_dir)
    out_root = Path(output_dir)
    if out_root.exists():
        shutil.rmtree(out_root)

    written = 0
    for cls in ("all", "hem"):
        (out_root / cls).mkdir(parents=True)
        for path in sorted((Path(source_dir) / cls).glob("*.png")):
            lab, bgr = _load_lab(path)
            foreground = bgr.max(axis=2) > FOREGROUND_MIN_VALUE
            if foreground.sum() >= MIN_FOREGROUND_PIXELS:
                fg = lab[foreground]
                lab[foreground] = (fg - fg.mean(axis=0)) / (fg.std(axis=0) + 1e-6) * ref_std + ref_mean
            normalized = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            normalized[~foreground] = bgr[~foreground]
            cv2.imwrite(str(out_root / cls / path.name), normalized)
            written += 1
    return written
