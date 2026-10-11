"""Target background bank and source-on-background compositing.

Modification of Baydilli (2025): C-NMC cells are segmented on black, ALL-IDB cells lie among red blood cells. A
discriminator that sees whole images separates the two domains by the background alone, and a generator cannot paint
red blood cells into a flat black area. Each source cell is therefore pasted onto a real, cell-free ALL-IDB background
patch before translation, so the GAN only has to adapt the cell itself.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PATCH_SIZE = 257
PATCHES_PER_SLIDE = 12
MAX_TRIES_PER_SLIDE = 200
MIN_CELL_DISTANCE = 220
MAX_STAIN_PIXELS = 150
MAX_ARROW_PIXELS = 20
SOURCE_FOREGROUND_THRESHOLD = 8
EDGE_EROSION_PX = 2
MASK_FEATHER_SIGMA = 1.0
FUSION_RIM_PX = 3


def extract_background_bank(
    im_path: Path,
    slides: list[str],
    cell_centroids: dict[str, list[list[int]]],
    out_dir: Path,
    image_size: int = 128,
    seed: int = 42,
) -> int:
    """Crop cell-free background patches from target slides.

    A patch is kept when its centre is at least MIN_CELL_DISTANCE px from every annotated cell and it contains no
    nucleus-like stain and no orange annotation arrow.

    Args:
        im_path: ALL-IDB1 image directory.
        slides: Slides allowed to provide backgrounds (training side of the split only).
        cell_centroids: Annotated cell centroids per slide, of every class and split.
        out_dir: Destination directory (created if missing).
        image_size: Output resolution.
        seed: Random seed of the crop positions.

    Returns:
        Number of patches written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    for slide in slides:
        image = Image.open(im_path / f"{slide}.jpg").convert("RGB")
        pixels = np.asarray(image).astype(np.int32)
        h, w = pixels.shape[:2]
        kept = 0
        for _ in range(MAX_TRIES_PER_SLIDE):
            if kept == PATCHES_PER_SLIDE:
                break
            x, y = int(rng.integers(0, w - PATCH_SIZE)), int(rng.integers(0, h - PATCH_SIZE))
            cx, cy = x + PATCH_SIZE // 2, y + PATCH_SIZE // 2
            if any(np.hypot(cx - px, cy - py) < MIN_CELL_DISTANCE for px, py in cell_centroids.get(slide, [])):
                continue
            crop = pixels[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
            stain = (((b - g) + (r - g)) > 40) & (255 - g > 90)
            arrow = (r > 180) & (b < 120) & (r - b > 100)
            if stain.sum() > MAX_STAIN_PIXELS or arrow.sum() > MAX_ARROW_PIXELS:
                continue
            patch = image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            patch.resize((image_size, image_size), Image.Resampling.BICUBIC).save(out_dir / f"{slide}_bg{kept:02d}.png")
            kept += 1
        written += kept
    return written


def load_background_paths(background_dir: str | Path) -> list[Path]:
    """List the background patches of a bank.

    Args:
        background_dir: Directory written by extract_background_bank.

    Returns:
        Sorted patch paths.

    Raises:
        ValueError: If the directory holds no patches.
    """
    paths = sorted(Path(background_dir).glob("*.png"))
    if not paths:
        raise ValueError(f"No background patches in {background_dir}; run python -m src.data.extract_all_idb")
    return paths


def cell_alpha(source: Image.Image) -> np.ndarray:
    """Soft foreground mask of a segmented source cell (black background).

    Args:
        source: Segmented source cell image.

    Returns:
        Float mask in [0, 1] of shape (H, W).
    """
    src = np.asarray(source.convert("RGB"))
    foreground = ndimage.binary_fill_holes(ndimage.binary_opening(src.max(axis=2) > SOURCE_FOREGROUND_THRESHOLD))
    # The resized segmentation edge is blended with black; dropping it avoids a dark ring around the pasted cell.
    foreground = ndimage.binary_erosion(foreground, iterations=EDGE_EROSION_PX)
    return ndimage.gaussian_filter(foreground.astype(np.float32), MASK_FEATHER_SIGMA)


def fusion_mask(alpha: np.ndarray) -> np.ndarray:
    """Region the generator may change: the cell plus a thin rim, so it can blend the pasted boundary.

    Args:
        alpha: Output of cell_alpha.

    Returns:
        Float mask in [0, 1] of shape (H, W).
    """
    region = ndimage.binary_dilation(alpha > 0.5, iterations=FUSION_RIM_PX)
    return ndimage.gaussian_filter(region.astype(np.float32), MASK_FEATHER_SIGMA)


def composite_with_mask(source: Image.Image, background: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Paste a segmented source cell onto a target background patch and return the generator's fusion mask.

    Args:
        source: Segmented source cell image.
        background: Target background patch.

    Returns:
        Tuple of (composite image, fusion mask of shape (H, W)).
    """
    alpha = cell_alpha(source)
    src = np.asarray(source.convert("RGB"), dtype=np.float32)
    bg = np.asarray(background.convert("RGB").resize(source.size, Image.Resampling.BICUBIC), dtype=np.float32)
    composite = alpha[..., None] * src + (1.0 - alpha[..., None]) * bg
    return Image.fromarray(composite.clip(0, 255).astype(np.uint8)), fusion_mask(alpha)


def composite_on_background(source: Image.Image, background: Image.Image) -> Image.Image:
    """Paste a segmented source cell (black background) onto a target background patch.

    Args:
        source: Segmented source cell image.
        background: Target background patch of the same size.

    Returns:
        The composite image.
    """
    return composite_with_mask(source, background)[0]


def background_for(name: str, backgrounds: list[Path]) -> Path:
    """Pick a background deterministically from a file name, so translations are reproducible.

    Args:
        name: Source file name.
        backgrounds: Background bank.

    Returns:
        The chosen background path.
    """
    return random.Random(name).choice(backgrounds)


def write_composites(source_dir: str, output_dir: str, background_dir: str) -> int:
    """Write every source cell pasted onto its deterministic background (the "composite" baseline, no GAN).

    Args:
        source_dir: Directory with 'all' and 'hem' subfolders of segmented source cells.
        output_dir: Destination directory with the same layout.
        background_dir: Bank of target backgrounds.

    Returns:
        Number of images written.
    """
    backgrounds = load_background_paths(background_dir)
    written = 0
    for cls in ("all", "hem"):
        out_dir = Path(output_dir) / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted((Path(source_dir) / cls).glob("*.png")):
            with Image.open(path) as img:
                source = img.convert("RGB")
            composite_on_background(source, Image.open(background_for(path.name, backgrounds))).save(out_dir / path.name)
            written += 1
    return written
