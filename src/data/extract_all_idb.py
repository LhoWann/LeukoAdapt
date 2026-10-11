"""Target Dataset (ALL-IDB) Extraction and Preprocessing Pipeline.

Every patch is cropped from the ALL-IDB1 slides with the same 257x257 procedure, using only expert annotations:
- 510 ALL blast patches from the ALL-IDB1 .xyc blast centroids.
- 125 Normal patches from the ALL-IDB2 '_0' crops ("the central cell is not a blast", healthy individuals). Each
  crop is located in its ALL-IDB1 source slide by template matching; ALL-IDB2 holds 130 such crops, 5 of which repeat
  a cell, leaving 125 unique cells. Baydilli (2025) reports 349 Normal cells without stating their source; ALL-IDB
  contains no further expert-labelled normal cells (see README Section 2.C).

ALL-IDB1 Im108_0 is a pixel-identical copy of Im093_0 and is never used.

Generates two experimental partitions:
- Scenario 1 (cell-level split, Baydilli 2025 baseline): seeded stratified random split with 100 ALL + 100 Normal test
  cells as in the paper; the rest (410 ALL, 25 Normal) is the target training set. Cells of one slide can fall on
  both sides.
- Scenario 2 (slide-level split): slides that image overlapping fields of the same smear are merged into groups, and
  whole groups are assigned to Train or Test. Each physical cell is kept once (copies on overlapping slides are
  excluded). Test groups are drawn with a seed until about 20% of the cells of every (class, slide resolution) stratum
  are in Test. ALL-IDB1 documents no patient IDs, so group isolation is necessary but not sufficient for patient
  isolation.

Each scenario also gets a bank of cell-free background patches (<scenario>/background), cropped only from slides on
its training side, onto which source cells are pasted before translation (see src/data/backgrounds.py).
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.data.backgrounds import extract_background_bank
from src.data.slide_overlap import find_overlapping_slides, overlap_groups

PATCH_SIZE = 257
TEST_CELLS_PER_CLASS = 100
DUPLICATE_SLIDES = {"Im108_0": "Im093_0"}
MATCH_SCALE = 4
MIN_MATCH_SCORE = 0.9
SAME_CELL_DISTANCE = 60
CROSS_SLIDE_CELL_DISTANCE = 40
S2_TEST_FRACTION = 0.2
OVERLAP_CACHE = "slide_overlaps.json"


def crop_patch_centered(image: Image.Image, cx: int, cy: int, patch_size: int = PATCH_SIZE) -> Image.Image:
    """Crop square patch of given size centered at (cx, cy).

    Args:
        image: Source PIL Image.
        cx: Center X coordinate.
        cy: Center Y coordinate.
        patch_size: Square patch dimensions (default 257).

    Returns:
        Cropped PIL Image of exact size (patch_size, patch_size).
    """
    w, h = image.size
    half = patch_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift_x = x2 - w
        x1 = max(0, x1 - shift_x)
        x2 = w
    if y2 > h:
        shift_y = y2 - h
        y1 = max(0, y1 - shift_y)
        y2 = h

    return image.crop((x1, y1, x2, y2))


def refine_centroid(image_np: np.ndarray, cx: int, cy: int, search_radius: int = 20) -> tuple[int, int]:
    """Refine centroid to local leukocyte nucleus center of mass.

    Args:
        image_np: RGB image array of shape (H, W, 3).
        cx: Initial centroid X.
        cy: Initial centroid Y.
        search_radius: Local search window radius (default 20).

    Returns:
        Refined (cx, cy) tuple.
    """
    h, w, _ = image_np.shape
    x1 = max(0, cx - search_radius)
    y1 = max(0, cy - search_radius)
    x2 = min(w, cx + search_radius)
    y2 = min(h, cy + search_radius)

    roi = image_np[y1:y2, x1:x2].astype(np.int32)
    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    stain = (b - g) + (r - g)
    dark = 255 - g
    mask = (stain > 40) & (dark > 90)

    if np.sum(mask) > 50:
        y_indices, x_indices = np.where(mask)
        mean_x = int(np.mean(x_indices)) + x1
        mean_y = int(np.mean(y_indices)) + y1
        return mean_x, mean_y
    return cx, cy


def _crop_axis_center(start: int, length: int, image_length: int) -> int:
    """Center of a cell along one axis of an ALL-IDB2 crop that may be clipped by the ALL-IDB1 image border."""
    if length >= PATCH_SIZE:
        return start + length // 2
    if start <= 2 * MATCH_SCALE:
        return length - PATCH_SIZE // 2 - 1
    if start + length >= image_length - 2 * MATCH_SCALE:
        return start + PATCH_SIZE // 2
    return start + length // 2


def locate_idb2_normals(idb1_im_path: Path, idb2_img_path: Path) -> list[dict[str, Any]]:
    """Locate the ALL-IDB2 Normal crops in their ALL-IDB1 healthy source slides.

    Args:
        idb1_im_path: ALL-IDB1 image directory.
        idb2_img_path: ALL-IDB2 image directory.

    Returns:
        One record per unique cell with the ALL-IDB1 slide, the cell center, and the ALL-IDB2 file name(s).

    Raises:
        ValueError: If an ALL-IDB2 crop has no confident match in ALL-IDB1.
    """
    slides = {}
    for path in sorted(idb1_im_path.glob("*_0.jpg")):
        if path.stem in DUPLICATE_SLIDES:
            continue
        bgr = cv2.imread(str(path))
        small = cv2.resize(
            bgr, (bgr.shape[1] // MATCH_SCALE, bgr.shape[0] // MATCH_SCALE), interpolation=cv2.INTER_AREA
        )
        slides[path.stem] = (small, bgr.shape[1], bgr.shape[0])

    cells: list[dict[str, Any]] = []
    for crop_path in sorted(idb2_img_path.glob("*_0.tif")):
        crop = cv2.imread(str(crop_path))
        crop_h, crop_w = crop.shape[:2]
        template = cv2.resize(crop, (crop_w // MATCH_SCALE, crop_h // MATCH_SCALE), interpolation=cv2.INTER_AREA)

        best_score, best_slide, best_loc = -1.0, "", (0, 0)
        for stem, (small, _, _) in slides.items():
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(small, template, cv2.TM_CCOEFF_NORMED))
            if score > best_score:
                best_score, best_slide, best_loc = score, stem, loc
        if best_score < MIN_MATCH_SCORE:
            raise ValueError(f"{crop_path.name} has no ALL-IDB1 match (best score {best_score:.3f} in {best_slide})")

        _, slide_w, slide_h = slides[best_slide]
        x0, y0 = best_loc[0] * MATCH_SCALE, best_loc[1] * MATCH_SCALE
        cx = _crop_axis_center(x0, crop_w, slide_w)
        cy = _crop_axis_center(y0, crop_h, slide_h)

        duplicate = next(
            (
                c
                for c in cells
                if c["source_image"] == best_slide and np.hypot(c["cx"] - cx, c["cy"] - cy) < SAME_CELL_DISTANCE
            ),
            None,
        )
        if duplicate is not None:
            duplicate["idb2_files"].append(crop_path.stem)
            continue
        cells.append(
            {"source_image": best_slide, "cx": cx, "cy": cy, "idb2_files": [crop_path.stem], "score": best_score}
        )
    return cells


def load_slide_overlaps(im_path: Path, stems: list[str], cache_path: Path) -> list[tuple[str, str, int, int]]:
    """Return the overlapping slide pairs, reusing the cache when it was computed for the same slides.

    Args:
        im_path: ALL-IDB1 image directory.
        stems: Slides to compare.
        cache_path: JSON cache of a previous search.

    Returns:
        Output of find_overlapping_slides.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached["stems"] == stems:
            return [tuple(pair) for pair in cached["pairs"]]
    print("Searching for overlapping fields of view (several minutes; cached afterwards)...")
    pairs = find_overlapping_slides(im_path, stems)
    cache_path.write_text(json.dumps({"stems": stems, "pairs": pairs}, indent=1))
    return pairs


def assign_physical_cell_ids(cells: list[dict[str, Any]], pairs: list[tuple[str, str, int, int]]) -> None:
    """Set "physical_cell_id" on every cell: copies of one cell on overlapping slides share the lowest index.

    Args:
        cells: Cell metadata (source_image, centroid, label), indexed by position.
        pairs: Overlapping slide pairs with their full-resolution offsets.
    """
    by_slide = defaultdict(list)
    for idx, cell in enumerate(cells):
        by_slide[cell["source_image"]].append(idx)
    parent = list(range(len(cells)))

    def root(idx: int) -> int:
        while parent[idx] != idx:
            idx = parent[idx]
        return idx

    for slide_a, slide_b, dx, dy in pairs:
        for i in by_slide[slide_a]:
            for j in by_slide[slide_b]:
                (xa, ya), (xb, yb) = cells[i]["centroid"], cells[j]["centroid"]
                same_place = np.hypot(xa + dx - xb, ya + dy - yb) < CROSS_SLIDE_CELL_DISTANCE
                if same_place and cells[i]["label"] == cells[j]["label"]:
                    ri, rj = root(i), root(j)
                    parent[max(ri, rj)] = min(ri, rj)
    for idx, cell in enumerate(cells):
        cell["physical_cell_id"] = root(idx)


def extract_all_idb(
    raw_dir: str = "data/raw/ALL-IDB/ALL_IDB1",
    idb2_dir: str = "data/raw/ALL-IDB/ALL_IDB2/img",
    output_dir: str = "data/processed/target_all_idb",
    image_size: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Extract the expert-annotated ALL-IDB cells and organize them into the two split scenarios.

    Args:
        raw_dir: Directory containing ALL_IDB1 images and xyc folders.
        idb2_dir: Directory containing the ALL_IDB2 crops (only the Normal '_0' crops are used, as annotations).
        output_dir: Output processed root directory.
        image_size: Normalized resolution for neural network inputs (default 128).
        seed: Random seed of the Scenario 1 stratified split.

    Returns:
        Summary dictionary with extraction statistics.

    Raises:
        ValueError: If a slide group appears in both the train and test side of Scenario 2.
    """
    idb1_path = Path(raw_dir)
    im_path = idb1_path / "im"
    xyc_path = idb1_path / "xyc"

    out_base = Path(output_dir)
    cell_level_dir = out_base / "cell_level"
    slide_level_dir = out_base / "slide_level"

    for scenario_dir in (cell_level_dir, slide_level_dir):
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)
        for split in ("train", "test"):
            for cls in ("all", "hem"):
                (scenario_dir / split / cls).mkdir(parents=True)

    print("--- [1/3] Extracting ALL Blast Cells from ALL-IDB1 .xyc ---")
    blast_metadata = []
    blast_patches = []
    for xf in sorted(xyc_path.glob("*.xyc")):
        img_file = im_path / f"{xf.stem}.jpg"
        if not img_file.exists():
            continue

        with Image.open(img_file) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
            with open(xf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        rcx, rcy = refine_centroid(img_np, int(parts[0]), int(parts[1]))
                        blast_patches.append(crop_patch_centered(img_rgb, rcx, rcy))
                        blast_metadata.append(
                            {"source_image": xf.stem, "centroid": [rcx, rcy], "label": 1, "class_name": "ALL"}
                        )
    print(f"Total blast cells extracted: {len(blast_patches)}")

    print("--- [2/3] Locating ALL-IDB2 Normal Cells in ALL-IDB1 Healthy Slides ---")
    normal_metadata = []
    normal_patches = []
    for cell in locate_idb2_normals(im_path, Path(idb2_dir)):
        with Image.open(im_path / f"{cell['source_image']}.jpg") as img:
            img_rgb = img.convert("RGB")
            rcx, rcy = refine_centroid(np.array(img_rgb), cell["cx"], cell["cy"])
            normal_patches.append(crop_patch_centered(img_rgb, rcx, rcy))
        normal_metadata.append(
            {
                "source_image": cell["source_image"],
                "centroid": [rcx, rcy],
                "label": 0,
                "class_name": "Normal",
                "idb2_files": cell["idb2_files"],
                "match_score": round(cell["score"], 4),
            }
        )
    print(f"Total unique normal cells extracted: {len(normal_patches)}")

    print("--- [3/3] Saving Processed Images and Split Metadata ---")
    classes = {"all": (blast_patches, blast_metadata), "hem": (normal_patches, normal_metadata)}

    # Scenario 1: seeded stratified random split (cells of one slide may land on both sides)
    rng = np.random.default_rng(seed)
    for patches, metadata in classes.values():
        test_ids = set(rng.permutation(len(patches))[:TEST_CELLS_PER_CLASS].tolist())
        for idx, item in enumerate(metadata):
            item["scenario_1_split"] = "test" if idx in test_ids else "train"

    # Scenario 2: whole overlap groups, unique physical cells, test groups drawn per (class, slide width) stratum
    cells = blast_metadata + normal_metadata
    slides = sorted({cell["source_image"] for cell in cells})
    pairs = load_slide_overlaps(im_path, slides, out_base / OVERLAP_CACHE)
    slide_group = overlap_groups(slides, pairs)
    assign_physical_cell_ids(cells, pairs)
    slide_width = {stem: Image.open(im_path / f"{stem}.jpg").size[0] for stem in slides}

    group_counts: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idx, cell in enumerate(cells):
        if cell["physical_cell_id"] == idx:
            stratum = (cell["label"], slide_width[cell["source_image"]])
            group_counts[stratum][slide_group[cell["source_image"]]] += 1

    split_rng = np.random.default_rng(seed)
    test_groups: set[str] = set()
    for stratum in sorted(group_counts):
        counts = group_counts[stratum]
        taken = 0
        for group in split_rng.permutation(sorted(counts)):
            if taken >= S2_TEST_FRACTION * sum(counts.values()):
                break
            test_groups.add(str(group))
            taken += counts[group]

    for idx, cell in enumerate(cells):
        cell["slide_group"] = slide_group[cell["source_image"]]
        if cell["physical_cell_id"] != idx:
            cell["scenario_2_split"] = "duplicate"
        else:
            cell["scenario_2_split"] = "test" if cell["slide_group"] in test_groups else "train"

    def slides_of(split_key: str, split: str) -> set[str]:
        return {item["source_image"] for _, meta in classes.values() for item in meta if item[split_key] == split}

    train_groups = {cell["slide_group"] for cell in cells if cell["scenario_2_split"] == "train"}
    shared_groups = train_groups & {cell["slide_group"] for cell in cells if cell["scenario_2_split"] == "test"}
    if shared_groups:
        raise ValueError(f"Scenario 2 slide groups present in both train and test: {sorted(shared_groups)}")

    summary: dict[str, dict[str, int]] = {}
    scenarios = (
        ("scenario_1_cell_level", cell_level_dir, "scenario_1_split", "cell"),
        ("scenario_2_slide_level", slide_level_dir, "scenario_2_split", "slide"),
    )
    centroids: dict[str, list[list[int]]] = defaultdict(list)
    for cell in cells:
        centroids[cell["source_image"]].append(cell["centroid"])
    for scenario_key, scenario_dir, split_key, tag in scenarios:
        counts = {f"{split}_{cls}": 0 for split in ("train", "test") for cls in classes}
        for cls, (patches, metadata) in classes.items():
            for idx, (patch, item) in enumerate(zip(patches, metadata, strict=True)):
                split = item[split_key]
                if split == "duplicate":
                    counts["excluded_duplicates"] = counts.get("excluded_duplicates", 0) + 1
                    continue
                resized = patch.resize((image_size, image_size), Image.Resampling.BICUBIC)
                resized.save(scenario_dir / split / cls / f"{cls}_{tag}_{split}_{idx:03d}.png")
                counts[f"{split}_{cls}"] += 1
        counts["train_slides"] = len(slides_of(split_key, "train"))
        counts["test_slides"] = len(slides_of(split_key, "test"))
        counts["slides_in_both_splits"] = len(slides_of(split_key, "train") & slides_of(split_key, "test"))
        background_slides = sorted(slides_of(split_key, "train"))
        background_dir = scenario_dir / "background"
        counts["background_patches"] = extract_background_bank(
            im_path, background_slides, centroids, background_dir, image_size, seed
        )
        summary[scenario_key] = counts

    with open(out_base / "metadata_target_cells.json", "w") as f:
        metadata_out = {"blasts": blast_metadata, "normals": normal_metadata, "summary": summary}
        json.dump({**metadata_out, "overlapping_slide_pairs": pairs}, f, indent=2)

    print("Data extraction finished. Summary:", summary)
    return summary


if __name__ == "__main__":
    extract_all_idb()
