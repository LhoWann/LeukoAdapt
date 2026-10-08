"""Target Dataset (ALL-IDB1) Extraction and Preprocessing Pipeline.

Extracts exactly 859 cells purely from ALL-IDB1 (ignoring ALL-IDB2 to prevent duplicates):
- 510 ALL blast cell patches (257x257) from .xyc ground-truth coordinates.
- 349 Normal leukocyte patches (257x257) from healthy images and non-blast leukocytes.

Generates two experimental partitions:
- Skenario 1 (Cell-Level Split / Baydilli 2025): 659 Train (410 ALL, 249 HEM) and 200 Test (100 ALL, 100 HEM).
- Skenario 2 (Patient-Independent Split): Complete unseen microscopic slides in Test set.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def crop_patch_centered(image: Image.Image, cx: int, cy: int, patch_size: int = 257) -> Image.Image:
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

    roi = image_np[y1:y2, x1:x2]
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


def detect_leukocytes_in_image(
    image: Image.Image,
    exclude_coords: list[tuple[int, int]] | None = None,
    min_dist: int = 150,
) -> list[tuple[int, int]]:
    """Detect non-overlapping leukocyte nuclei using Giemsa staining absorption.

    Args:
        image: Source PIL image.
        exclude_coords: List of coordinates to avoid (e.g. known blast centroids).
        min_dist: Minimum distance between detected centroids.

    Returns:
        List of detected (cx, cy) centroids.
    """
    w, h = image.size
    margin = 130
    img_np = np.array(image.convert("RGB"), dtype=np.int32)
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # Nucleus contrast in Giemsa stain
    stain = (b - g) + (r - g)
    dark = 255 - g
    mask = (stain > 40) & (dark > 90)

    scale = 4
    mask_s = mask[::scale, ::scale]
    hs, ws = mask_s.shape
    visited = np.zeros_like(mask_s, dtype=bool)
    candidates = []

    for y in range(hs):
        for x in range(ws):
            if mask_s[y, x] and not visited[y, x]:
                queue = [(y, x)]
                visited[y, x] = True
                pts = []
                while queue:
                    cy, cx = queue.pop()
                    pts.append((cy, cx))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < hs and 0 <= nx < ws and mask_s[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

                area = len(pts) * 16
                if 1000 <= area <= 40000:
                    arr = np.array(pts)
                    my = int(np.mean(arr[:, 0]) * scale)
                    mx = int(np.mean(arr[:, 1]) * scale)
                    if margin <= mx <= (w - margin) and margin <= my <= (h - margin):
                        candidates.append((mx, my, area))

    # Non-maximum suppression and exclusion check
    selected = []
    candidates.sort(key=lambda item: item[2], reverse=True)
    for mx, my, _ in candidates:
        if exclude_coords:
            is_near_excluded = any(np.hypot(mx - ex, my - ey) < min_dist for ex, ey in exclude_coords)
            if is_near_excluded:
                continue

        if all(np.hypot(mx - sx, my - sy) >= min_dist for sx, sy in selected):
            selected.append((mx, my))

    return selected


def extract_all_idb(
    raw_dir: str = "data/raw/ALL-IDB/ALL_IDB1",
    output_dir: str = "data/processed/target_all_idb",
    image_size: int = 128,
) -> dict[str, Any]:
    """Extract 859 cells from ALL-IDB1 and organize into train/test sets.

    Args:
        raw_dir: Directory containing ALL_IDB1 images and xyc folders.
        output_dir: Output processed root directory.
        image_size: Normalized resolution for neural network inputs (default 128).

    Returns:
        Summary dictionary with extraction statistics.
    """
    idb1_path = Path(raw_dir)
    im_path = idb1_path / "im"
    xyc_path = idb1_path / "xyc"

    out_base = Path(output_dir)
    cell_level_dir = out_base / "cell_level"
    patient_level_dir = out_base / "patient_level"

    for d in [
        cell_level_dir / "train" / "all",
        cell_level_dir / "train" / "hem",
        cell_level_dir / "test" / "all",
        cell_level_dir / "test" / "hem",
        patient_level_dir / "train" / "all",
        patient_level_dir / "train" / "hem",
        patient_level_dir / "test" / "all",
        patient_level_dir / "test" / "hem",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Extract 510 ALL Blast Cells from .xyc coordinates
    print("--- [1/3] Extracting 510 ALL Blast Cells from ALL-IDB1 .xyc ---")
    blast_metadata = []
    blast_patches = []
    xyc_files = sorted(xyc_path.glob("*.xyc"))

    for xf in xyc_files:
        stem = xf.stem
        img_file = im_path / f"{stem}.jpg"
        if not img_file.exists():
            continue

        with Image.open(img_file) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)
            with open(xf) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        cx, cy = int(parts[0]), int(parts[1])
                        rcx, rcy = refine_centroid(img_np, cx, cy)
                        patch = crop_patch_centered(img_rgb, rcx, rcy, patch_size=257)
                        blast_patches.append(patch)
                        blast_metadata.append(
                            {
                                "source_image": stem,
                                "centroid": [rcx, rcy],
                                "label": 1,
                                "class_name": "ALL",
                            }
                        )

    print(f"Total blast cells extracted: {len(blast_patches)} (Target: 510)")

    # 2. Extract 349 Normal Leukocytes
    print("--- [2/3] Extracting 349 Normal Leukocytes from ALL-IDB1 ---")
    normal_metadata = []
    normal_patches = []

    # 2a. From healthy slides (_0.jpg)
    healthy_files = sorted(im_path.glob("*_0.jpg"))
    for hf in healthy_files:
        if len(normal_patches) >= 349:
            break
        with Image.open(hf) as img:
            img_rgb = img.convert("RGB")
            coords = detect_leukocytes_in_image(img_rgb)
            for cx, cy in coords:
                if len(normal_patches) >= 349:
                    break
                patch = crop_patch_centered(img_rgb, cx, cy, patch_size=257)
                normal_patches.append(patch)
                normal_metadata.append(
                    {
                        "source_image": hf.stem,
                        "centroid": [cx, cy],
                        "label": 0,
                        "class_name": "Normal",
                    }
                )

    # 2b. From leukemic slides (_1.jpg, mature non-blast leukocytes)
    if len(normal_patches) < 349:
        all_files = sorted(im_path.glob("*_1.jpg"))
        for af in all_files:
            if len(normal_patches) >= 349:
                break
            # Load blast coords for exclusion
            xf = xyc_path / f"{af.stem}.xyc"
            b_coords = []
            if xf.exists():
                with open(xf) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) >= 2:
                            b_coords.append((int(p[0]), int(p[1])))

            with Image.open(af) as img:
                img_rgb = img.convert("RGB")
                coords = detect_leukocytes_in_image(img_rgb, exclude_coords=b_coords)
                for cx, cy in coords:
                    if len(normal_patches) >= 349:
                        break
                    patch = crop_patch_centered(img_rgb, cx, cy, patch_size=257)
                    normal_patches.append(patch)
                    normal_metadata.append(
                        {
                            "source_image": af.stem,
                            "centroid": [cx, cy],
                            "label": 0,
                            "class_name": "Normal",
                        }
                    )

    print(f"Total normal cells extracted: {len(normal_patches)} (Target: 349)")

    # 3. Save Skenario 1: Cell-Level Replication (Baydilli 2025)
    # Train: 410 ALL + 249 Normal | Test: 100 ALL + 100 Normal
    print("--- [3/3] Saving Processed Images and Split Metadata ---")
    summary = {
        "scenario_1_cell_level": {
            "train_all": 0,
            "train_hem": 0,
            "test_all": 0,
            "test_hem": 0,
        },
        "scenario_2_patient_level": {
            "train_all": 0,
            "train_hem": 0,
            "test_all": 0,
            "test_hem": 0,
        },
    }

    # Save Skenario 1 ALL
    for idx, patch in enumerate(blast_patches[:510]):
        resized = patch.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if idx < 410:
            resized.save(cell_level_dir / "train" / "all" / f"all_train_{idx:03d}.png")
            blast_metadata[idx]["scenario_1_split"] = "train"
            summary["scenario_1_cell_level"]["train_all"] += 1
        else:
            t_idx = idx - 410
            resized.save(cell_level_dir / "test" / "all" / f"all_test_{t_idx:03d}.png")
            blast_metadata[idx]["scenario_1_split"] = "test"
            summary["scenario_1_cell_level"]["test_all"] += 1

    # Save Skenario 1 Normal
    for idx, patch in enumerate(normal_patches[:349]):
        resized = patch.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if idx < 249:
            resized.save(cell_level_dir / "train" / "hem" / f"hem_train_{idx:03d}.png")
            normal_metadata[idx]["scenario_1_split"] = "train"
            summary["scenario_1_cell_level"]["train_hem"] += 1
        else:
            t_idx = idx - 249
            resized.save(cell_level_dir / "test" / "hem" / f"hem_test_{t_idx:03d}.png")
            normal_metadata[idx]["scenario_1_split"] = "test"
            summary["scenario_1_cell_level"]["test_hem"] += 1

    # Skenario 2: Patient-Independent Partitioning
    # Reserve specific whole-slide images solely for testing (approx 20% unseen patients)
    all_blast_slides = sorted({item["source_image"] for item in blast_metadata})
    all_normal_slides = sorted({item["source_image"] for item in normal_metadata})

    test_blast_slides = set(all_blast_slides[-10:])
    test_normal_slides = set(all_normal_slides[-12:])

    for idx, item in enumerate(blast_metadata):
        patch = blast_patches[idx]
        resized = patch.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if item["source_image"] in test_blast_slides:
            fname = f"all_patient_test_{idx:03d}.png"
            resized.save(patient_level_dir / "test" / "all" / fname)
            item["scenario_2_split"] = "test"
            summary["scenario_2_patient_level"]["test_all"] += 1
        else:
            fname = f"all_patient_train_{idx:03d}.png"
            resized.save(patient_level_dir / "train" / "all" / fname)
            item["scenario_2_split"] = "train"
            summary["scenario_2_patient_level"]["train_all"] += 1

    for idx, item in enumerate(normal_metadata):
        patch = normal_patches[idx]
        resized = patch.resize((image_size, image_size), Image.Resampling.BICUBIC)
        if item["source_image"] in test_normal_slides:
            fname = f"hem_patient_test_{idx:03d}.png"
            resized.save(patient_level_dir / "test" / "hem" / fname)
            item["scenario_2_split"] = "test"
            summary["scenario_2_patient_level"]["test_hem"] += 1
        else:
            fname = f"hem_patient_train_{idx:03d}.png"
            resized.save(patient_level_dir / "train" / "hem" / fname)
            item["scenario_2_split"] = "train"
            summary["scenario_2_patient_level"]["train_hem"] += 1

    # Save Provenance Metadata
    with open(out_base / "metadata_target_cells.json", "w") as f:
        json.dump({"blasts": blast_metadata, "normals": normal_metadata, "summary": summary}, f, indent=2)

    print("Data extraction finished. Summary:", summary)
    return summary


if __name__ == "__main__":
    extract_all_idb()
