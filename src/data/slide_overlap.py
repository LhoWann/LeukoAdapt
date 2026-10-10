"""Detection of ALL-IDB1 slides that image overlapping fields of view.

Several ALL-IDB1 photographs are adjacent, overlapping fields of the same smear. They must stay on the same side of a
slide-level split, and an annotated cell inside the overlap is the same physical cell in both photographs.
"""

from pathlib import Path

import cv2
import numpy as np

MATCH_SCALE = 8
TEMPLATE_SIZE = 32
GRID = 5
MIN_SCORE = 0.9
MIN_TEMPLATE_STD = 5.0


def find_overlapping_slides(
    im_dir: Path,
    stems: list[str],
) -> list[tuple[str, str, int, int]]:
    """Find slide pairs that share image content.

    A grid of GRID x GRID templates is cut from every slide at 1/MATCH_SCALE resolution and matched against every other
    slide of the same size and the same ALL-IDB1 class suffix (_0 healthy / _1 ALL; overlapping fields come from one
    smear). Normalised cross-correlation tolerates the exposure differences between re-photographed fields.

    Args:
        im_dir: ALL-IDB1 image directory.
        stems: Slide names (without extension) to compare.

    Returns:
        Sorted (slide_a, slide_b, dx, dy) tuples with slide_a < slide_b, where pixel (x, y) of slide_a shows the same
        content as pixel (x + dx, y + dy) of slide_b at full resolution.
    """
    small = {}
    for stem in stems:
        bgr = cv2.imread(str(im_dir / f"{stem}.jpg"))
        small[stem] = cv2.resize(
            bgr, (bgr.shape[1] // MATCH_SCALE, bgr.shape[0] // MATCH_SCALE), interpolation=cv2.INTER_AREA
        )

    pairs: dict[tuple[str, str], tuple[int, int]] = {}
    for stem_a in stems:
        img_a = small[stem_a]
        h, w = img_a.shape[:2]
        for y in np.linspace(0, h - TEMPLATE_SIZE, GRID).astype(int):
            for x in np.linspace(0, w - TEMPLATE_SIZE, GRID).astype(int):
                template = img_a[y : y + TEMPLATE_SIZE, x : x + TEMPLATE_SIZE]
                if template.std() < MIN_TEMPLATE_STD:
                    continue
                for stem_b in stems:
                    key = tuple(sorted((stem_a, stem_b)))
                    same_kind = small[stem_b].shape == img_a.shape and stem_b[-2:] == stem_a[-2:]
                    if stem_b == stem_a or key in pairs or not same_kind:
                        continue
                    _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(small[stem_b], template, cv2.TM_CCOEFF_NORMED))
                    if score > MIN_SCORE:
                        dx, dy = int(loc[0] - x) * MATCH_SCALE, int(loc[1] - y) * MATCH_SCALE
                        pairs[key] = (dx, dy) if key[0] == stem_a else (-dx, -dy)
    return sorted((a, b, dx, dy) for (a, b), (dx, dy) in pairs.items())


def overlap_groups(stems: list[str], pairs: list[tuple[str, str, int, int]]) -> dict[str, str]:
    """Merge overlapping slides into connected groups.

    Args:
        stems: All slide names.
        pairs: Output of find_overlapping_slides.

    Returns:
        Mapping from slide name to its group name (the alphabetically first slide of the group).
    """
    parent = {stem: stem for stem in stems}

    def root(stem: str) -> str:
        while parent[stem] != stem:
            stem = parent[stem]
        return stem

    for a, b, _, _ in pairs:
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    return {stem: root(stem) for stem in stems}
