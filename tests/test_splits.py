"""Unit tests for the Scenario 2 slide grouping, duplicate-cell handling and the generated split metadata."""

import json
import unittest
from pathlib import Path

from src.data.extract_all_idb import assign_physical_cell_ids
from src.data.slide_overlap import overlap_groups

METADATA = Path("data/processed/target_all_idb/metadata_target_cells.json")


class TestSlideGrouping(unittest.TestCase):
    """Overlapping slides must form connected groups and shared cells must be recognised as one physical cell."""

    def test_overlap_groups_are_transitive(self) -> None:
        """A-B and B-C overlaps put A, B and C in one group; D stays alone."""
        groups = overlap_groups(["A", "B", "C", "D"], [("A", "B", 0, 0), ("B", "C", 0, 0)])
        if groups != {"A": "A", "B": "A", "C": "A", "D": "D"}:
            raise AssertionError(f"Unexpected groups {groups}")

    def test_physical_cell_ids(self) -> None:
        """A cell shifted by the slide offset is a copy; a different class at the same place is not."""
        cells = [
            {"source_image": "A", "centroid": [100, 100], "label": 1},
            {"source_image": "B", "centroid": [150, 80], "label": 1},
            {"source_image": "B", "centroid": [150, 80], "label": 0},
            {"source_image": "B", "centroid": [600, 600], "label": 1},
        ]
        assign_physical_cell_ids(cells, [("A", "B", 50, -20)])
        ids = [cell["physical_cell_id"] for cell in cells]
        if ids != [0, 0, 2, 3]:
            raise AssertionError(f"Unexpected physical cell ids {ids}")


class TestScenario2Metadata(unittest.TestCase):
    """The generated Scenario 2 split must not share slide groups or physical cells between train and test."""

    def test_no_group_or_cell_crosses_the_split(self) -> None:
        """Every slide group and every physical cell lies on one side only."""
        meta = json.loads(METADATA.read_text())
        cells = meta["blasts"] + meta["normals"]
        sides: dict[str, set[str]] = {}
        for cell in cells:
            sides.setdefault(f"group:{cell['slide_group']}", set()).add(cell["scenario_2_split"])
            sides.setdefault(f"cell:{cell['physical_cell_id']}", set()).add(cell["scenario_2_split"])
        crossing = [key for key, splits in sides.items() if {"train", "test"} <= splits]
        if crossing:
            raise AssertionError(f"Train/test leakage through {crossing}")


if __name__ == "__main__":
    unittest.main()
