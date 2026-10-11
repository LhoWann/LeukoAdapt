"""Unit tests for the target background bank and source-on-background compositing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.backgrounds import background_for, composite_on_background, load_background_paths


class TestCompositing(unittest.TestCase):
    """A pasted cell must keep its pixels while the black source background is replaced."""

    def setUp(self) -> None:
        """Create a segmented source cell (disc on black) and a uniform background."""
        yy, xx = np.mgrid[:128, :128]
        disc = (yy - 64) ** 2 + (xx - 64) ** 2 < 30**2
        source = np.zeros((128, 128, 3), np.uint8)
        source[disc] = (120, 60, 160)
        self.source, self.disc = Image.fromarray(source), disc
        self.background = Image.fromarray(np.full((128, 128, 3), (210, 170, 170), np.uint8))

    def test_cell_kept_and_background_replaced(self) -> None:
        """Verify cell interior and far background pixels."""
        out = np.asarray(composite_on_background(self.source, self.background)).astype(int)
        if np.abs(out[64, 64] - (120, 60, 160)).max() > 2:
            raise AssertionError(f"cell centre changed: {out[64, 64]}")
        if np.abs(out[5, 5] - (210, 170, 170)).max() > 2:
            raise AssertionError(f"background not replaced: {out[5, 5]}")

    def test_background_choice_is_deterministic(self) -> None:
        """The same file name always gets the same background."""
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                self.background.save(Path(tmp) / f"bg{i}.png")
            bank = load_background_paths(tmp)
            if len({background_for("cnmc_all_0001.png", bank) for _ in range(10)}) != 1:
                raise AssertionError("background choice must not change between calls")

    def test_empty_bank_raises(self) -> None:
        """An empty bank directory is an error, not a silent no-op."""
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_background_paths(tmp)


if __name__ == "__main__":
    unittest.main()
