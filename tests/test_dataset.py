"""Unit tests for dataset loaders and data integrity."""

import unittest

from torch.utils.data import DataLoader

from src.data.dataset import LeukemiaClassificationDataset, UnpairedLeukemiaDataset, get_default_transform


class TestProcessedDatasets(unittest.TestCase):
    """Test suite verifying processed datasets can be loaded without error."""

    def setUp(self) -> None:
        """Set up dataset paths."""
        self.source_dir = "data/processed/source_cnmc/train"
        self.target_s1_train = "data/processed/target_all_idb/cell_level/train"
        self.target_s1_test = "data/processed/target_all_idb/cell_level/test"

    def test_unpaired_dataset_loading(self) -> None:
        """Verify UnpairedLeukemiaDataset loads valid (source, target) batches."""
        transform = get_default_transform(image_size=128, is_train=True)
        dataset = UnpairedLeukemiaDataset(
            source_dir=self.source_dir,
            target_dir=self.target_s1_train,
            transform=transform,
        )

        if len(dataset) < 659:
            raise AssertionError(f"Expected at least 659 samples, got {len(dataset)}")

        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        img_s, img_t = next(iter(loader))

        if img_s.shape != (4, 3, 128, 128):
            raise AssertionError(f"Expected source shape (4, 3, 128, 128), got {img_s.shape}")
        if img_t.shape != (4, 3, 128, 128):
            raise AssertionError(f"Expected target shape (4, 3, 128, 128), got {img_t.shape}")

    def test_classification_test_dataset(self) -> None:
        """Verify LeukemiaClassificationDataset loads exactly 200 test samples."""
        transform = get_default_transform(image_size=128, is_train=False)
        test_dataset = LeukemiaClassificationDataset(self.target_s1_test, transform=transform)

        if len(test_dataset) != 200:
            raise AssertionError(f"Expected exactly 200 test samples, got {len(test_dataset)}")

        loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        images, labels = next(iter(loader))

        if images.shape != (16, 3, 128, 128):
            raise AssertionError(f"Expected batch shape (16, 3, 128, 128), got {images.shape}")
        if labels.shape != (16,):
            raise AssertionError(f"Expected labels shape (16,), got {labels.shape}")


if __name__ == "__main__":
    unittest.main()
