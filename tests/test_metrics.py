"""Unit tests for classification metrics and the Baydilli (2025) Table 5 column mapping."""

import unittest

import numpy as np

from src.utils.metrics import calculate_classification_metrics, paper_table5_columns


class TestPaperTable5Columns(unittest.TestCase):
    """The proposed-model row of Table 5 (TN=91, TP=87 on 100 ALL + 100 Normal) must be reproduced exactly."""

    def test_proposed_model_row(self) -> None:
        """Verify the paper's printed values are recovered from its confusion matrix."""
        y_true = np.array([1] * 100 + [0] * 100)
        y_pred = np.array([1] * 87 + [0] * 13 + [0] * 91 + [1] * 9)
        paper = paper_table5_columns(calculate_classification_metrics(y_true, y_pred))

        expected = {"Precision": 0.8700, "Recall": 0.9063, "Specificity": 0.8750, "Accuracy": 0.8900, "F-Score": 0.8878}
        for key, value in expected.items():
            if abs(paper[key] - value) > 5e-5:
                raise AssertionError(f"{key}: expected {value}, got {paper[key]:.4f}")


if __name__ == "__main__":
    unittest.main()
