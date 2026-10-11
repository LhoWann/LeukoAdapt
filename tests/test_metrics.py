"""Unit tests for classification metrics, the Baydilli (2025) Table 5 column mapping, and McNemar's test."""

import unittest

import numpy as np

from src.utils.metrics import calculate_classification_metrics, mcnemar_test, paper_table5_columns


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


class TestMcNemar(unittest.TestCase):
    """McNemar's test with continuity correction, Eq. (9) of Baydilli (2025)."""

    @staticmethod
    def _paired_predictions(b: int, c: int, both_right: int, both_wrong: int) -> tuple[np.ndarray, ...]:
        """Build labels and two prediction vectors with the requested discordant and concordant counts."""
        n = b + c + both_right + both_wrong
        y_true = np.ones(n, dtype=int)
        pred_a = np.concatenate([np.ones(b), np.zeros(c), np.ones(both_right), np.zeros(both_wrong)]).astype(int)
        pred_b = np.concatenate([np.zeros(b), np.ones(c), np.ones(both_right), np.zeros(both_wrong)]).astype(int)
        return y_true, pred_a, pred_b

    def test_hand_computed(self) -> None:
        """b=10, c=2 gives chi2 = 49/12, chi2 p ~ 0.0433 and exact p = 158/4096."""
        result = mcnemar_test(*self._paired_predictions(b=10, c=2, both_right=80, both_wrong=8))

        expected = {"b": 10.0, "c": 2.0, "chi2": 49 / 12, "p_value_exact": 158 / 4096, "significant": 1.0}
        for key, value in expected.items():
            if abs(result[key] - value) > 1e-9:
                raise AssertionError(f"{key}: expected {value}, got {result[key]}")
        if abs(result["p_value_chi2"] - 0.04331) > 1e-4:
            raise AssertionError(f"p_value_chi2: expected ~0.0433, got {result['p_value_chi2']:.5f}")

    def test_no_discordant_pairs(self) -> None:
        """Identical predictions give chi2 = 0, p = 1 and no significance."""
        y_true = np.array([1, 0, 1, 0, 1])
        pred = np.array([1, 0, 0, 0, 1])
        result = mcnemar_test(y_true, pred, pred.copy())

        expected = {"b": 0.0, "c": 0.0, "chi2": 0.0, "p_value_chi2": 1.0, "p_value_exact": 1.0, "significant": 0.0}
        if result != expected:
            raise AssertionError(f"expected {expected}, got {result}")

    def test_symmetry(self) -> None:
        """Swapping A and B swaps b and c but keeps chi2 and both p-values."""
        y_true, pred_a, pred_b = self._paired_predictions(b=7, c=3, both_right=50, both_wrong=5)
        forward = mcnemar_test(y_true, pred_a, pred_b)
        backward = mcnemar_test(y_true, pred_b, pred_a)

        if (forward["b"], forward["c"]) != (backward["c"], backward["b"]):
            raise AssertionError(f"b/c not swapped: {forward} vs {backward}")
        for key in ("chi2", "p_value_chi2", "p_value_exact", "significant"):
            if abs(forward[key] - backward[key]) > 1e-12:
                raise AssertionError(f"{key} not symmetric: {forward[key]} vs {backward[key]}")


if __name__ == "__main__":
    unittest.main()
