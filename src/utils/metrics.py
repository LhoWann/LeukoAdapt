"""Evaluation metrics for classification and image translation quality.

Implements the Table 5 metrics of Baydilli (2025) (TN, TP, Precision, Recall, Specificity, Accuracy, F-Score) and the
PSNR of Table 4, plus NPV, Balanced Accuracy, AUROC, a Wilson confidence interval for accuracy, and McNemar's test
(Eq. 9) for paired model comparison.
"""

import numpy as np
import torch
from scipy.stats import binomtest, chi2 as chi2_dist
from sklearn.metrics import roc_auc_score

WILSON_Z_95 = 1.96
MCNEMAR_CRITICAL_95 = 3.841


def wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of correct predictions.
        total: Number of evaluated samples.
        z: Normal quantile of the confidence level (default 1.96 for 95%).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(center - half), float(center + half)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate binary classification performance metrics.

    Class 1 = ALL (positive), Class 0 = Normal / HEM (negative).

    Args:
        y_true: Array of ground truth binary labels.
        y_pred: Array of predicted binary labels.
        y_prob: Optional array of positive-class probabilities used for AUROC.

    Returns:
        Dictionary containing TN, TP, FP, FN, accuracy, balanced accuracy, precision, recall, specificity, NPV,
        f1_score, the 95% Wilson interval of accuracy, and AUROC when probabilities are given.
    """
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    ci_low, ci_high = wilson_interval(tp + tn, total)

    metrics = {
        "TN": float(tn),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "Accuracy": float(accuracy),
        "Accuracy_CI95_Low": ci_low,
        "Accuracy_CI95_High": ci_high,
        "Balanced_Accuracy": float((recall + specificity) / 2.0),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "NPV": float(npv),
        "F_Score": float(f1_score),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def paper_table5_columns(metrics: dict[str, float]) -> dict[str, float]:
    """Map metrics onto the column labels of Baydilli (2025) Table 5 for a like-for-like comparison.

    The paper's columns are mislabelled: its row for the proposed model (TN=91, TP=87 on 100+100 test cells) lists
    "Precision" 0.8700 = TP/(TP+FN), "Recall" 0.9063 = TP/(TP+FP) and "Specificity" 0.8750 = TN/(TN+FN). Accuracy and
    F-Score are unaffected because F1 is symmetric in precision and recall.

    Args:
        metrics: Output of calculate_classification_metrics.

    Returns:
        Dictionary keyed by the paper's column names.
    """
    return {
        "TN": metrics["TN"],
        "TP": metrics["TP"],
        "Precision": metrics["Recall"],
        "Recall": metrics["Precision"],
        "Specificity": metrics["NPV"],
        "Accuracy": metrics["Accuracy"],
        "F-Score": metrics["F_Score"],
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Compare two classifiers on the same samples with McNemar's test, Eq. (9) of Baydilli (2025).

    Uses the continuity-corrected statistic chi2 = (|b - c| - 1)^2 / (b + c) with one degree of freedom, and also
    reports the exact two-sided binomial p-value, which is preferable when b + c < 25.

    Args:
        y_true: Array of ground truth binary labels.
        pred_a: Predictions of model A, aligned with y_true.
        pred_b: Predictions of model B, aligned with y_true.

    Returns:
        Dictionary with b (A correct, B wrong), c (A wrong, B correct), chi2, p_value_chi2, p_value_exact, and
        significant (1.0 when chi2 exceeds 3.841, the paper's alpha = 0.05 rule, else 0.0).
    """
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        chi2, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_chi2 = float(chi2_dist.sf(chi2, df=1))
        p_exact = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)

    return {
        "b": float(b),
        "c": float(c),
        "chi2": float(chi2),
        "p_value_chi2": p_chi2,
        "p_value_exact": p_exact,
        "significant": 1.0 if chi2 > MCNEMAR_CRITICAL_95 else 0.0,
    }


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two image batches.

    Args:
        img1: Tensor of shape (B, C, H, W).
        img2: Tensor of shape (B, C, H, W).
        max_val: Maximum possible pixel value (1.0 or 255.0).

    Returns:
        PSNR value in dB.
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val**2) / mse)
