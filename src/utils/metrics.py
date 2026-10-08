"""Evaluation metrics for classification and image translation quality.

Implements metrics from Table 4 and Table 5 of Baydilli (2025):
- Accuracy, Precision, Recall, Specificity, F-Score, Confusion Matrix
- PSNR, SSIM
"""

import numpy as np
import torch


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification performance metrics.

    Class 1 = ALL (positive), Class 0 = Normal / HEM (negative).

    Args:
        y_true: Array of ground truth binary labels.
        y_pred: Array of predicted binary labels.

    Returns:
        Dictionary containing TN, TP, FP, FN, accuracy, precision, recall,
        specificity, and f1_score.
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

    f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TN": float(tn),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "F_Score": float(f1_score),
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
