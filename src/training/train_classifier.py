"""Training script for ResNet34 Leukemia Classifier.

Trains the classifier on translated source images (or a source/target baseline) and evaluates on the unseen target
test set exactly once. Baydilli (2025) trains for 50 epochs and tests the last epoch; an optional source-domain
validation set selects the epoch instead. The target test set is never used for model selection.
Terminal output matches PyTorch Lightning aesthetics.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LeukemiaClassificationDataset, get_default_transform
from src.models.classifier import LeukemiaClassifier
from src.utils.lightning_logger import (
    print_lightning_header,
    print_metrics_table,
    print_model_summary,
)
from src.utils.metrics import calculate_classification_metrics, paper_table5_columns


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate classifier on a dataset.

    Args:
        model: Trained classifier model.
        dataloader: Evaluation DataLoader.
        device: Device to run evaluation on.

    Returns:
        Tuple of (metrics_dict, y_true, y_pred).
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = calculate_classification_metrics(y_true, y_pred, np.array(all_probs))
    return metrics, y_true, y_pred


def train_classifier(
    train_dir: str,
    test_dir: str,
    val_dir: str | None = None,
    output_dir: str = "checkpoints/classifier",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    image_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[LeukemiaClassifier, dict[str, float]]:
    """Train ResNet34 classifier and evaluate once on the target test set.

    Args:
        train_dir: Path to training data directory containing 'all' and 'hem'.
        test_dir: Path to target test directory containing 'all' and 'hem'.
        val_dir: Optional source-domain validation directory used to select the epoch. Without it the last epoch
            is evaluated.
        output_dir: Directory to save trained model checkpoints and results.json.
        epochs: Number of training epochs (default 50).
        batch_size: Batch size (default 32).
        lr: Learning rate (default 0.001).
        weight_decay: Adam weight decay (default 0, the paper specifies none).
        image_size: Image input size (default 128).
        device: Target execution device.

    Returns:
        Tuple of (selected LeukemiaClassifier, target test metrics).
    """
    target_device = torch.device(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print_lightning_header(target_device)

    train_transform = get_default_transform(image_size=image_size, is_train=True)
    eval_transform = get_default_transform(image_size=image_size, is_train=False)

    train_dataset = LeukemiaClassificationDataset(train_dir, transform=train_transform)
    test_dataset = LeukemiaClassificationDataset(test_dir, transform=eval_transform)
    val_dataset = LeukemiaClassificationDataset(val_dir, transform=eval_transform) if val_dir else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None

    model = LeukemiaClassifier(num_classes=2, pretrained=True).to(target_device)
    print_model_summary(model, model_name="ResNet34 LeukemiaClassifier")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    selected_epoch = epochs
    selected_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:2d}/{epochs}",
            bar_format="{l_bar}{bar:25}{r_bar}",
            dynamic_ncols=True,
        )

        for images, labels in pbar:
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            current_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "v_num": 0})

        if val_loader is not None:
            val_metrics, _, _ = evaluate(model, val_loader, target_device)
            if val_metrics["Accuracy"] > best_val_acc:
                best_val_acc = val_metrics["Accuracy"]
                selected_epoch = epoch
                selected_state = copy.deepcopy(model.state_dict())
            if epoch % 5 == 0 or epoch == epochs:
                print_metrics_table(val_metrics, title=f"Epoch {epoch} Source Validation Metrics")

    torch.save(model.state_dict(), out_path / "last_classifier.pth")
    if selected_state is not None:
        model.load_state_dict(selected_state)
        torch.save(selected_state, out_path / "selected_classifier.pth")

    test_metrics, _, y_pred = evaluate(model, test_loader, target_device)
    selection_rule = f"best source-validation accuracy (epoch {selected_epoch})" if val_loader else "last epoch"
    print_metrics_table(test_metrics, title=f"Target Test Results ({selection_rule})")
    paper_view = paper_table5_columns(test_metrics)
    print_metrics_table(paper_view, title="Same Results in Baydilli (2025) Table 5 Column Convention")

    result = {
        "selection_rule": selection_rule,
        "selected_epoch": selected_epoch,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset) if val_dataset else 0,
        "num_test": len(test_dataset),
        "test_metrics": test_metrics,
        "test_metrics_paper_table5_columns": paper_view,
        "test_predictions": [
            {"file": str(path), "label": int(label), "pred": int(pred)}
            for (path, label), pred in zip(test_dataset.samples, y_pred, strict=True)
        ],
    }
    with open(out_path / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    return model, test_metrics
