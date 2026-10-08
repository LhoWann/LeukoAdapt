"""Training script for ResNet34 Leukemia Classifier.

Trains classifier on translated source images (or original source baseline)
and evaluates on unseen target test images (ALL-IDB 200 samples).
Terminal output matches PyTorch Lightning aesthetics.
"""

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
from src.utils.metrics import calculate_classification_metrics


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
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = calculate_classification_metrics(y_true, y_pred)
    return metrics, y_true, y_pred


def train_classifier(
    train_dir: str,
    test_dir: str,
    output_dir: str = "checkpoints/classifier",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    image_size: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> LeukemiaClassifier:
    """Train ResNet34 classifier and evaluate on target test set.

    Args:
        train_dir: Path to training data directory containing 'all' and 'hem'.
        test_dir: Path to test data directory containing 'all' and 'hem'.
        output_dir: Directory to save trained model checkpoints.
        epochs: Number of training epochs (default 50).
        batch_size: Batch size (default 32).
        lr: Learning rate (default 0.001).
        image_size: Image input size (default 128).
        device: Target execution device.

    Returns:
        Trained LeukemiaClassifier model.
    """
    target_device = torch.device(device)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print_lightning_header(target_device)

    # Datasets and Loaders
    train_transform = get_default_transform(image_size=image_size, is_train=True)
    test_transform = get_default_transform(image_size=image_size, is_train=False)

    train_dataset = LeukemiaClassificationDataset(train_dir, transform=train_transform)
    test_dataset = LeukemiaClassificationDataset(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Model, Loss, Optimizer
    model = LeukemiaClassifier(num_classes=2, pretrained=True).to(target_device)
    print_model_summary(model, model_name="ResNet34 LeukemiaClassifier")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_metrics = {}

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

        # Evaluation at each epoch
        test_metrics, _, _ = evaluate(model, test_loader, target_device)
        test_acc = test_metrics["Accuracy"]

        if test_acc > best_acc:
            best_acc = test_acc
            best_metrics = test_metrics
            torch.save(model.state_dict(), out_path / "best_classifier.pth")

        if epoch % 5 == 0 or epoch == epochs:
            print_metrics_table(test_metrics, title=f"Epoch {epoch} Test Metrics")

    torch.save(model.state_dict(), out_path / "last_classifier.pth")
    print_metrics_table(best_metrics, title="Final Best Test Results on ALL-IDB Target")
    return model
