"""PyTorch Lightning style logging and terminal output formatting.

Replicates the clean terminal style of PyTorch Lightning (model summary,
device diagnostics, progress formatting, and metrics tables).
"""

from typing import Any

import torch
import torch.nn as nn


def format_num_params(num_params: int) -> str:
    """Format parameter count into human-readable string (e.g. 11.4 M, 850 K).

    Args:
        num_params: Integer parameter count.

    Returns:
        Formatted parameter string.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f} K"
    return str(num_params)


def print_lightning_header(device: torch.device) -> None:
    """Print PyTorch Lightning device and accelerator diagnostics header.

    Args:
        device: Active torch device.
    """
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else "None"

    print(f"  * GPU available: {is_cuda} ({gpu_name}), used: {is_cuda}")
    print("  * TPU available: False, using: 0 TPU cores")
    print("  * IPU available: False, using: 0 IPUs")
    print("  * HPU available: False, using: 0 HPUs")
    if is_cuda:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"  * CUDA Device: {torch.cuda.current_device()} | Dedicated VRAM: {vram_gb:.2f} GB")
    print("-" * 70)


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print PyTorch Lightning style model layer and parameter summary table.

    Args:
        model: Target PyTorch neural network.
        model_name: Optional custom title for model.
    """
    rows: list[tuple[int, str, str, str]] = []
    total_params = 0
    trainable_params = 0

    for idx, (name, submodule) in enumerate(model.named_children()):
        sub_total = sum(p.numel() for p in submodule.parameters())
        sub_trainable = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        total_params += sub_total
        trainable_params += sub_trainable
        type_name = submodule.__class__.__name__
        rows.append((idx, name, type_name, format_num_params(sub_total)))

    non_trainable = total_params - trainable_params
    model_size_mb = (total_params * 4) / (1024 * 1024)

    print(f"\n  [Model: {model_name}]")
    print(f"  | Name{' ' * 16} | Type{' ' * 20} | Params")
    print("-" * 70)
    for r_idx, r_name, r_type, r_params in rows:
        name_pad = r_name.ljust(20)
        type_pad = r_type.ljust(24)
        params_pad = r_params.rjust(8)
        print(f"{r_idx:2d} | {name_pad} | {type_pad} | {params_pad}")
    print("-" * 70)
    print(f"{format_num_params(trainable_params).ljust(10)} Trainable params")
    print(f"{format_num_params(non_trainable).ljust(10)} Non-trainable params")
    print(f"{format_num_params(total_params).ljust(10)} Total params")
    print(f"{model_size_mb:.3f} MB   Total estimated model params size (FP32)\n")


def print_metrics_table(metrics: dict[str, Any], title: str = "Validation Results") -> None:
    """Print formatted metrics table in PyTorch Lightning terminal style.

    Args:
        metrics: Dictionary of metric names and numeric values.
        title: Table header title.
    """
    col_w_name = 28
    col_w_val = 18
    table_w = col_w_name + col_w_val + 7

    print("+" + "-" * (table_w - 2) + "+")
    print(f"| {title.center(table_w - 4)} |")
    print("+" + "-" * col_w_name + "+" + "-" * (col_w_val + 2) + "+")
    print(f"| {'Metric Name'.ljust(col_w_name - 2)} | {'Metric Value'.rjust(col_w_val)} |")
    print("+" + "-" * col_w_name + "+" + "-" * (col_w_val + 2) + "+")

    for k, v in metrics.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"| {k.ljust(col_w_name - 2)} | {val_str.rjust(col_w_val)} |")

    print("+" + "-" * (table_w - 2) + "+\n")
