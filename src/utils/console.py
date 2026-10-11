"""Terminal output: device header, model parameter summary and metric tables."""

from typing import Any

import torch
import torch.nn as nn

RULE_WIDTH = 70


def format_num_params(num_params: int) -> str:
    """Format a parameter count as a short string (e.g. 11.4 M, 850.0 K).

    Args:
        num_params: Parameter count.

    Returns:
        Formatted count.
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f} K"
    return str(num_params)


def print_device_header(device: torch.device) -> None:
    """Print the execution device and, on CUDA, the GPU name and memory.

    Args:
        device: Active torch device.
    """
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"  * Device: {properties.name} (CUDA), {properties.total_memory / 1024**3:.2f} GB")
    else:
        print(f"  * Device: {device.type}")
    print("-" * RULE_WIDTH)


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print the parameter count of every top-level submodule and the totals.

    Args:
        model: Network to summarise.
        model_name: Title of the summary.
    """
    rows = [
        (name, type(child).__name__, sum(p.numel() for p in child.parameters()))
        for name, child in model.named_children()
    ]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  [Model: {model_name}]")
    print(f"   | {'Name':<20} | {'Type':<24} | {'Params':>8}")
    print("-" * RULE_WIDTH)
    for idx, (name, type_name, params) in enumerate(rows):
        print(f"{idx:2d} | {name:<20} | {type_name:<24} | {format_num_params(params):>8}")
    print("-" * RULE_WIDTH)
    print(f"{format_num_params(trainable):<10} Trainable params")
    print(f"{format_num_params(total - trainable):<10} Non-trainable params")
    print(f"{format_num_params(total):<10} Total params ({total * 4 / 1024**2:.1f} MB in FP32)\n")


def print_metrics_table(metrics: dict[str, Any], title: str = "Validation Results") -> None:
    """Print a two-column table of metric names and values.

    Args:
        metrics: Metric names and values; floats are shown with four decimals.
        title: Table title.
    """
    values = {name: f"{value:.4f}" if isinstance(value, float) else str(value) for name, value in metrics.items()}
    name_width = max(len("Metric"), *(len(name) for name in values))
    value_width = max(len("Value"), *(len(value) for value in values.values()))
    inner = max(name_width + value_width + 3, len(title))
    value_width = inner - name_width - 3

    border = "+" + "-" * (inner + 2) + "+"
    print(border)
    print(f"| {title.center(inner)} |")
    print(border)
    print(f"| {'Metric':<{name_width}} | {'Value':>{value_width}} |")
    print(border)
    for name, value in values.items():
        print(f"| {name:<{name_width}} | {value:>{value_width}} |")
    print(border + "\n")
