"""Utilities for evaluating trained models on held-out datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import compute_regression_metrics
from ..models.baseline import LinearAtomicBaseline


def evaluate_baseline_model(
    model: LinearAtomicBaseline,
    dataset: Dataset,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate a baseline model on a dataset returning metrics and predictions."""

    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")
    if device is not None:
        model_device = device
    else:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(model_device)
            target = target.to(model_device)
            outputs = model(inputs)
            predictions.append(outputs.detach().cpu())
            targets.append(target.detach().cpu())
    all_predictions = torch.cat(predictions) if predictions else torch.tensor([])
    all_targets = torch.cat(targets) if targets else torch.tensor([])
    metrics = compute_regression_metrics(all_predictions, all_targets)
    return metrics, all_predictions, all_targets


def evaluate_potential_model(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    baseline: LinearAtomicBaseline | None = None,
    residual_mode: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate a potential model on a dataset returning metrics and predictions."""

    from ..training.datasets import collate_graphs

    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")
    if device is not None:
        model_device = device
    else:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graphs,
    )
    model.eval()
    if baseline is not None:
        baseline.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            energy_target = batch["energies"]
            baseline_energy = None
            if baseline is not None and residual_mode:
                baseline_energy = baseline(batch["formula_vectors"])
            output = model(
                batch["node_indices"],
                batch["positions"],
                batch["adjacency"],
                batch["mask"],
            )
            energy_pred = output.energy
            if baseline_energy is not None:
                energy_pred = energy_pred + baseline_energy
            predictions.append(energy_pred.detach().cpu())
            targets.append(energy_target.detach().cpu())
    all_predictions = torch.cat(predictions) if predictions else torch.tensor([])
    all_targets = torch.cat(targets) if targets else torch.tensor([])
    metrics = compute_regression_metrics(all_predictions, all_targets)
    return metrics, all_predictions, all_targets


def plot_predictions_vs_targets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> Path:
    """Save a scatter plot comparing predictions with ground truth values."""

    import matplotlib.pyplot as plt

    preds = predictions.flatten().cpu().numpy()
    trues = targets.flatten().cpu().numpy()
    if preds.size == 0 or trues.size == 0:
        raise ValueError("No predictions or targets available for plotting")
    plt.figure(figsize=(5, 5))
    plt.scatter(trues, preds, alpha=0.6, edgecolor="none")
    min_val = float(min(preds.min(), trues.min()))
    max_val = float(max(preds.max(), trues.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    if title:
        plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


__all__ = [
    "evaluate_baseline_model",
    "evaluate_potential_model",
    "plot_predictions_vs_targets",
]

