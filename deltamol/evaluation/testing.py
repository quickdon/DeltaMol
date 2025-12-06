"""Utilities for evaluating trained models on held-out datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

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
) -> Tuple[
    Dict[str, float],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
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
    try:
        target_dtype = next(model.parameters()).dtype
    except StopIteration:
        target_dtype = torch.float32
    if baseline is not None:
        baseline = baseline.to(device=model_device, dtype=target_dtype)
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
    energy_predictions = []
    energy_targets = []
    force_predictions = []
    force_targets = []
    for batch in loader:
        cast_batch = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            moved = value.to(model_device)
            if moved.is_floating_point():
                moved = moved.to(target_dtype)
            cast_batch[key] = moved
        batch = cast_batch
        forces_available = batch.get("forces") is not None
        positions = batch["positions"]
        if forces_available:
            positions = positions.detach().clone().requires_grad_(True)
            batch["positions"] = positions
        energy_target = batch["energies"]
        baseline_energy = None
        if baseline is not None and residual_mode:
            baseline_energy = baseline(batch["formula_vectors"])
        context = torch.enable_grad() if forces_available else torch.no_grad()
        with context:
            output = model(
                batch["node_indices"],
                batch["positions"],
                batch["adjacency"],
                batch["mask"],
            )
            energy_pred = output.energy
            if baseline_energy is not None:
                energy_pred = energy_pred + baseline_energy
        energy_predictions.append(energy_pred.detach().cpu())
        energy_targets.append(energy_target.detach().cpu())
        if forces_available:
            mask = batch["mask"].to(model_device)
            mask_expanded = mask.unsqueeze(-1).expand_as(batch["positions"])
            if output.forces is not None:
                predicted_forces = output.forces
            else:
                grads = torch.autograd.grad(
                    energy_pred.sum(),
                    positions,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                predicted_forces = -grads
            target_forces = batch["forces"].to(model_device)
            valid_mask = mask_expanded.bool()
            predicted_forces = predicted_forces[valid_mask]
            target_forces = target_forces[valid_mask]
            force_predictions.append(predicted_forces.detach().cpu())
            force_targets.append(target_forces.detach().cpu())
    all_predictions = torch.cat(energy_predictions) if energy_predictions else torch.tensor([])
    all_targets = torch.cat(energy_targets) if energy_targets else torch.tensor([])
    metrics = compute_regression_metrics(all_predictions, all_targets)
    all_force_predictions_tensor: Optional[torch.Tensor]
    all_force_targets_tensor: Optional[torch.Tensor]
    all_force_predictions_tensor = None
    all_force_targets_tensor = None
    if force_predictions:
        all_force_predictions_tensor = torch.cat(force_predictions)
        all_force_targets_tensor = torch.cat(force_targets)
        force_metrics = compute_regression_metrics(
            all_force_predictions_tensor, all_force_targets_tensor
        )
        metrics.update({f"force_{name}": value for name, value in force_metrics.items()})
        if all_force_predictions_tensor.numel() % 3 == 0:
            per_atom_force_pred = all_force_predictions_tensor.view(-1, 3)
            per_atom_force_target = all_force_targets_tensor.view(-1, 3)
            per_atom_diff = per_atom_force_pred - per_atom_force_target
            per_atom_component_mse = (per_atom_diff**2).mean(dim=1)
            per_atom_component_mae = per_atom_diff.abs().mean(dim=1)
            per_atom_metrics = {
                "mse": per_atom_component_mse.mean().item(),
                "rmse": per_atom_component_mse.mean().sqrt().item(),
                "mae": per_atom_component_mae.mean().item(),
            }
            metrics.update(
                {f"force_per_atom_{name}": value for name, value in per_atom_metrics.items()}
            )
    return (
        metrics,
        all_predictions,
        all_targets,
        all_force_predictions_tensor,
        all_force_targets_tensor,
    )


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
    "save_predictions_and_targets",
    "save_force_predictions_and_targets",
]


def save_predictions_and_targets(
    predictions: torch.Tensor, targets: torch.Tensor, output_path: Path
) -> Path:
    """Persist flattened predictions and targets to an ``.npz`` archive."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        predictions=predictions.detach().cpu().numpy().reshape(-1),
        targets=targets.detach().cpu().numpy().reshape(-1),
    )
    return output_path


def save_force_predictions_and_targets(
    predictions: torch.Tensor, targets: torch.Tensor, output_path: Path
) -> Path:
    """Persist force predictions and targets while preserving atom triplets."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_cpu = predictions.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    if predictions_cpu.size % 3 == 0 and targets_cpu.size % 3 == 0:
        predictions_cpu = predictions_cpu.reshape(-1, 3)
        targets_cpu = targets_cpu.reshape(-1, 3)
    np.savez(output_path, predictions=predictions_cpu, targets=targets_cpu)
    return output_path

