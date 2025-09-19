"""Evaluation utilities for DeltaMol models."""
from __future__ import annotations

from typing import Dict

import torch


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Return a dictionary with MAE, RMSE and MSE."""

    predictions = predictions.detach()
    targets = targets.detach()
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = mse ** 0.5
    return {"mse": mse, "rmse": rmse, "mae": mae}
