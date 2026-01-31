"""Utilities for loading trained DeltaMol models for MD."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..config.manager import load_config
from ..models import build_potential_model
from ..training.configs import PotentialExperimentConfig


@dataclass
class LoadedPotential:
    """Bundle holding a reconstructed model and metadata."""

    model: torch.nn.Module
    experiment: PotentialExperimentConfig
    checkpoint_path: Path


def load_trained_potential(
    experiment_path: str | Path,
    *,
    checkpoint_path: Optional[str | Path] = None,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> LoadedPotential:
    """Load a trained potential and its experiment metadata."""

    experiment_path = Path(experiment_path)
    experiment = load_config(experiment_path, PotentialExperimentConfig)
    if experiment.dataset.species is None:
        raise ValueError("Experiment config must define dataset.species to rebuild the model")
    model = build_potential_model(experiment.model, experiment.dataset.species)
    if checkpoint_path is None:
        checkpoint_path = experiment_path.parent / "potential.pt"
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
    else:
        model.load_state_dict(state)  # type: ignore[arg-type]
    model.to(device)
    if eval_mode:
        model.eval()
    return LoadedPotential(model=model, experiment=experiment, checkpoint_path=checkpoint_path)


__all__ = ["LoadedPotential", "load_trained_potential"]
