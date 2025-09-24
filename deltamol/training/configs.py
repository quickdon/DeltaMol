"""Dataclasses describing configurable training components."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from .pipeline import PotentialTrainingConfig


@dataclass
class DatasetConfig:
    """Options that control dataset preparation for potential training."""

    path: Optional[Path] = None
    cutoff: float = 5.0
    dtype: str = "float32"
    species: Optional[Tuple[int, ...]] = None


@dataclass
class BaselineConfig:
    """Configuration describing a precomputed linear baseline model."""

    checkpoint: Optional[Path] = None
    species: Optional[Tuple[int, ...]] = None
    requires_grad: bool = True


@dataclass
class ModelConfig:
    """Lightweight description of the neural potential to construct."""

    name: str = "gcn"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_coordinate_features: bool = True
    predict_forces: bool = False
    num_heads: int = 8
    ffn_dim: int = 256


@dataclass
class PotentialExperimentConfig:
    """Bundle dataset, model, and training configuration together."""

    training: PotentialTrainingConfig
    model: ModelConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    baseline: Optional[BaselineConfig] = None


__all__ = [
    "BaselineConfig",
    "DatasetConfig",
    "ModelConfig",
    "PotentialExperimentConfig",
]

