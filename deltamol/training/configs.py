"""Dataclasses describing configurable training components."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

from .pipeline import PotentialTrainingConfig


@dataclass
class DatasetConfig:
    """Options that control dataset preparation for potential training."""

    path: Optional[Path] = None
    format: Optional[str] = None
    test_path: Optional[Path] = None
    test_format: Optional[str] = None
    cutoff: float = 5.0
    dtype: str = "float32"
    species: Optional[Tuple[int, ...]] = None
    key_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class BaselineConfig:
    """Configuration describing a precomputed linear baseline model."""

    checkpoint: Optional[Path] = None
    species: Optional[Tuple[int, ...]] = None
    requires_grad: bool = True


@dataclass
class ModelConfig:
    """Lightweight description of the neural potential to construct."""

    name: Literal["gcn", "transformer", "external", "hybrid", "se3", "schnet"] = "transformer"
    hidden_dim: int = 128
    gcn_layers: int = 2
    transformer_layers: int = 2
    dropout: float = 0.1
    use_coordinate_features: bool = True
    predict_forces: bool = False
    num_heads: int = 4
    ffn_dim: int = 256
    cutoff: float = 5.0
    soap_num_radial: int = 8
    soap_cutoff: float = 5.0
    soap_gaussian_width: float = 0.5
    residual_mode: bool = True
    adapter: Optional[str] = None
    adapter_weights: Optional[Path] = None
    neighbor_strategy: str = "adjacency"
    neighbor_cutoff: Optional[float] = None
    se3_distance_embedding: int = 32
    se3_layers: Optional[int] = None
    schnet_num_interactions: int = 3
    schnet_num_gaussians: int = 50
    schnet_num_filters: Optional[int] = None


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

