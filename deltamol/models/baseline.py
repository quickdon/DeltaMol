"""Baseline models for DeltaMol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def build_formula_vector(atomic_numbers: Iterable[int], *, species: Iterable[int]) -> torch.Tensor:
    mapping = {int(z): i for i, z in enumerate(species)}
    vector = torch.zeros(len(mapping), dtype=torch.float32)
    for number in atomic_numbers:
        idx = mapping[int(number)]
        vector[idx] += 1.0
    return vector


@dataclass
class LinearBaselineConfig:
    """Configuration for the linear atomic energy baseline."""

    species: tuple[int, ...]


class LinearAtomicBaseline(nn.Module):
    """Predict the total energy as a linear combination of atom counts."""

    def __init__(self, config: LinearBaselineConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(len(config.species), 1, bias=False)

    def forward(self, formula_vectors: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return self.linear(formula_vectors).squeeze(-1)


__all__ = [
    "LinearAtomicBaseline",
    "LinearBaselineConfig",
    "build_formula_vector",
]
