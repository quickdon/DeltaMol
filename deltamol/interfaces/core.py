"""Shared helpers for MD engine interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from ..models.potential import PotentialOutput


@dataclass
class PotentialPrediction:
    """Container for single-structure energy and forces."""

    energy: float
    forces: np.ndarray


class PotentialCalculator:
    """Prepare inputs and evaluate DeltaMol models for MD engines."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        species: Sequence[int],
        cutoff: float = 5.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.species = tuple(int(z) for z in species)
        self.cutoff = float(cutoff)
        self.device = torch.device(device)
        self.dtype = dtype
        self.index_map = {z: i + 1 for i, z in enumerate(self.species)}

    def _map_atomic_numbers(self, numbers: Iterable[int]) -> torch.Tensor:
        indices = [self.index_map[int(z)] for z in numbers]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _build_adjacency(self, positions: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(positions, positions)
        adjacency = (distances < self.cutoff).to(positions.dtype)
        adjacency.fill_diagonal_(0.0)
        return adjacency

    def predict(
        self,
        atomic_numbers: Iterable[int],
        positions: np.ndarray | torch.Tensor,
        *,
        require_forces: bool = True,
    ) -> PotentialPrediction:
        """Predict energy and forces for a single structure."""

        node_indices = self._map_atomic_numbers(atomic_numbers)
        if not torch.is_tensor(positions):
            positions = torch.tensor(np.asarray(positions, dtype=float), dtype=self.dtype)
        positions = positions.to(self.device)
        positions = positions.clone().requires_grad_(require_forces)
        adjacency = self._build_adjacency(positions)
        mask = torch.ones(node_indices.shape, dtype=torch.bool, device=self.device)
        batched = {
            "node_indices": node_indices.unsqueeze(0),
            "positions": positions.unsqueeze(0),
            "adjacency": adjacency.unsqueeze(0),
            "mask": mask.unsqueeze(0),
        }
        with torch.set_grad_enabled(require_forces):
            output = self.model(
                batched["node_indices"],
                batched["positions"],
                batched["adjacency"],
                batched["mask"],
            )
        prediction = self._resolve_prediction(output, positions, require_forces=require_forces)
        return prediction

    def _resolve_prediction(
        self,
        output: PotentialOutput,
        positions: torch.Tensor,
        *,
        require_forces: bool,
    ) -> PotentialPrediction:
        energy = output.energy.detach().squeeze().to("cpu").item()
        if output.forces is not None:
            forces = output.forces.detach().squeeze(0).to("cpu").numpy()
            return PotentialPrediction(energy=energy, forces=forces)
        if not require_forces:
            raise RuntimeError("Model did not return forces and require_forces=False")
        grads = torch.autograd.grad(output.energy.sum(), positions)[0]
        forces = (-grads).detach().to("cpu").numpy()
        return PotentialPrediction(energy=energy, forces=forces)


__all__ = ["PotentialCalculator", "PotentialPrediction"]
