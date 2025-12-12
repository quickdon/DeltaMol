"""SchNet potential energy model implementation.

This module provides a PyTorch implementation of the SchNet architecture
introduced in "SchNet: A continuous-filter convolutional neural network for
modeling quantum interactions" (Schütt et al., 2018). The implementation is
inspired by the reference code available at
https://github.com/atomistic-machine-learning/SchNet and follows the
continuous-filter convolution design for molecular graphs.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .potential import PotentialOutput


def _masked_distances(positions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise distances and neighbour mask.

    Args:
        positions: Tensor of shape ``(batch, atoms, 3)``.
        mask: Boolean tensor of shape ``(batch, atoms)``.

    Returns:
        distances: Tensor of shape ``(batch, atoms, atoms)``.
        neighbour_mask: Boolean tensor indicating valid pairs.
    """

    mask_bool = mask.bool()
    pos_i = positions.unsqueeze(2)
    pos_j = positions.unsqueeze(1)
    displacement = pos_i - pos_j
    distances = torch.linalg.norm(displacement, dim=-1)
    neighbour_mask = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
    return distances, neighbour_mask


class GaussianSmearing(nn.Module):
    """Expand interatomic distances using Gaussian basis functions."""

    def __init__(self, start: float, stop: float, num_gaussians: int):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("offset", offset)
        # Width mirrors the reference SchNet implementation.
        self.width = (stop - start) / float(num_gaussians)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        diff = distances.unsqueeze(-1) - self.offset
        return torch.exp(-0.5 * (diff / self.width) ** 2)


class InteractionBlock(nn.Module):
    """Single SchNet interaction block combining filter and atomwise updates."""

    def __init__(self, hidden_dim: int, num_filters: int, num_gaussians: int):
        super().__init__()
        self.dense_rbf = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.Softplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.dense_f = nn.Linear(hidden_dim, num_filters)
        self.dense_out = nn.Sequential(
            nn.Linear(num_filters, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, features: torch.Tensor, rbf: torch.Tensor, neighbour_mask: torch.Tensor
    ) -> torch.Tensor:
        # rbf: (batch, i, j, num_gaussians), features: (batch, atoms, hidden)
        filter_weights = self.dense_rbf(rbf)
        neighbour_features = self.dense_f(features).unsqueeze(1)
        messages = filter_weights * neighbour_features
        messages = messages * neighbour_mask.unsqueeze(-1)
        aggregated = messages.sum(dim=2)
        return features + self.dense_out(aggregated)


@dataclass
class SchNetConfig:
    """Configuration for :class:`SchNetPotential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    num_filters: Optional[int] = None
    num_interactions: int = 3
    num_gaussians: int = 50
    cutoff: float = 5.0
    predict_forces: bool = False


class SchNetPotential(nn.Module):
    """SchNet potential energy model with optional force prediction."""

    def __init__(self, config: SchNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        num_filters = config.num_filters or config.hidden_dim
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, config.cutoff, config.num_gaussians)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(config.hidden_dim, num_filters, config.num_gaussians)
                for _ in range(config.num_interactions)
            ]
        )
        self.atomwise = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Softplus(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        mask_float = mask_bool.float()
        if self.config.predict_forces and not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)

        grad_context = nullcontext()
        if self.config.predict_forces and not torch.is_grad_enabled():
            grad_context = torch.enable_grad()

        with grad_context:
            distances, neighbour_mask = _masked_distances(positions, mask_bool)
            # Apply cutoff-based neighbourhoods when no adjacency is supplied and
            # enforce the cutoff even when an adjacency is provided.
            cutoff_mask = distances <= self.config.cutoff
            if adjacency is None:
                neighbour_mask = neighbour_mask & cutoff_mask
            else:
                neighbour_mask = neighbour_mask & adjacency.bool() & cutoff_mask

            # Exclude self-interactions to mirror the original SchNet design.
            eye = torch.eye(
                neighbour_mask.size(1), device=neighbour_mask.device, dtype=torch.bool
            )
            neighbour_mask = neighbour_mask & ~eye.unsqueeze(0)
            rbf = self.distance_expansion(distances) * neighbour_mask.unsqueeze(-1)

            features = self.embedding(node_indices)
            for interaction in self.interactions:
                features = interaction(features, rbf, neighbour_mask)

            per_atom_energy = self.atomwise(features).squeeze(-1) * mask_float
            energy = per_atom_energy.sum(dim=1)

            forces = None
            if self.config.predict_forces:
                forces = -torch.autograd.grad(
                    energy.sum(), positions, create_graph=self.training
                )[0]
                forces = forces * mask_float.unsqueeze(-1)

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["SchNetConfig", "SchNetPotential"]
