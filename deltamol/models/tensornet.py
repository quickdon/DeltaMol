"""TensorNet-inspired potential energy model.

This module implements a lightweight adaptation of the TensorNet architecture
from `torchmd/torchmd-net <https://github.com/torchmd/torchmd-net>`_. The
original model combines tensor products between learned scalar and vector
features to capture directional information. The simplified variant below keeps
the same spirit for testing: it mixes radial distance embeddings with
orientation-aware filters to update atomic embeddings before predicting total
energies and optional analytic forces.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


class GaussianRadialBasis(nn.Module):
    """Expand distances with Gaussian basis functions."""

    def __init__(self, cutoff: float, num_radial: int):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_radial)
        self.register_buffer("centers", centers)
        self.width = cutoff / max(num_radial, 1)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / (self.width + 1e-8)) ** 2)


class TensorNetBlock(nn.Module):
    """Orientation-aware interaction block inspired by TensorNet."""

    def __init__(self, hidden_dim: int, direction_dim: int, num_radial: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.direction_net = nn.Sequential(
            nn.Linear(3, direction_dim),
            nn.SiLU(),
            nn.Linear(direction_dim, hidden_dim),
        )
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        rbf: torch.Tensor,
        displacement: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        filters = self.filter_net(rbf)

        # Normalise displacement to obtain orientation vectors. Mask avoids NaNs.
        norm = displacement.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        direction = displacement / norm
        direction = direction * edge_mask.unsqueeze(-1)
        direction_emb = self.direction_net(direction)

        neighbour = self.neighbor_proj(features)
        neighbour = neighbour.unsqueeze(1)  # (B, 1, N, H)
        filters = filters + direction_emb
        message = neighbour * filters  # broadcast over central atoms
        message = message * edge_mask.unsqueeze(-1)
        aggregated = message.sum(dim=2)

        update = self.update_net(torch.cat([features, aggregated], dim=-1))
        return features + update


@dataclass
class TensorNetConfig:
    """Configuration for :class:`TensorNetPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 3
    num_radial: int = 16
    direction_dim: int = 32
    cutoff: float = 5.0
    predict_forces: bool = False


class TensorNetPotential(nn.Module):
    """TensorNet-style potential with optional force prediction."""

    def __init__(self, config: TensorNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)

        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.radial_basis = GaussianRadialBasis(config.cutoff, config.num_radial)
        self.blocks = nn.ModuleList(
            [TensorNetBlock(config.hidden_dim, config.direction_dim, config.num_radial) for _ in range(config.num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _build_masks(
        self,
        positions: torch.Tensor,
        mask: torch.Tensor,
        adjacency: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_bool = mask.bool()
        displacement = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(displacement, dim=-1)

        edge_mask = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
        cutoff_mask = distances <= self.config.cutoff
        edge_mask = edge_mask & cutoff_mask
        if adjacency is not None:
            edge_mask = edge_mask & adjacency.bool()

        eye = torch.eye(edge_mask.size(1), device=edge_mask.device, dtype=torch.bool)
        edge_mask = edge_mask & ~eye.unsqueeze(0)
        return displacement, distances, edge_mask

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
            displacement, distances, edge_mask = self._build_masks(positions, mask_bool, adjacency)
            rbf = self.radial_basis(distances)
            rbf = rbf * edge_mask.unsqueeze(-1)

            features = self.embedding(node_indices)
            features = features * mask_float.unsqueeze(-1)

            for block in self.blocks:
                features = block(features, rbf, displacement, edge_mask)

            per_atom = self.readout(features).squeeze(-1)
            energy = (per_atom * mask_float).sum(dim=1)

        forces = None
        if self.config.predict_forces:
            grads = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=self.training,
                retain_graph=self.training,
                allow_unused=True,
            )[0]
            if grads is None:
                grads = torch.zeros_like(positions)
            forces = -grads
            forces = forces * mask_float.unsqueeze(-1)

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["TensorNetConfig", "TensorNetPotential"]

