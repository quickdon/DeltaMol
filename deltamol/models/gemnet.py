"""Lightweight GemNet-inspired potential energy model.

This module provides a simplified GemNet-style architecture for potential
training, drawing inspiration from the directional message passing design in
the reference TensorFlow implementation at
`TUM-DAML/gemnet_tf <https://github.com/TUM-DAML/gemnet_tf>`_. The PyTorch
implementation below focuses on testing workflows by combining radial distance
embeddings, directional edge features, and residual node updates while keeping
the configuration surface compact.
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


class GemNetBlock(nn.Module):
    """Single directional message passing block."""

    def __init__(self, hidden_dim: int, num_radial: int, num_spherical: int):
        super().__init__()
        self.rbf_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(num_spherical, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        rbf: torch.Tensor,
        directions: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Embed radial basis and directions.
        rbf_emb = self.rbf_mlp(rbf)
        dir_emb = self.dir_mlp(directions)

        # Prepare central and neighbour node features for pairwise messages.
        central = features.unsqueeze(2).expand(-1, -1, features.size(1), -1)
        neighbour = features.unsqueeze(1).expand(-1, features.size(1), -1, -1)

        message_input = torch.cat([central, neighbour, rbf_emb, dir_emb], dim=-1)
        messages = self.edge_mlp(message_input)
        messages = messages * edge_mask.unsqueeze(-1)

        aggregated = messages.sum(dim=2)
        update = self.node_mlp(torch.cat([features, aggregated], dim=-1))
        return features + update


@dataclass
class GemNetConfig:
    """Configuration for :class:`GemNetPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_blocks: int = 3
    num_radial: int = 6
    num_spherical: int = 4
    cutoff: float = 5.0
    predict_forces: bool = False


class GemNetPotential(nn.Module):
    """GemNet-inspired potential with optional force prediction."""

    def __init__(self, config: GemNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)

        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.radial_basis = GaussianRadialBasis(config.cutoff, config.num_radial)
        self.blocks = nn.ModuleList(
            [GemNetBlock(config.hidden_dim, config.num_radial, config.num_spherical) for _ in range(config.num_blocks)]
        )
        self.direction_proj = nn.Linear(3, config.num_spherical)

        self.output_mlp = nn.Sequential(
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

            # Directional unit vectors scaled by mask to avoid NaNs on padded entries.
            direction = displacement / (distances.unsqueeze(-1) + 1e-8)
            direction = direction * edge_mask.unsqueeze(-1)

            rbf = self.radial_basis(distances)
            rbf = rbf * edge_mask.unsqueeze(-1)

            # Project raw directions to a compact set of spherical features.
            dir_features = torch.tanh(self.direction_proj(direction))

            features = self.embedding(node_indices)
            for block in self.blocks:
                features = block(features, rbf, dir_features, edge_mask)

            per_atom_energy = self.output_mlp(features).squeeze(-1) * mask_float
            energy = per_atom_energy.sum(dim=1)

            forces = None
            if self.config.predict_forces:
                forces = -torch.autograd.grad(energy.sum(), positions, create_graph=self.training)[0]
                forces = forces * mask_float.unsqueeze(-1)

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["GemNetConfig", "GemNetPotential"]
