"""DimeNet-inspired potential energy model.

This module implements a lightweight DimeNet-style architecture following
Klicpera, Groß, and Günnemann, "Directional Message Passing for Molecular
Graphs" (ICLR 2020) and the open-source reference implementation at
https://github.com/gasteigerjo/dimenet. The model combines radial Bessel basis
functions with angle embeddings over atom triplets to capture directional
interactions when predicting molecular energies and forces.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


def _masked_distances(positions: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise distances and neighbour masks.

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


class RadialBesselLayer(nn.Module):
    """Radial Bessel basis with smooth cutoff envelope."""

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)
        freq = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * torch.pi
        self.register_buffer("freq", freq)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:  # pragma: no cover - mathematical
        distances = torch.clamp(distances, min=1e-8, max=self.cutoff)
        scaled = distances / self.cutoff
        # Smooth envelope from the DimeNet paper that goes to zero at the cutoff.
        envelope = 1 - 6 * scaled**5 + 15 * scaled**4 - 10 * scaled**3
        bessel = torch.sin(self.freq * distances / self.cutoff) / distances
        norm = torch.sqrt(torch.tensor(2.0 / self.cutoff, device=distances.device, dtype=distances.dtype))
        return norm * envelope.unsqueeze(-1) * bessel.unsqueeze(-1)


class AngleEmbedding(nn.Module):
    """Embed angles using a Fourier-style basis over ``acos``."""

    def __init__(self, num_spherical: int, hidden_dim: int):
        super().__init__()
        self.num_spherical = num_spherical
        self.mlp = nn.Sequential(
            nn.Linear(num_spherical, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, cos_angles: torch.Tensor) -> torch.Tensor:  # pragma: no cover - mathematical
        cos_clamped = torch.clamp(cos_angles, -1 + 1e-7, 1 - 1e-7)
        angles = torch.acos(cos_clamped)
        k = torch.arange(1, self.num_spherical + 1, device=cos_angles.device, dtype=cos_angles.dtype)
        basis = torch.cos(angles.unsqueeze(-1) * k)
        return self.mlp(basis)


class DimeNetInteraction(nn.Module):
    """Single directional message passing block."""

    def __init__(self, hidden_dim: int, num_spherical: int):
        super().__init__()
        self.angle_embedding = AngleEmbedding(num_spherical, hidden_dim)
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        directions: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate directional messages.

        Args:
            edge_features: Tensor of shape ``(batch, atoms, neighbours, hidden)`` with
                the central atom on dimension 1.
            directions: Unit displacement vectors of shape ``(batch, atoms, neighbours, 3)``
                aligned with ``edge_features``.
            edge_mask: Boolean mask of shape ``(batch, atoms, neighbours)``.
        """

        # Compute angles for all triplets centred on j with neighbours i and k.
        cos_angles = torch.einsum("bjid,bjkd->bjik", directions, directions)
        triplet_mask = edge_mask.unsqueeze(2) & edge_mask.unsqueeze(3)
        # Remove degenerate angles where i == k.
        num_atoms = edge_features.size(1)
        diag = torch.eye(num_atoms, device=edge_features.device, dtype=torch.bool).view(1, 1, num_atoms, num_atoms)
        triplet_mask = triplet_mask & ~diag

        angle_emb = self.angle_embedding(cos_angles)
        angle_emb = angle_emb * triplet_mask.unsqueeze(-1)

        # Weight neighbour edge features by angle embeddings.
        neighbour_messages = (angle_emb * edge_features.unsqueeze(2)).sum(dim=3)
        updated = self.message_proj(torch.cat([edge_features, neighbour_messages], dim=-1))
        return edge_features + updated * edge_mask.unsqueeze(-1)


@dataclass
class DimeNetConfig:
    """Configuration for :class:`DimeNetPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_blocks: int = 3
    num_radial: int = 6
    num_spherical: int = 7
    cutoff: float = 5.0
    predict_forces: bool = False


class DimeNetPotential(nn.Module):
    """DimeNet-style potential energy model with optional force prediction."""

    def __init__(self, config: DimeNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.radial_basis = RadialBesselLayer(config.num_radial, config.cutoff)
        self.edge_init = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + config.num_radial, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [DimeNetInteraction(config.hidden_dim, config.num_spherical) for _ in range(config.num_blocks)]
        )
        self.node_update = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        mask_float = mask_bool.float()
        if self.config.predict_forces and not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)

        distances, neighbour_mask = _masked_distances(positions, mask_bool)
        adjacency_mask = neighbour_mask & (adjacency > 0)

        # Align to (batch, center, neighbour, ...).
        distances = distances.transpose(1, 2)
        adjacency_mask = adjacency_mask.transpose(1, 2)

        radial = self.radial_basis(distances) * adjacency_mask.unsqueeze(-1)
        node_embeddings = self.embedding(node_indices)
        center_emb = node_embeddings.unsqueeze(2)
        neighbour_emb = node_embeddings.unsqueeze(1)
        edge_input = torch.cat([radial, center_emb.expand_as(radial), neighbour_emb.expand_as(radial)], dim=-1)
        edge_features = self.edge_init(edge_input) * adjacency_mask.unsqueeze(-1)

        displacement = positions.unsqueeze(2) - positions.unsqueeze(1)
        displacement = displacement.transpose(1, 2)
        directions = displacement / (distances.unsqueeze(-1) + 1e-8)

        for block in self.blocks:
            edge_features = block(edge_features, directions, adjacency_mask)

        aggregated = edge_features * adjacency_mask.unsqueeze(-1)
        aggregated = aggregated.sum(dim=2)
        node_state = self.node_update(torch.cat([node_embeddings, aggregated], dim=-1))
        per_atom_energy = self.output(node_state).squeeze(-1) * mask_float
        energy = per_atom_energy.sum(dim=1)

        forces = None
        if self.config.predict_forces:
            forces = -torch.autograd.grad(energy.sum(), positions, create_graph=self.training)[0]
            forces = forces * mask_float.unsqueeze(-1)

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["DimeNetConfig", "DimeNetPotential"]
