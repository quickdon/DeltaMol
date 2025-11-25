"""Graph convolutional potential models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .potential import PotentialOutput
from ..features.soap import AtomicSOAPConfig, AtomicSOAPDescriptor


@dataclass
class GCNConfig:
    """Configuration for :class:`GCNPotential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    use_coordinate_features: bool = True
    soap_num_radial: int = 8
    soap_cutoff: float = 5.0
    soap_gaussian_width: float = 0.5
    predict_forces: bool = False


class GCNLayer(nn.Module):
    """Lightweight GCN layer operating on dense adjacency matrices."""

    def __init__(self, in_dim: int, out_dim: int, *, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(adjacency, features)
        h = self.linear(h)
        h = self.activation(h)
        return self.dropout(h)


class GCNPotential(nn.Module):
    """Predict molecular energies (and optionally forces) with a GCN."""

    def __init__(self, config: GCNConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        if config.use_coordinate_features:
            soap_config = AtomicSOAPConfig(
                num_radial=config.soap_num_radial,
                cutoff=config.soap_cutoff,
                gaussian_width=config.soap_gaussian_width,
            )
            self.descriptor = AtomicSOAPDescriptor(soap_config)
            self.descriptor_projection = nn.Sequential(
                nn.Linear(config.soap_num_radial, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.descriptor = None
            self.descriptor_projection = None
        layers = []
        for _ in range(config.num_layers):
            layers.append(GCNLayer(config.hidden_dim, config.hidden_dim, dropout=config.dropout))
        self.layers = nn.ModuleList(layers)
        self.energy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.predict_forces:
            self.force_head = nn.Linear(config.hidden_dim, 3)
        else:
            self.force_head = None

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask = mask.bool()
        mask_float = mask.float()
        adj = self._normalize_adjacency(adjacency, mask_float)
        h = self.embedding(node_indices)
        if self.descriptor is not None and self.descriptor_projection is not None:
            descriptor = self.descriptor(positions, adjacency, mask)
            h = h + self.descriptor_projection(descriptor)
        for layer in self.layers:
            h = layer(adj, h)
        pooled = self._masked_mean(h, mask_float)
        energy = self.energy_head(pooled).squeeze(-1)
        forces = None
        if self.force_head is not None:
            forces = self.force_head(h) * mask_float.unsqueeze(-1)
        return PotentialOutput(energy=energy, forces=forces)

    def _normalize_adjacency(self, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(adjacency.size(-1), device=adjacency.device).unsqueeze(0)
        adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
        adjacency = adjacency + eye * mask.unsqueeze(-1)
        degree = adjacency.sum(dim=-1)
        inv_sqrt_degree = degree.clamp(min=1e-6).pow(-0.5)
        inv_sqrt_degree = inv_sqrt_degree * mask
        norm = adjacency * inv_sqrt_degree.unsqueeze(-1) * inv_sqrt_degree.unsqueeze(-2)
        return norm

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = tensor * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked.sum(dim=1) / denom


__all__ = ["GCNConfig", "GCNPotential"]

