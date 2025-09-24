"""Transformer-based potential energy model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .potential import PotentialOutput


@dataclass
class TransformerConfig:
    """Configuration options for :class:`TransformerPotential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ffn_dim: int = 256
    use_coordinate_features: bool = True
    predict_forces: bool = False


class TransformerPotential(nn.Module):
    """Encode atom-wise features with a Transformer encoder."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        if config.use_coordinate_features:
            self.coordinate_mlp = nn.Sequential(
                nn.Linear(3, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.coordinate_mlp = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
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
        adjacency: torch.Tensor,  # kept for API parity, not used directly
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask = mask.bool()
        mask_float = mask.float()
        x = self.embedding(node_indices)
        if self.coordinate_mlp is not None:
            x = x + self.coordinate_mlp(positions)
        encoded = self.encoder(x, src_key_padding_mask=~mask)
        pooled = self._masked_mean(encoded, mask_float)
        energy = self.energy_head(pooled).squeeze(-1)
        forces = None
        if self.force_head is not None:
            forces = self.force_head(encoded) * mask_float.unsqueeze(-1)
        return PotentialOutput(energy=energy, forces=forces)

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = tensor * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked.sum(dim=1) / denom


__all__ = ["TransformerConfig", "TransformerPotential"]

