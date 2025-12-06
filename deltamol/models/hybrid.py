"""Hybrid SOAP + graph attention architecture for potential energy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .potential import PotentialOutput
from ..features.soap import AtomicSOAPConfig, AtomicSOAPDescriptor


@dataclass
class HybridPotentialConfig:
    """Configuration for :class:`HybridPotential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    gcn_layers: int = 2
    transformer_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    cutoff: float = 5.0
    use_coordinate_features: bool = True
    soap_num_radial: int = 8
    soap_max_angular: int = 4
    soap_cutoff: float = 5.0
    soap_gaussian_width: float = 0.5
    predict_forces: bool = False


def _validate_attention_heads(hidden_dim: int, num_heads: int) -> None:
    """Ensure the attention head configuration is valid.

    ``torch.nn.MultiheadAttention`` requires that ``hidden_dim`` is divisible
    by ``num_heads``. When users misconfigure this relationship the underlying
    module raises a generic ``AssertionError``. Performing the check here
    surfaces a clear, actionable ``ValueError`` instead so that the offending
    configuration can be corrected before model construction begins.
    """

    if hidden_dim % num_heads == 0:
        return

    valid_heads = [d for d in range(1, hidden_dim + 1) if hidden_dim % d == 0]
    suggestion = ", ".join(str(d) for d in valid_heads)
    raise ValueError(
        "hidden_dim must be divisible by num_heads for multi-head attention "
        f"(got hidden_dim={hidden_dim}, num_heads={num_heads}). "
        f"Choose num_heads from the divisors of hidden_dim: {suggestion}"
    )


class _ResidualGCNLayer(nn.Module):
    """GCN block with residual connection and layer normalization."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(adjacency, features)
        projected = self.linear(aggregated)
        projected = self.dropout(projected)
        return F.relu(self.norm(projected + features))


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with dropout."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        y = self.norm2(x)
        x = x + self.ff(y)
        return x


class _AttentionPooling(nn.Module):
    """Masked attention pooling over atoms."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, fill_value)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class HybridPotential(nn.Module):
    """SOAP-guided GCN + Transformer potential suitable for energy and forces."""

    def __init__(self, config: HybridPotentialConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        _validate_attention_heads(config.hidden_dim, config.num_heads)
        if config.use_coordinate_features:
            soap_config = AtomicSOAPConfig(
                num_radial=config.soap_num_radial,
                max_angular=config.soap_max_angular,
                cutoff=config.soap_cutoff,
                gaussian_width=config.soap_gaussian_width,
            )
            self.descriptor = AtomicSOAPDescriptor(soap_config)
            self.descriptor_projection = nn.Sequential(
                nn.Linear(self.descriptor.num_features, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.descriptor = None
            self.descriptor_projection = None
        self.gcn_layers = nn.ModuleList(
            [_ResidualGCNLayer(config.hidden_dim, config.dropout) for _ in range(config.gcn_layers)]
        )
        self.transformer_layers = nn.ModuleList(
            [
                _TransformerBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.ffn_dim,
                    config.dropout,
                )
                for _ in range(config.transformer_layers)
            ]
        )
        self.pool = _AttentionPooling(config.hidden_dim)
        self.energy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        if config.predict_forces:
            self.force_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 3),
            )
        else:
            self.force_head = None

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        mask_float = mask_bool.float()
        x = self.embedding(node_indices)
        if self.descriptor is not None and self.descriptor_projection is not None:
            descriptor = self.descriptor(positions, adjacency, mask_bool)
            x = x + self.descriptor_projection(descriptor)
        x = x * mask_float.unsqueeze(-1)
        norm_adj = self._normalize_adjacency(adjacency, positions, mask_bool)
        for layer in self.gcn_layers:
            x = layer(norm_adj, x)
        key_padding_mask = ~mask_bool
        for block in self.transformer_layers:
            x = block(x, key_padding_mask=key_padding_mask)
        pooled = self.pool(x, mask_bool)
        energy = self.energy_head(pooled).squeeze(-1)
        forces = None
        if self.force_head is not None:
            forces = self.force_head(x) * mask_float.unsqueeze(-1)
        return PotentialOutput(energy=energy, forces=forces)

    def _normalize_adjacency(
        self, adjacency: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if adjacency is None:
            distances = torch.cdist(positions, positions)
            adjacency = (distances <= self.config.cutoff).to(positions.dtype)
        adjacency = adjacency.to(positions.dtype)
        adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
        eye = torch.eye(adjacency.size(-1), device=adjacency.device, dtype=adjacency.dtype)
        adjacency = adjacency + eye.unsqueeze(0) * mask.unsqueeze(-1)
        degree = adjacency.sum(dim=-1).clamp(min=1e-6)
        inv_sqrt = degree.pow(-0.5) * mask
        norm = adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
        return norm


__all__ = ["HybridPotentialConfig", "HybridPotential"]
