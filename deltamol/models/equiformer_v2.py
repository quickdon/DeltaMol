"""EquiformerV2-inspired equivariant transformer potential.

This compact implementation draws on the reference design from
`EquiformerV2 <https://github.com/atomicarchitects/equiformer_v2>`_, retaining
its radial-basis attention biases and force-aware readout while keeping the
interface aligned with other DeltaMol potentials. The model is intentionally
lightweight so it can run inside unit tests without pulling the full upstream
dependency tree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .potential import PotentialOutput


def _bessel_envelope(distances: torch.Tensor, cutoff: float, dim: int) -> torch.Tensor:
    """Encode pairwise distances using sinusoidal Bessel features.

    The features follow the sinusoidal basis used in the public EquiformerV2
    implementation but omit spherical harmonics, which keeps the embedding
    inexpensive for quick experiments and tests.
    """

    scaled = distances / cutoff
    k = torch.arange(1, dim + 1, device=distances.device, dtype=distances.dtype)
    angles = math.pi * k * scaled.unsqueeze(-1)
    safe_dist = distances.unsqueeze(-1) + 1e-6
    envelope = torch.sin(angles) / safe_dist
    # Suppress contributions outside the cutoff to mirror the radial envelope.
    envelope = envelope * (distances.unsqueeze(-1) <= cutoff).to(distances.dtype)
    return envelope


class _RadialBias(nn.Module):
    def __init__(self, distance_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(distance_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, distance_features: torch.Tensor) -> torch.Tensor:
        # (B, N, N, H)
        return self.mlp(distance_features)


class _EquiformerBlock(nn.Module):
    """Self-attention block with radial biases and gated feedforward."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        distance_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.radial_bias = _RadialBias(distance_dim, hidden_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, distance_features: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        logits = torch.einsum("bihd,bjhd->bhij", q, k) / self.scale
        bias = self.radial_bias(distance_features)  # (B, N, N, H)
        logits = logits + bias.permute(0, 3, 1, 2)

        masked_logits = logits.masked_fill(
            ~edge_mask.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attn = torch.softmax(masked_logits, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, N, -1)

        gated = self.gate(x)
        x = residual + self.dropout(self.out_proj(context * torch.sigmoid(gated)))

        y = self.norm2(x)
        x = x + self.ff(y)
        return x


@dataclass
class EquiformerV2Config:
    """Configuration for :class:`EquiformerV2Potential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    distance_embedding_dim: int = 32
    cutoff: float = 5.0
    dropout: float = 0.1
    predict_forces: bool = False


class EquiformerV2Potential(nn.Module):
    """Lightweight EquiformerV2-style model for molecular energies and forces."""

    def __init__(self, config: EquiformerV2Config) -> None:
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                _EquiformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    distance_dim=config.distance_embedding_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.energy_head = nn.Linear(config.hidden_dim, 1)
        if config.predict_forces:
            self.force_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 3),
            )
        else:
            self.force_head = None

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        x = self.embedding(node_indices) * mask_bool.unsqueeze(-1)

        positions = positions * mask_bool.unsqueeze(-1)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(rel_pos + 1e-9, dim=-1)
        if adjacency is None:
            adjacency = (distances <= self.config.cutoff).to(positions.dtype)
        adjacency = adjacency * mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2)
        edge_mask = adjacency > 0

        # Ensure each valid node participates in attention.
        has_edges = edge_mask.any(dim=-1)
        isolated = mask_bool & ~has_edges
        if isolated.any():
            adjacency = adjacency + torch.diag_embed(isolated.to(adjacency.dtype))
            edge_mask = adjacency > 0

        distance_features = _bessel_envelope(
            distances, cutoff=self.config.cutoff, dim=self.config.distance_embedding_dim
        )
        distance_features = distance_features * adjacency.unsqueeze(-1)

        for block in self.layers:
            x = block(x, distance_features, edge_mask)

        pooled = self.pool(x)
        masked = pooled * mask_bool.unsqueeze(-1)
        summed = masked.sum(dim=1)
        counts = mask_bool.sum(dim=1).clamp(min=1)
        graph_repr = summed / counts.unsqueeze(-1)

        energy = self.energy_head(graph_repr).squeeze(-1)
        forces = None
        if self.force_head is not None:
            forces = self.force_head(pooled) * mask_bool.unsqueeze(-1)
        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["EquiformerV2Config", "EquiformerV2Potential"]
