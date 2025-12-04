"""SE(3)-Transformer-inspired potential energy model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .potential import PotentialOutput


@dataclass
class SE3TransformerConfig:
    """Configuration for :class:`SE3TransformerPotential`."""

    species: Tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    ffn_dim: int = 256
    distance_embedding_dim: int = 32
    dropout: float = 0.1
    cutoff: float = 5.0
    predict_forces: bool = False


def _make_distance_expansion(cutoff: float, dim: int, device: torch.device) -> torch.Tensor:
    centers = torch.linspace(0.0, cutoff, dim, device=device)
    return centers


def _expand_distances(distances: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    width = (centers[1] - centers[0]).clamp(min=1e-6)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-(diff**2) / (width**2))


class _SE3AttentionBlock(nn.Module):
    """Self-attention block with distance-aware biases and feedforward network."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        distance_embedding_dim: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.distance_proj = nn.Linear(distance_embedding_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        distance_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        scores = torch.einsum("bihd,bjhd->bhij", q, k) / self.scale
        bias = self.distance_proj(distance_features)  # (B, N, N, H)
        scores = scores + bias.permute(0, 3, 1, 2)

        masked_scores = scores.masked_fill(~edge_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attn = torch.softmax(masked_scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.einsum("bhij,bjhd->bihd", attn, v)
        context = context.reshape(B, N, -1)
        x = residual + self.dropout(self.out_proj(context))

        y = self.norm2(x)
        x = x + self.ff(y)
        return x


class _AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        fill = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, fill)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class SE3TransformerPotential(nn.Module):
    """Lightweight SE(3)-Transformer style model for energies and forces."""

    def __init__(self, config: SE3TransformerConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                _SE3AttentionBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.ffn_dim,
                    config.distance_embedding_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.pool = _AttentionPool(config.hidden_dim)
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
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        x = self.embedding(node_indices)
        x = x * mask_bool.unsqueeze(-1)

        positions = positions * mask_bool.unsqueeze(-1)
        rel_pos = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(rel_pos + 1e-9, dim=-1)
        if adjacency is None:
            adjacency = (distances <= self.config.cutoff).to(positions.dtype)
        adjacency = adjacency * mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2)
        edge_mask = adjacency > 0

        # Ensure every active node has at least one edge to avoid NaNs from softmax.
        has_neighbors = edge_mask.any(dim=-1)
        isolated_nodes = mask_bool & ~has_neighbors
        if isolated_nodes.any():
            adjacency = adjacency + torch.diag_embed(isolated_nodes.to(adjacency.dtype))
            edge_mask = adjacency > 0

        centers = _make_distance_expansion(self.config.cutoff, self.config.distance_embedding_dim, positions.device)
        distance_features = _expand_distances(distances, centers)
        distance_features = distance_features * adjacency.unsqueeze(-1)

        for block in self.layers:
            x = block(x, distance_features, edge_mask)

        pooled = self.pool(x, mask_bool)
        energy = self.energy_head(pooled).squeeze(-1)
        forces = None
        if self.force_head is not None:
            forces = self.force_head(x) * mask_bool.unsqueeze(-1)
        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["SE3TransformerConfig", "SE3TransformerPotential"]
