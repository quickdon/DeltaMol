"""LiteFusion potential energy model.

This lightweight architecture blends directional edge features, mixed radial
embeddings, and distance-aware attention into a compact potential suitable for
energy and force prediction. The design fuses ideas from radial basis
expansions, directional message passing, and attention-style aggregation while
staying lightweight for experimentation.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


class BesselRadialBasis(nn.Module):
    """Expand distances with a sinusoidal Bessel basis and smooth cutoff."""

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        freqs = torch.arange(1, num_radial + 1, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled = distances / (self.cutoff + 1e-8)
        safe_dist = distances.clamp(min=1e-6)
        angles = math.pi * scaled.unsqueeze(-1) * self.freqs
        basis = torch.sin(angles) / safe_dist.unsqueeze(-1)
        cutoff_envelope = 0.5 * (torch.cos(math.pi * scaled) + 1.0)
        cutoff_envelope = cutoff_envelope * (distances <= self.cutoff).to(distances.dtype)
        return basis * cutoff_envelope.unsqueeze(-1)


class GaussianRadialBasis(nn.Module):
    """Expand distances with Gaussian basis functions."""

    def __init__(self, cutoff: float, num_gaussians: int):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_gaussians)
        self.register_buffer("centers", centers)
        self.width = cutoff / max(num_gaussians, 1)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / (self.width + 1e-8)) ** 2)


class LiteFusionBlock(nn.Module):
    """Distance-aware attention block with directional gating."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        rbf_dim: int,
        direction_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn_bias = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(rbf_dim + direction_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        radial_features: torch.Tensor,
        direction_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = features.shape
        residual = features
        features = self.norm1(features)

        qkv = self.qkv(features)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        logits = torch.einsum("bihd,bjhd->bhij", q, k) / self.scale
        bias = self.attn_bias(radial_features)
        logits = logits + bias.permute(0, 3, 1, 2)

        masked_logits = logits.masked_fill(
            ~edge_mask.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attn = torch.softmax(masked_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        edge_inputs = torch.cat([radial_features, direction_features], dim=-1)
        edge_update = self.edge_mlp(edge_inputs)
        edge_update = edge_update.view(B, N, N, self.num_heads, self.head_dim)
        edge_update = edge_update * edge_mask.unsqueeze(-1).unsqueeze(-1)

        edge_values = v.unsqueeze(2) * (1.0 + edge_update)
        edge_values = edge_values.permute(0, 3, 1, 2, 4)
        context = torch.einsum("bhij,bhijd->bihd", attn, edge_values)
        context = context.reshape(B, N, self.hidden_dim)

        features = residual + self.dropout(self.out_proj(context))
        features = features + self.ff(self.norm2(features))
        return features


@dataclass
class LiteFusionConfig:
    """Configuration for :class:`LiteFusionPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_blocks: int = 3
    num_heads: int = 4
    num_radial: int = 6
    num_gaussians: int = 6
    num_spherical: int = 4
    rbf_dim: int = 32
    cutoff: float = 5.0
    dropout: float = 0.1
    predict_forces: bool = False


class LiteFusionPotential(nn.Module):
    """LiteFusion potential with radial fusion and directional attention."""

    def __init__(self, config: LiteFusionConfig) -> None:
        super().__init__()
        self.config = config
        num_species = len(config.species)

        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.bessel_basis = BesselRadialBasis(config.num_radial, config.cutoff)
        self.gaussian_basis = GaussianRadialBasis(config.cutoff, config.num_gaussians)
        self.radial_fusion = nn.Sequential(
            nn.Linear(config.num_radial + config.num_gaussians, config.rbf_dim),
            nn.SiLU(),
            nn.Linear(config.rbf_dim, config.rbf_dim),
        )
        self.direction_proj = nn.Linear(3, config.num_spherical)

        self.blocks = nn.ModuleList(
            [
                LiteFusionBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    rbf_dim=config.rbf_dim,
                    direction_dim=config.num_spherical,
                    dropout=config.dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _build_geometry(
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
        has_edges = edge_mask.any(dim=-1)
        isolated = mask_bool & ~has_edges
        if isolated.any():
            edge_mask = edge_mask | torch.diag_embed(isolated)
        distances = distances.clamp(min=1e-6)
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
            displacement, distances, edge_mask = self._build_geometry(positions, mask_bool, adjacency)

            direction = displacement / (distances.unsqueeze(-1) + 1e-8)
            direction = direction * edge_mask.unsqueeze(-1)
            direction_features = torch.tanh(self.direction_proj(direction))

            bessel = self.bessel_basis(distances)
            gaussian = self.gaussian_basis(distances)
            radial_raw = torch.cat([bessel, gaussian], dim=-1)
            radial_features = self.radial_fusion(radial_raw)
            radial_features = radial_features * edge_mask.unsqueeze(-1)

            features = self.embedding(node_indices) * mask_float.unsqueeze(-1)
            for block in self.blocks:
                features = block(features, radial_features, direction_features, edge_mask)

            per_atom_energy = self.readout(features).squeeze(-1) * mask_float
            energy = per_atom_energy.sum(dim=1)

            forces = None
            if self.config.predict_forces:
                forces = -torch.autograd.grad(energy.sum(), positions, create_graph=self.training)[0]
                forces = forces * mask_float.unsqueeze(-1)

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["LiteFusionConfig", "LiteFusionPotential"]
