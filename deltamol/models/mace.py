"""MACE-inspired message passing potential energy model.

This compact implementation follows the design of the `ACEsuit/mace
<https://github.com/ACEsuit/mace>`_ project, mixing radial Bessel embeddings
with distance-aware attention to capture many-body interactions. The code is
intentionally lightweight so it can run inside the test suite without pulling
the full upstream dependency tree while retaining the same energy/force
interface as other DeltaMol potentials.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


class _BesselEmbedding(nn.Module):
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


class _MACEInteraction(nn.Module):
    """Distance-aware self-attention block with radial gating."""

    def __init__(self, hidden_dim: int, num_heads: int, num_radial: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.distance_mlp = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_bias = nn.Sequential(
            nn.Linear(num_radial, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
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
        bias = self.attn_bias(distance_features)  # (B, N, N, H)
        logits = logits + bias.permute(0, 3, 1, 2)

        masked_logits = logits.masked_fill(
            ~edge_mask.unsqueeze(1), torch.finfo(logits.dtype).min
        )
        attn = torch.softmax(masked_logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        distance_filters = self.distance_mlp(distance_features)  # (B, N, N, H)
        distance_filters = distance_filters.view(B, N, N, self.num_heads, self.head_dim)
        distance_filters = distance_filters * edge_mask.unsqueeze(-1).unsqueeze(-1)

        edge_values = v.unsqueeze(2) * (1.0 + distance_filters)  # (B, N, N, H, D)
        edge_values = edge_values.permute(0, 3, 1, 2, 4)  # (B, H, N, N, D)
        context = torch.einsum("bhij,bhijd->bihd", attn, edge_values)
        context = context.reshape(B, N, self.hidden_dim)

        x = residual + self.dropout(self.out_proj(context))
        x = x + self.ff(self.norm2(x))
        return x


@dataclass
class MACEConfig:
    """Configuration for :class:`MACEPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 4
    num_radial: int = 16
    num_heads: int = 4
    cutoff: float = 5.0
    dropout: float = 0.1
    predict_forces: bool = False


class MACEPotential(nn.Module):
    """MACE-style potential with analytic force support."""

    def __init__(self, config: MACEConfig) -> None:
        super().__init__()
        self.config = config
        num_species = len(config.species)
        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.radial_basis = _BesselEmbedding(config.num_radial, config.cutoff)
        self.layers = nn.ModuleList(
            [
                _MACEInteraction(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_radial=config.num_radial,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self._grad_layout_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._register_grad_layout_hook()

    def _register_grad_layout_hook(self) -> None:
        """Keep readout gradients contiguous to appease DDP bucket formation."""

        weight = self.readout[-1].weight
        target_stride = weight.stride()

        def _fix_layout(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            if grad.stride() != target_stride:
                return grad.contiguous()
            return grad

        if self._grad_layout_hook_handle is not None:
            self._grad_layout_hook_handle.remove()
        self._grad_layout_hook_handle = weight.register_hook(_fix_layout)

    def refresh_grad_layout_hooks(self) -> None:
        """Re-register layout hooks after DDP installs its own autograd hooks."""

        self._register_grad_layout_hook()

    def _build_geometry(
        self,
        positions: torch.Tensor,
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_bool = mask.bool()
        positions = positions * mask_bool.unsqueeze(-1)
        distances = torch.linalg.norm(positions.unsqueeze(2) - positions.unsqueeze(1), dim=-1)
        if adjacency is None:
            adjacency = (distances <= self.config.cutoff).to(positions.dtype)
        adjacency = adjacency * mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2)
        edge_mask = adjacency > 0
        eye = torch.eye(edge_mask.size(1), device=edge_mask.device, dtype=edge_mask.dtype).bool()
        edge_mask = edge_mask & ~eye.unsqueeze(0)
        has_edges = edge_mask.any(dim=-1)
        isolated = mask_bool & ~has_edges
        if isolated.any():
            edge_mask = edge_mask | torch.diag_embed(isolated)
        distances = distances.clamp(min=1e-6)
        distance_features = self.radial_basis(distances) * edge_mask.unsqueeze(-1).to(distances.dtype)
        return distance_features, edge_mask

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        mask_bool = mask.bool()
        mask_float = mask_bool.float()
        needs_force_grad = self.config.predict_forces

        if needs_force_grad and not positions.requires_grad:
            positions = positions.detach().clone().requires_grad_(True)

        def _compute_energy(current_positions: torch.Tensor) -> torch.Tensor:
            distance_features, edge_mask = self._build_geometry(
                current_positions, adjacency, mask_bool
            )
            x = self.embedding(node_indices) * mask_float.unsqueeze(-1)
            for layer in self.layers:
                x = layer(x, distance_features, edge_mask)
            pooled = self.readout(x) * mask_float.unsqueeze(-1)
            summed = pooled.sum(dim=1)
            counts = mask_float.sum(dim=1).clamp(min=1).unsqueeze(-1)
            graph_repr = summed / counts
            return graph_repr.squeeze(-1)

        forces: torch.Tensor | None = None
        if needs_force_grad:
            with torch.enable_grad():
                energy = _compute_energy(positions)
                grads = torch.autograd.grad(
                    energy.sum(),
                    positions,
                    create_graph=self.training,
                    retain_graph=self.training,
                    allow_unused=True,
                )[0]
            if grads is None:
                grads = torch.zeros_like(positions)
            forces = -grads * mask_float.unsqueeze(-1)
        else:
            with torch.set_grad_enabled(torch.is_grad_enabled()):
                energy = _compute_energy(positions)
        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["MACEConfig", "MACEPotential"]
