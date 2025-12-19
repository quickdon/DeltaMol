"""PhysNet-inspired potential energy model.

This module implements a compact adaptation of the PhysNet architecture from
`MMunibas/PhysNet <https://github.com/MMunibas/PhysNet>`_. The model combines
trainable radial basis expansions with attention-weighted message passing and
residual updates to predict total molecular energies and optional analytic
forces.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


class TrainableRadialBasis(nn.Module):
    """Learnable Gaussian radial basis with smooth cutoff."""

    def __init__(self, cutoff: float, num_basis: int):
        super().__init__()
        if num_basis <= 0:
            raise ValueError("num_basis must be positive")
        self.cutoff = float(cutoff)
        mu = torch.linspace(0.0, cutoff, num_basis)
        self.mu = nn.Parameter(mu)
        # Log sigma keeps widths positive while remaining trainable.
        self.log_sigma = nn.Parameter(torch.zeros(num_basis))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        sigma = torch.exp(self.log_sigma) + 1e-6
        diff = distances.unsqueeze(-1) - self.mu
        basis = torch.exp(-0.5 * (diff / sigma) ** 2)
        envelope = 0.5 * (torch.cos(distances.unsqueeze(-1) * math.pi / self.cutoff) + 1.0)
        envelope = envelope * (distances <= self.cutoff).unsqueeze(-1)
        return basis * envelope


class PhysNetBlock(nn.Module):
    """Single PhysNet-style interaction block with attention pooling."""

    def __init__(self, hidden_dim: int, num_basis: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, features: torch.Tensor, rbf: torch.Tensor, edge_mask: torch.Tensor
    ) -> torch.Tensor:
        # features: (B, N, H), rbf: (B, N, N, num_basis), edge_mask: (B, N, N)
        filtered = self.filter_net(rbf)
        neighbour = self.message_net(features).unsqueeze(1)
        messages = filtered * neighbour

        central = features.unsqueeze(2).expand_as(messages)
        attn_logits = self.attention(torch.cat([central, messages], dim=-1)).squeeze(-1)
        neg_inf = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(~edge_mask, neg_inf)
        attn_weights = torch.softmax(attn_logits, dim=2)
        attn_weights = attn_weights * edge_mask
        normaliser = attn_weights.sum(dim=2, keepdim=True).clamp(min=1e-9)
        attn_weights = attn_weights / normaliser

        aggregated = (messages * attn_weights.unsqueeze(-1)).sum(dim=2)
        update = self.update_net(aggregated)
        return features + update


@dataclass
class PhysNetConfig:
    """Configuration for :class:`PhysNetPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_blocks: int = 5
    num_basis: int = 64
    cutoff: float = 6.0
    predict_forces: bool = False


class PhysNetPotential(nn.Module):
    """PhysNet-inspired potential energy model with optional forces."""

    def __init__(self, config: PhysNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)

        self.embedding = nn.Embedding(num_species + 1, config.hidden_dim, padding_idx=0)
        self.radial_basis = TrainableRadialBasis(config.cutoff, config.num_basis)
        self.blocks = nn.ModuleList(
            [PhysNetBlock(config.hidden_dim, config.num_basis) for _ in range(config.num_blocks)]
        )
        self.atomwise = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self._grad_layout_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._register_grad_layout_hooks()

    def _register_grad_layout_hooks(self) -> None:
        """Preserve gradient layout for DDP bucketing on the readout weight."""

        readout_weight = self.atomwise[-1].weight
        target_stride = readout_weight.stride()

        def _fix_layout(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            if grad.stride() != target_stride:
                return grad.contiguous()
            return grad

        if self._grad_layout_hook_handle is not None:
            self._grad_layout_hook_handle.remove()
        self._grad_layout_hook_handle = readout_weight.register_hook(_fix_layout)

    def refresh_grad_layout_hooks(self) -> None:
        """Reinstall layout hooks after DDP attaches its own autograd hooks."""

        self._register_grad_layout_hooks()

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
        needs_forces = self.config.predict_forces

        if needs_forces and not positions.requires_grad:
            positions = positions.clone().detach().requires_grad_(True)

        grad_context = nullcontext()
        if needs_forces and not torch.is_grad_enabled():
            grad_context = torch.enable_grad()

        with grad_context:
            displacement, distances, edge_mask = self._build_geometry(positions, mask_bool, adjacency)
            rbf = self.radial_basis(distances) * edge_mask.unsqueeze(-1)

            features = self.embedding(node_indices)
            features = features * mask_float.unsqueeze(-1)

            for block in self.blocks:
                features = block(features, rbf, edge_mask)

            per_atom = self.atomwise(features).squeeze(-1)
            energy = (per_atom * mask_float).sum(dim=1)

            forces = None
            if needs_forces:
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

        return PotentialOutput(energy=energy, forces=forces)


__all__ = ["PhysNetConfig", "PhysNetPotential"]
