"""LEFTNet-inspired equivariant potential energy model.

This module adapts the reference implementation from
`yuanqidu/LeftNet <https://github.com/yuanqidu/LeftNet>`_ to DeltaMol's dense
batch format. The design couples invariant scalar channels with equivariant
vector channels, uses local edge and node frames to retain orientational
information, and supports analytic force computation via energy gradients. The
implementation keeps the attention-style weighting differentiable even when the
caller wraps the forward pass in ``torch.no_grad`` by re-enabling gradients for
force prediction. Gradient-layout hooks keep the DDP bucket formation stable
when running distributed training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .potential import PotentialOutput


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    return vec / norm


class _RadialBasis(nn.Module):
    """Exponentially spaced radial basis with a smooth cosine cutoff."""

    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = float(cutoff)
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        end_value = torch.exp(torch.scalar_tensor(0.0))
        means = torch.linspace(start_value, end_value, num_radial)
        betas = torch.tensor([(2 / max(num_radial, 1) * (end_value - start_value)) ** -2] * num_radial)
        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled = distances.unsqueeze(-1)
        cutoff = 0.5 * (torch.cos(scaled * math.pi / (self.cutoff + 1e-8)) + 1.0)
        cutoff = cutoff * (distances < self.cutoff).to(distances.dtype).unsqueeze(-1)
        exp_term = torch.exp(-scaled)
        return cutoff * torch.exp(-self.betas * (exp_term - self.means) ** 2)


class _NeighborEmbedding(nn.Module):
    """Aggregate neighbour embeddings weighted by radial filters."""

    def __init__(self, hidden_dim: int, num_species: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_species + 1, hidden_dim, padding_idx=0)

    def forward(
        self,
        node_indices: torch.Tensor,
        radial_hidden: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbour_features = self.embedding(node_indices)  # (B, N, H)
        weighted = radial_hidden * neighbour_features[:, None, :, :]
        weighted = weighted * edge_mask.unsqueeze(-1)
        return weighted.sum(dim=2)


class _SubstructureEncoder(nn.Module):
    """Encode local 3D substructures as vector features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(
        self,
        scalar_features: torch.Tensor,
        displacement: torch.Tensor,
        radial_hidden: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        scalar_features = self.proj(scalar_features)
        displacement = displacement.unsqueeze(-1) * radial_hidden.unsqueeze(-2)
        message = displacement * scalar_features[:, None, :, :].unsqueeze(-2)
        message = message * edge_mask.unsqueeze(-1).unsqueeze(-1)
        return message.sum(dim=2)  # (B, N, 3, H)


class _EquivariantMessagePassing(nn.Module):
    """Equivariant message passing with distance-aware filters."""

    def __init__(self, hidden_dim: int, num_radial: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inv_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + num_radial, hidden_dim * 3),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
        )
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_dim * 3)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_dim)

    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        radial_emb: torch.Tensor,
        edge_weights: torch.Tensor,
        displacement: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xh = self.x_proj(scalar_features).view(scalar_features.size(0), scalar_features.size(1), 3, self.hidden_dim)
        rbfh = self.rbf_proj(radial_emb).view(radial_emb.size(0), radial_emb.size(1), radial_emb.size(2), 3, self.hidden_dim)
        weight = self.inv_proj(edge_weights).view_as(rbfh)
        rbfh = rbfh * weight

        xh = xh[:, None, :, :, :]  # (B, 1, N, 3, H)
        vector_features = vector_features[:, None, :, :, :]  # (B, 1, N, 3, H)

        weighted = xh * rbfh  # (B, N, N, 3, H)
        scalar_part, scaled_vec, vec_bias = torch.unbind(weighted, dim=3)
        scaled_vec = scaled_vec * self.inv_sqrt_3

        disp = displacement.unsqueeze(-1)  # (B, N, N, 3, 1)
        vec_update = vector_features * scaled_vec.unsqueeze(3) + vec_bias.unsqueeze(3) * disp
        vec_update = vec_update * self.inv_sqrt_h

        scalar_part = scalar_part * edge_mask.unsqueeze(-1)
        vec_update = vec_update * edge_mask.unsqueeze(-1).unsqueeze(-1)

        scalar_out = scalar_part.sum(dim=2)
        vec_out = vec_update.sum(dim=2)
        return scalar_out, vec_out


class _FrameTransitionEncoder(nn.Module):
    """Frame transition encoding to mix scalar and vector channels."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.equi_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.xequi_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_dim)

    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        node_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vec = self.equi_proj(vector_features)
        vec1, vec2 = torch.split(vec, self.hidden_dim, dim=-1)

        scalar_norm = vec1.norm(dim=-2)

        vec_dot = (vec1 * vec2).sum(dim=2) * self.inv_sqrt_h

        xvec = self.xequi_proj(torch.cat([scalar_features, scalar_norm], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(xvec, self.hidden_dim, dim=-1)

        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(2) * vec2
        return dx, dvec


class _LeftInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_radial: int):
        super().__init__()
        self.message = _EquivariantMessagePassing(hidden_dim, num_radial)
        self.frame_encoder = _FrameTransitionEncoder(hidden_dim)

    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        radial_emb: torch.Tensor,
        edge_weights: torch.Tensor,
        displacement: torch.Tensor,
        node_frame: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ds, dv = self.message(
            scalar_features,
            vector_features,
            radial_emb,
            edge_weights,
            displacement,
            edge_mask,
        )
        scalar_features = scalar_features + ds
        vector_features = vector_features + dv
        ds, dv = self.frame_encoder(scalar_features, vector_features, node_frame)
        return scalar_features + ds, vector_features + dv


@dataclass
class LeftNetConfig:
    """Configuration for :class:`LeftNetPotential`."""

    species: tuple[int, ...]
    hidden_dim: int = 128
    num_layers: int = 4
    num_radial: int = 32
    cutoff: float = 5.0
    predict_forces: bool = False


class LeftNetPotential(nn.Module):
    """LEFTNet-style potential energy model."""

    def __init__(self, config: LeftNetConfig):
        super().__init__()
        self.config = config
        num_species = len(config.species)

        self.radial_basis = _RadialBasis(config.num_radial, config.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(config.num_radial, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.neighbour_emb = _NeighborEmbedding(config.hidden_dim, num_species)
        self.substructure = _SubstructureEncoder(config.hidden_dim)
        self.frame_mlp = nn.Sequential(
            nn.Linear(3, max(config.hidden_dim // 4, 1)),
            nn.SiLU(inplace=True),
            nn.Linear(max(config.hidden_dim // 4, 1), 1),
        )
        self.layers = nn.ModuleList(
            [_LeftInteractionBlock(config.hidden_dim, config.num_radial) for _ in range(config.num_layers)]
        )
        self.readout = nn.Linear(config.hidden_dim, 1)
        self.embedding = self.neighbour_emb.embedding
        self._grad_layout_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._register_grad_layout_hook()

    def _register_grad_layout_hook(self) -> None:
        """Keep readout gradients contiguous for stable DDP bucket formation."""

        weight = self.readout.weight
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
        """Re-register gradient layout hooks after DDP wraps the module."""

        self._register_grad_layout_hook()

    def _build_geometry(
        self,
        positions: torch.Tensor,
        adjacency: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_bool = mask.bool()
        displacement = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(displacement, dim=-1)
        if adjacency is None:
            adjacency = (distances <= self.config.cutoff).to(positions.dtype)
        edge_mask = adjacency.bool() & mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)
        eye = torch.eye(edge_mask.size(1), device=edge_mask.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask & ~eye

        displacement = displacement * edge_mask.unsqueeze(-1)
        safe_distances = distances.clamp(min=1e-6)
        radial_emb = self.radial_basis(safe_distances) * edge_mask.unsqueeze(-1)
        radial_hidden = self.radial_lin(radial_emb) * edge_mask.unsqueeze(-1)

        edge_dir = _normalize(displacement + (~edge_mask).unsqueeze(-1) * 0.0)
        cross = _normalize(
            torch.cross(positions.unsqueeze(2), positions.unsqueeze(1), dim=-1)
            + (~edge_mask).unsqueeze(-1) * 0.0
        )
        vertical = _normalize(torch.cross(edge_dir, cross, dim=-1) + (~edge_mask).unsqueeze(-1) * 0.0)
        edge_frame = torch.stack([edge_dir, cross, vertical], dim=-2)  # (B, N, N, 3, 3)

        neighbour_sum = (positions.unsqueeze(1) * edge_mask.unsqueeze(-1)).sum(dim=2)
        counts = edge_mask.sum(dim=2).clamp(min=1).unsqueeze(-1)
        mean_neighbor = neighbour_sum / counts
        node_dir = _normalize(positions - mean_neighbor)
        node_cross = _normalize(torch.cross(positions, mean_neighbor, dim=-1))
        node_vertical = _normalize(torch.cross(node_dir, node_cross, dim=-1))
        node_frame = torch.stack([node_dir, node_cross, node_vertical], dim=-2)  # (B, N, 3, 3)

        return radial_emb, radial_hidden, edge_frame, node_frame

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
            positions = positions.detach().clone().requires_grad_(True)

        def _compute_energy(current_positions: torch.Tensor) -> torch.Tensor:
            radial_emb, radial_hidden, edge_frame, node_frame = self._build_geometry(
                current_positions, adjacency, mask_bool
            )
            displacement = current_positions.unsqueeze(2) - current_positions.unsqueeze(1)
            displacement = displacement * (radial_emb.sum(dim=-1, keepdim=True) > 0).to(displacement.dtype)
            soft_cutoff = (radial_emb.sum(dim=-1) > 0).to(current_positions.dtype)

            scalar = self.neighbour_emb(node_indices, radial_hidden, soft_cutoff)
            scalar = scalar * mask_float.unsqueeze(-1)
            vector = torch.zeros(
                (*scalar.shape[:2], 3, scalar.shape[-1]),
                device=scalar.device,
                dtype=scalar.dtype,
            )

            substructure = self.substructure(scalar, displacement, radial_hidden, soft_cutoff)
            scalrization1 = torch.einsum("bidh,bijdk->bijkh", substructure, edge_frame)
            scalrization2 = torch.einsum("bjdh,bijdk->bijkh", substructure, edge_frame)
            scalrization1 = torch.stack(
                [
                    scalrization1[:, :, :, 0, :],
                    scalrization1[:, :, :, 1, :].abs(),
                    scalrization1[:, :, :, 2, :],
                ],
                dim=3,
            )
            scalrization2 = torch.stack(
                [
                    scalrization2[:, :, :, 0, :],
                    scalrization2[:, :, :, 1, :].abs(),
                    scalrization2[:, :, :, 2, :],
                ],
                dim=3,
            )

            scalar3 = self.frame_mlp(scalrization1.permute(0, 1, 2, 4, 3)).squeeze(-1) + scalrization1[:, :, :, 0, :]
            scalar4 = self.frame_mlp(scalrization2.permute(0, 1, 2, 4, 3)).squeeze(-1) + scalrization2[:, :, :, 0, :]
            edge_weights = torch.cat([scalar3, scalar4, radial_hidden, radial_emb], dim=-1)
            edge_weights = edge_weights * soft_cutoff.unsqueeze(-1)

            displacement = _normalize(displacement)
            displacement = displacement * soft_cutoff.unsqueeze(-1)

            for layer in self.layers:
                scalar, vector = layer(
                    scalar,
                    vector,
                    radial_emb,
                    edge_weights,
                    displacement,
                    node_frame,
                    soft_cutoff.bool(),
                )
            per_atom = self.readout(scalar).squeeze(-1) * mask_float
            return per_atom.sum(dim=1)

        forces: torch.Tensor | None = None
        if needs_forces:
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


__all__ = ["LeftNetConfig", "LeftNetPotential"]
