"""Differentiable SOAP-inspired features implemented with PyTorch."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _descriptor_autocast(device: torch.device) -> torch.autocast:
    """Return an autocast context matching the input device when enabled."""

    enabled = torch.is_autocast_enabled()
    if not enabled or device.type not in {"cuda", "cpu"}:
        return nullcontext()  # type: ignore[return-value]

    try:
        if device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        else:
            dtype = torch.get_autocast_cpu_dtype()
    except AttributeError:  # pragma: no cover - older torch fallback
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    try:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    except TypeError:  # pragma: no cover - torch<2.0 fallback
        if device.type == "cuda":
            from torch.cuda.amp import autocast  # type: ignore

            return autocast(enabled=enabled)
        return nullcontext()  # type: ignore[return-value]


@dataclass
class AtomicSOAPConfig:
    """Configuration for :class:`AtomicSOAPDescriptor`."""

    num_radial: int = 8
    cutoff: float = 5.0
    gaussian_width: float = 0.5
    include_self: bool = False
    eps: float = 1e-8


class AtomicSOAPDescriptor(nn.Module):
    """Lightweight, differentiable SOAP-like descriptor.

    The descriptor expands interatomic distances onto Gaussian radial basis
    functions and aggregates neighbour contributions per atom. All operations
    are differentiable with respect to ``positions``.
    """

    def __init__(self, config: Optional[AtomicSOAPConfig] = None):
        super().__init__()
        self.config = config or AtomicSOAPConfig()
        centers = torch.linspace(0.0, float(self.config.cutoff), self.config.num_radial)
        self.register_buffer("centers", centers)

    def forward(
        self,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-atom SOAP-like features.

        Args:
            positions: Tensor of shape ``(batch, num_atoms, 3)``.
            adjacency: Tensor of shape ``(batch, num_atoms, num_atoms)``. Values
                are used as neighbour weights; typically 0/1.
            mask: Boolean or float tensor of shape ``(batch, num_atoms)``.

        Returns:
            Tensor of shape ``(batch, num_atoms, num_features)`` containing
            radial density features for each atom.
        """

        mask_bool = mask.bool()
        batch, num_atoms, _ = positions.shape
        adjacency = adjacency * mask_bool.unsqueeze(1) * mask_bool.unsqueeze(2)
        if self.config.include_self:
            eye = torch.eye(num_atoms, device=positions.device, dtype=adjacency.dtype)
            adjacency = adjacency + eye.unsqueeze(0) * mask_bool.unsqueeze(-1)

        with _descriptor_autocast(positions.device):
            pos_i = positions.unsqueeze(2)
            pos_j = positions.unsqueeze(1)
            displacement = pos_i - pos_j
            distances = torch.linalg.norm(displacement, dim=-1).clamp(min=self.config.eps)
            centers = self.centers.to(dtype=positions.dtype)
            diff = distances.unsqueeze(-1) - centers
            radial = torch.exp(-0.5 * (diff / self.config.gaussian_width) ** 2)
            weighted = radial * adjacency.unsqueeze(-1)
            features = weighted.sum(dim=2)
            features = features * mask_bool.unsqueeze(-1)
        return features


__all__ = ["AtomicSOAPConfig", "AtomicSOAPDescriptor"]
