"""Differentiable SOAP-inspired features implemented with PyTorch."""
from __future__ import annotations

import math
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
    """Configuration for :class:`AtomicSOAPDescriptor`.

    The parameters mirror the original SOAP descriptor in `dscribe` with a
    simplified radial basis. ``max_angular`` controls the maximum spherical
    harmonic degree and ``num_radial`` the number of Gaussian radial basis
    functions. ``cutoff`` is the radial cutoff, ``gaussian_width`` sets the
    width of the Gaussian basis, and ``include_self`` determines whether the
    central atom contributes to its own density.
    """

    num_radial: int = 8
    max_angular: int = 4
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

    @property
    def num_features(self) -> int:
        radial_pairs = self.config.num_radial * (self.config.num_radial + 1) // 2
        return (self.config.max_angular + 1) * radial_pairs

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
            neighbour_mask = adjacency > 0
            distances = torch.where(neighbour_mask, distances, torch.ones_like(distances))
            centers = self.centers.to(dtype=positions.dtype)
            diff = distances.unsqueeze(-1) - centers
            radial = torch.exp(-0.5 * (diff / self.config.gaussian_width) ** 2)
            complex_dtype = torch.cdouble if radial.dtype == torch.float64 else torch.cfloat
            radial_complex = radial.to(dtype=complex_dtype)

            # Angular part via real spherical harmonics. torch.special returns
            # complex values; we keep them until forming the power spectrum.
            x, y, z = displacement.unbind(-1)
            theta = torch.acos((z / distances).clamp(-1.0, 1.0))
            phi = torch.atan2(torch.where(neighbour_mask, y, torch.zeros_like(y)), torch.where(neighbour_mask, x, torch.ones_like(x)))

            def _double_factorial(n: int) -> float:
                if n <= 0:
                    return 1.0
                result = 1.0
                for k in range(n, 0, -2):
                    result *= float(k)
                return result

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            def _associated_legendre(l: int, m: int) -> torch.Tensor:
                # Compute P_l^m(cos(theta)) using the standard recursive
                # definition. Results broadcast over (batch, i, j).
                if m == l == 0:
                    return torch.ones_like(cos_theta)

                # P_m^m
                sign = -1.0 if (m % 2 == 1) else 1.0
                p_mm = sign * _double_factorial(2 * m - 1) * (sin_theta ** m)
                if l == m:
                    return p_mm

                # P_{m+1}^m
                p_m1m = (2 * m + 1) * cos_theta * p_mm
                if l == m + 1:
                    return p_m1m

                p_lm_minus2 = p_mm
                p_lm_minus1 = p_m1m
                p_lm = p_lm_minus1
                for ell in range(m + 2, l + 1):
                    p_lm = ((2 * ell - 1) * cos_theta * p_lm_minus1 - (ell + m - 1) * p_lm_minus2) / (
                        ell - m
                    )
                    p_lm_minus2, p_lm_minus1 = p_lm_minus1, p_lm
                return p_lm

            # Build c_{nlm} coefficients following dscribe's SOAP definition.
            coeffs = []
            for l in range(self.config.max_angular + 1):
                harmonics = []
                for m in range(-l, l + 1):
                    abs_m = abs(m)
                    norm = math.sqrt(
                        (2 * l + 1)
                        / (4 * math.pi)
                        * math.exp(math.lgamma(l - abs_m + 1) - math.lgamma(l + abs_m + 1))
                    )
                    if m < 0:
                        # Relation Y_l^{-m} = (-1)^m * conj(Y_l^m)
                        p_lm = _associated_legendre(l, abs_m)
                        y_pos_m = norm * p_lm * torch.exp(1j * float(abs_m) * phi)
                        Y_lm = ((-1) ** abs_m) * torch.conj(y_pos_m)
                    else:
                        p_lm = _associated_legendre(l, m)
                        Y_lm = norm * p_lm * torch.exp(1j * float(m) * phi)

                    Y_lm = Y_lm.to(dtype=complex_dtype, device=radial.device)
                    harmonics.append(Y_lm)
                # (batch, i, j, m)
                harmonics = torch.stack(harmonics, dim=-1)
                # (batch, i, j, n, m)
                c_nlm = radial_complex.unsqueeze(-1) * harmonics.unsqueeze(-2)
                c_nlm = c_nlm * adjacency.unsqueeze(-1).unsqueeze(-1)
                # Sum over neighbours j
                c_nlm = c_nlm.sum(dim=2)
                coeffs.append(c_nlm)

            # Power spectrum P_{n n' l} = sum_m c_{n l m} c_{n' l m}^*
            power_spectra = []
            for l, c_l in enumerate(coeffs):
                # c_l: (batch, i, n, m)
                c_l_conj = torch.conj(c_l)
                for n in range(self.config.num_radial):
                    for n2 in range(n, self.config.num_radial):
                        prod = c_l[:, :, n, :] * c_l_conj[:, :, n2, :]
                        spectrum = prod.sum(dim=-1).real
                        power_spectra.append(spectrum)

            features = torch.stack(power_spectra, dim=-1)
            features = features * mask_bool.unsqueeze(-1)
        return features


__all__ = ["AtomicSOAPConfig", "AtomicSOAPDescriptor"]
