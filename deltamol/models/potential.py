"""Shared utilities for potential energy models."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PotentialOutput:
    """Container holding predictions from a potential model."""

    energy: torch.Tensor
    forces: torch.Tensor | None = None


__all__ = ["PotentialOutput"]

