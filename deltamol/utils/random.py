"""Randomness helpers for reproducible experiments."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int], *, rank: int = 0) -> Optional[torch.Generator]:
    """Seed Python, NumPy, and Torch RNGs.

    Parameters
    ----------
    seed:
        Base seed value. When ``None`` the RNG state is left untouched.
    rank:
        Distributed rank offset so every worker receives a distinct seed while
        preserving determinism across launches.
    """

    if seed is None:
        return None
    base_seed = int(seed) + int(rank)
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():  # pragma: no cover - dependent on hardware
        torch.cuda.manual_seed_all(base_seed)
    try:  # pragma: no cover - backend availability is platform specific
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(base_seed))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    return generator


def seed_worker(worker_id: int) -> None:  # pragma: no cover - executed in worker processes
    """Initialise worker RNGs using PyTorch's deterministic seed propagation."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = ["seed_everything", "seed_worker"]

