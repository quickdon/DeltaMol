"""Dataset splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


@dataclass
class DataSplit:
    """Indices describing a train/validation/test partition."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def stratified_split(n_samples: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), *, seed: int | None = 42) -> DataSplit:
    """Return deterministic indices for a dataset partition."""

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(ratios[0] * n_samples)
    n_val = int(ratios[1] * n_samples)
    train = indices[:n_train]
    validation = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    return DataSplit(train=train, validation=validation, test=test)


def subset(sequence: Sequence, indices: Iterable[int]):
    return [sequence[i] for i in indices]
