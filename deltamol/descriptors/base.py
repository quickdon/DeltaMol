"""Descriptor abstraction layer for DeltaMol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from ase import Atoms
except ImportError:  # pragma: no cover - soft dependency
    Atoms = "Atoms"  # type: ignore


class Descriptor(ABC):
    """Base interface for all descriptor generators.

    Concrete implementations focus on transforming raw atomic numbers and
    coordinates into machine-learning ready representations.
    """

    name: str

    def __init__(self, species: Iterable[int]):
        self.species = tuple(sorted(set(int(s) for s in species)))

    @abstractmethod
    def create(self, atoms: "Atoms") -> np.ndarray:
        """Return the descriptor matrix for a given :class:`ase.Atoms`."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(species={self.species!r})"


def infer_species(atomic_numbers: Sequence[Sequence[int]]) -> Sequence[int]:
    """Infer the unique atomic species present in the dataset."""

    species: set[int] = set()
    for numbers in atomic_numbers:
        species.update(int(n) for n in numbers)
    return tuple(sorted(species))
