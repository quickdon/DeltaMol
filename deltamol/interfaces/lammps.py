"""LAMMPS interface helpers for DeltaMol potentials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .core import PotentialCalculator, PotentialPrediction


@dataclass
class LammpsTypeMap:
    """Map LAMMPS atom types (1-based) to atomic numbers."""

    atomic_numbers: Sequence[int]

    def resolve(self, types: Iterable[int]) -> list[int]:
        mapping = list(self.atomic_numbers)
        return [mapping[int(t) - 1] for t in types]


class LammpsPairPotential:
    """Minimal pair_style python adapter for DeltaMol potentials."""

    def __init__(
        self,
        calculator: PotentialCalculator,
        *,
        type_map: LammpsTypeMap,
    ) -> None:
        self.calculator = calculator
        self.type_map = type_map

    def compute(self, timestep, nlocal, x, f, types, *args, **kwargs) -> float:
        """LAMMPS callback that fills forces and returns total energy."""

        positions = np.asarray(x[:nlocal], dtype=float)
        atom_numbers = self.type_map.resolve(types[:nlocal])
        prediction = self.calculator.predict(atom_numbers, positions, require_forces=True)
        f[:nlocal, :] = prediction.forces
        return float(prediction.energy)

    def compute_single(self, positions: np.ndarray, types: Iterable[int]) -> PotentialPrediction:
        """Utility method for testing without LAMMPS."""

        atom_numbers = self.type_map.resolve(types)
        return self.calculator.predict(atom_numbers, positions, require_forces=True)


__all__ = ["LammpsPairPotential", "LammpsTypeMap"]
