"""FCHL19 descriptor interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


from .base import Descriptor, infer_species

try:  # pragma: no cover - optional dependency
    from qmllib.representations import generate_fchl19
except ImportError as exc:  # pragma: no cover
    generate_fchl19 = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class FCHL19Config:
    species: Sequence[int]


class FCHL19Descriptor(Descriptor):
    name = "fchl19"

    def __init__(self, config: FCHL19Config | None = None, *, species: Iterable[int] | None = None):
        if config is None and species is None:
            raise ValueError("Either config or species must be provided")
        if config is None:
            config = FCHL19Config(species=tuple(sorted(set(int(s) for s in species or ()))) )
        if generate_fchl19 is None:  # pragma: no cover
            raise ImportError("qmllib is required for FCHL19 descriptors") from _IMPORT_ERROR
        super().__init__(config.species)
        self.config = config

    def create(self, atoms) -> np.ndarray:  # pragma: no cover - requires qmllib
        numbers = atoms.get_atomic_numbers()
        coords = atoms.get_positions()
        return generate_fchl19(numbers, coords, elements=self.config.species)


def build_fchl19_descriptor(atomic_numbers: Sequence[Sequence[int]]) -> FCHL19Descriptor:
    species = infer_species(atomic_numbers)
    return FCHL19Descriptor(FCHL19Config(species=species))
