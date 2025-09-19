"""SLATM descriptor wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


from .base import Descriptor, infer_species

try:  # pragma: no cover - optional dependency
    from qmllib.representations import generate_slatm
except ImportError as exc:  # pragma: no cover
    generate_slatm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class SLATMConfig:
    species: Sequence[int]


class SLATMDescriptor(Descriptor):
    name = "slatm"

    def __init__(self, config: SLATMConfig | None = None, *, species: Iterable[int] | None = None):
        if config is None and species is None:
            raise ValueError("Either config or species must be provided")
        if config is None:
            config = SLATMConfig(species=tuple(sorted(set(int(s) for s in species or ()))) )
        if generate_slatm is None:  # pragma: no cover
            raise ImportError("qmllib is required for SLATM descriptors") from _IMPORT_ERROR
        super().__init__(config.species)
        self.config = config

    def create(self, atoms) -> np.ndarray:  # pragma: no cover - requires qmllib
        numbers = atoms.get_atomic_numbers()
        coords = atoms.get_positions()
        return generate_slatm(numbers, coords, elements=self.config.species)


def build_slatm_descriptor(atomic_numbers: Sequence[Sequence[int]]) -> SLATMDescriptor:
    species = infer_species(atomic_numbers)
    return SLATMDescriptor(SLATMConfig(species=species))
