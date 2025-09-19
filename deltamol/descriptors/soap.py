"""Smooth overlap of atomic positions (SOAP) descriptor wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


from .base import Descriptor, infer_species

try:  # pragma: no cover - optional dependency
    from ase import Atoms
except ImportError:  # pragma: no cover - soft dependency
    Atoms = "Atoms"  # type: ignore

try:  # pragma: no cover - optional dependency
    from dscribe.descriptors import SOAP as _SOAP
except ImportError as exc:  # pragma: no cover
    _SOAP = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class SOAPConfig:
    """Configuration payload for the SOAP descriptor."""

    species: Sequence[int]
    r_cut: float = 6.0
    n_max: int = 4
    l_max: int = 3
    periodic: bool = False


class SOAPDescriptor(Descriptor):
    """Thin wrapper for :class:`dscribe.descriptors.SOAP`."""

    name = "soap"

    def __init__(self, config: SOAPConfig | None = None, *, species: Iterable[int] | None = None):
        if config is None and species is None:
            raise ValueError("Either a config instance or a list of species must be provided.")

        if config is None:
            config = SOAPConfig(species=tuple(sorted(set(int(s) for s in species or ()))) )

        if _SOAP is None:  # pragma: no cover
            raise ImportError("dscribe is required to build SOAP descriptors") from _IMPORT_ERROR

        super().__init__(config.species)
        self.config = config
        self._descriptor = _SOAP(
            species=config.species,
            periodic=config.periodic,
            r_cut=config.r_cut,
            n_max=config.n_max,
            l_max=config.l_max,
        )

    def create(self, atoms: "Atoms") -> np.ndarray:
        return self._descriptor.create(atoms, n_jobs=-1)


def build_soap_descriptor(atomic_numbers: Sequence[Sequence[int]]) -> SOAPDescriptor:
    species = infer_species(atomic_numbers)
    return SOAPDescriptor(SOAPConfig(species=species))
