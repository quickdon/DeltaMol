"""Local many-body tensor representation (LMBTR) descriptor wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


from .base import Descriptor, infer_species

try:  # pragma: no cover
    from dscribe.descriptors import LMBTR as _LMBTR
except ImportError as exc:  # pragma: no cover
    _LMBTR = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class LMBTRConfig:
    species: Sequence[int]
    r_cut: float = 5.0
    n_grid: int = 100


class LMBTRDescriptor(Descriptor):
    name = "lmbtr"

    def __init__(self, config: LMBTRConfig | None = None, *, species: Iterable[int] | None = None):
        if config is None and species is None:
            raise ValueError("Either config or species must be provided")
        if config is None:
            config = LMBTRConfig(species=tuple(sorted(set(int(s) for s in species or ()))) )
        if _LMBTR is None:  # pragma: no cover
            raise ImportError("dscribe is required for LMBTR descriptors") from _IMPORT_ERROR
        super().__init__(config.species)
        self.config = config
        self._descriptor = _LMBTR(species=config.species, k2={"geometry": {"function": "inverse_distance"}}, k3={"geometry": {"function": "cosine"}})

    def create(self, atoms) -> np.ndarray:  # pragma: no cover
        return self._descriptor.create(atoms, n_jobs=-1)


def build_lmbtr_descriptor(atomic_numbers: Sequence[Sequence[int]]) -> LMBTRDescriptor:
    species = infer_species(atomic_numbers)
    return LMBTRDescriptor(LMBTRConfig(species=species))
