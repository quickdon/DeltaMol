"""Atomic Cluster Symmetry Functions (ACSF) descriptor interface."""
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
    from dscribe.descriptors import ACSF as _ACSF
except ImportError as exc:  # pragma: no cover - optional dependency
    _ACSF = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _default_g2_params() -> list[list[float]]:
    rs = np.linspace(0.5, 4.0, 8)
    eta_r = np.geomspace(0.01, 0.1, 3)
    return [[float(eta), float(r)] for eta in eta_r for r in rs]


def _default_g4_params() -> list[list[float]]:
    lambda_vals = (-1, 1)
    zeta_vals = (1, 2, 4, 8)
    eta_a = (0.01, 0.05)
    return [[float(eta), float(zeta), float(lam)]
            for eta in eta_a
            for zeta in zeta_vals
            for lam in lambda_vals]


@dataclass
class ACSFConfig:
    """Configuration container for the ACSF descriptor."""

    species: Sequence[int]
    r_cut: float = 4.0
    g2_params: Sequence[Sequence[float]] | None = None
    g4_params: Sequence[Sequence[float]] | None = None


class ACSFDescriptor(Descriptor):
    """Thin wrapper around :class:`dscribe.descriptors.ACSF`."""

    name = "acsf"

    def __init__(self, config: ACSFConfig | None = None, *, species: Iterable[int] | None = None):
        if config is None and species is None:
            raise ValueError("Either a config instance or a list of species must be provided.")

        if config is None:
            species = infer_species([species or ()])  # type: ignore[arg-type]
            config = ACSFConfig(species=species)

        if config.g2_params is None:
            config.g2_params = _default_g2_params()
        if config.g4_params is None:
            config.g4_params = _default_g4_params()

        if _ACSF is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "dscribe is required to build ACSF descriptors"
            ) from _IMPORT_ERROR

        super().__init__(config.species)
        self.config = config
        self._descriptor = _ACSF(
            species=config.species,
            r_cut=config.r_cut,
            g2_params=config.g2_params,
            g4_params=config.g4_params,
        )

    def create(self, atoms: "Atoms") -> np.ndarray:
        return self._descriptor.create(atoms, n_jobs=-1)


def build_acsf_descriptor(atomic_numbers: Sequence[Sequence[int]]) -> ACSFDescriptor:
    """Convenience helper that infers species from a dataset."""

    species = infer_species(atomic_numbers)
    return ACSFDescriptor(ACSFConfig(species=species))
