"""Utilities for loading raw molecular datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # pragma: no cover - optional dependency for descriptor caching
    import h5py
except ImportError as exc:  # pragma: no cover - optional dependency
    h5py = None
    _H5PY_ERROR = exc
else:
    _H5PY_ERROR = None


@dataclass
class MolecularDataset:
    """Container that stores the minimal pieces required across the pipeline."""

    atoms: np.ndarray
    coordinates: np.ndarray
    energies: np.ndarray | None = None
    forces: np.ndarray | None = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "atoms": self.atoms,
            "coordinates": self.coordinates,
        }
        if self.energies is not None:
            data["energies"] = self.energies
        if self.forces is not None:
            data["forces"] = self.forces
        if self.metadata:
            data.update(self.metadata)
        return data


def load_npz_dataset(path: str | Path) -> MolecularDataset:
    """Load a dataset stored in the NumPy ``.npz`` format."""

    archive = np.load(Path(path), allow_pickle=True)
    arrays = {key: archive[key] for key in archive.files}
    atoms = arrays["atoms"]
    coordinates = arrays["coordinates"]
    energies = arrays.get("Etot")
    if energies is None:
        energies = arrays.get("energies")
    if energies is None:
        raise KeyError("Dataset must contain either 'Etot' or 'energies' array")
    forces = arrays.get("forces")
    metadata = {
        k: arrays[k]
        for k in arrays
        if k not in {"atoms", "coordinates", "Etot", "energies", "forces"}
    }
    return MolecularDataset(
        atoms=atoms,
        coordinates=coordinates,
        energies=energies,
        forces=forces,
        metadata=metadata,
    )


def cache_descriptor_matrix(
    path: str | Path, name: str, matrix: np.ndarray, *, compression: str = "gzip"
) -> None:
    """Persist a descriptor matrix to a HDF5 cache file."""

    if h5py is None:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required for descriptor caching") from _H5PY_ERROR

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "a") as handle:
        if name in handle:
            del handle[name]
        handle.create_dataset(str(name), data=matrix, compression=compression)


def load_descriptor_matrix(path: str | Path, name: str) -> np.ndarray:
    if h5py is None:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to load descriptor caches") from _H5PY_ERROR
    with h5py.File(Path(path), "r") as handle:
        return handle[str(name)][()]
