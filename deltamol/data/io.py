"""Utilities for loading raw molecular datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

try:  # pragma: no cover - optional dependency for descriptor caching
    import h5py
except ImportError as exc:  # pragma: no cover - optional dependency
    h5py = None
    _H5PY_ERROR = exc
else:
    _H5PY_ERROR = None

try:  # pragma: no cover - optional dependency for YAML datasets
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None

try:  # pragma: no cover - optional dependency for torch serialized datasets
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


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


_CANONICAL_KEYS: Dict[str, tuple[str, ...]] = {
    "atoms": ("atoms", "atomic_numbers", "z", "species"),
    "coordinates": ("coordinates", "xyz", "positions"),
    "energies": ("energies", "energy", "Etot", "total_energy"),
    "forces": ("forces", "force", "F"),
}


def _ensure_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise TypeError("Dataset field must contain multiple entries")
        if value.dtype == object:
            return value.tolist()
        return value
    raise TypeError("Dataset field must be a sequence of per-molecule entries")


def _coerce_atoms(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            coerced = [np.asarray(entry, dtype=int) for entry in value]
            return np.array(coerced, dtype=object)
        return value.astype(int, copy=False)
    entries = [np.asarray(entry, dtype=int) for entry in _ensure_sequence(value)]
    return np.array(entries, dtype=object)


def _coerce_coordinates(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            entries = [np.asarray(entry, dtype=float) for entry in value]
            return np.array(entries, dtype=object)
        return value.astype(float, copy=False)
    entries = [np.asarray(entry, dtype=float) for entry in _ensure_sequence(value)]
    return np.array(entries, dtype=object)


def _coerce_optional(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            entries = [np.asarray(entry, dtype=float) for entry in value]
            return np.array(entries, dtype=object)
        return value.astype(float, copy=False)
    entries = [np.asarray(entry, dtype=float) for entry in _ensure_sequence(value)]
    return np.array(entries, dtype=object)


def _resolve_field(
    data: MutableMapping[str, Any],
    canonical: str,
    key_map: Optional[Mapping[str, str]] = None,
    *,
    required: bool = True,
) -> Any:
    lower_map = {str(key).lower(): key for key in data}
    if key_map and canonical in key_map:
        key = key_map[canonical]
        if key in data:
            return data.pop(key)
        lower_key = key.lower()
        if lower_key in lower_map:
            return data.pop(lower_map[lower_key])
        if required:
            raise KeyError(f"Key '{key}' (mapped from '{canonical}') missing from dataset")
        return None
    for candidate in _CANONICAL_KEYS[canonical]:
        if candidate in data:
            return data.pop(candidate)
        lowered = candidate.lower()
        if lowered in lower_map:
            return data.pop(lower_map[lowered])
    if required:
        raise KeyError(f"Dataset must provide a '{canonical}' field")
    return None


def molecular_dataset_from_dict(
    mapping: Mapping[str, Any],
    *,
    key_map: Optional[Mapping[str, str]] = None,
) -> MolecularDataset:
    data = dict(mapping)
    atoms = _coerce_atoms(_resolve_field(data, "atoms", key_map))
    coordinates = _coerce_coordinates(_resolve_field(data, "coordinates", key_map))
    energies_raw = _resolve_field(data, "energies", key_map, required=False)
    forces_raw = _resolve_field(data, "forces", key_map, required=False)
    energies = (
        np.asarray(energies_raw, dtype=float) if energies_raw is not None else None
    )
    forces = _coerce_optional(forces_raw) if forces_raw is not None else None
    num_molecules = len(atoms)
    if len(coordinates) != num_molecules:
        raise ValueError("Coordinates must contain one entry per molecule")
    if energies is not None and len(energies) != num_molecules:
        raise ValueError("Energies must contain one value per molecule")
    if forces is not None and len(forces) != num_molecules:
        raise ValueError("Forces must contain one array per molecule")
    metadata: Dict[str, Any] = {}
    if "metadata" in data:
        metadata_value = data.pop("metadata")
        if isinstance(metadata_value, Mapping):
            metadata.update(metadata_value)
        else:
            metadata["metadata"] = metadata_value
    metadata.update(data)
    return MolecularDataset(
        atoms=atoms,
        coordinates=coordinates,
        energies=energies,
        forces=forces,
        metadata=metadata or None,
    )


def _load_npz(path: Path) -> Dict[str, Any]:
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}


def _load_numpy(path: Path) -> Dict[str, Any]:
    array = np.load(path, allow_pickle=True)
    if isinstance(array, np.ndarray) and array.shape == () and isinstance(array.item(), dict):
        return dict(array.item())
    raise TypeError("NumPy dataset must contain a pickled dictionary of arrays")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Dict[str, Any]:  # pragma: no cover - optional dependency
    if yaml is None:
        raise ImportError("PyYAML is required to read YAML datasets") from _YAML_ERROR
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_torch(path: Path) -> Dict[str, Any]:  # pragma: no cover - optional dependency
    if torch is None:
        raise ImportError("PyTorch is required to read torch serialized datasets")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, Mapping):
        return dict(loaded)
    raise TypeError("Torch dataset must serialize a mapping of dataset fields")


_LOADERS: Dict[str, Callable[[Path], Dict[str, Any]]] = {
    "npz": _load_npz,
    "npy": _load_numpy,
    "npz_dict": _load_npz,
    "json": _load_json,
    "yaml": _load_yaml,
    "yml": _load_yaml,
    "pt": _load_torch,
    "pth": _load_torch,
}


def load_dataset(
    path: str | Path,
    *,
    format: Optional[str] = None,
    key_map: Optional[Mapping[str, str]] = None,
) -> MolecularDataset:
    """Load a dataset from a variety of serialisation formats.

    Parameters
    ----------
    path:
        Location of the dataset file.
    format:
        Optional override for the file format. When ``None`` or ``"auto"`` the
        loader is chosen based on the file suffix.
    key_map:
        Mapping from canonical dataset keys (``atoms``, ``coordinates``, ``energies``,
        ``forces``) to the field names used in the source file.
    """

    dataset_path = Path(path)
    resolved_format = (format or "auto").lower()
    loader: Callable[[Path], Dict[str, Any]]
    if resolved_format == "auto":
        suffix = dataset_path.suffix.lstrip(".").lower()
        if suffix not in _LOADERS:
            raise ValueError(
                f"Could not infer dataset format from suffix '.{suffix}'. "
                "Provide --dataset-format to override."
            )
        loader = _LOADERS[suffix]
    else:
        if resolved_format not in _LOADERS:
            raise ValueError(
                f"Unsupported dataset format '{format}'. Supported formats: {sorted(_LOADERS)}"
            )
        loader = _LOADERS[resolved_format]
    raw = loader(dataset_path)
    if not isinstance(raw, Mapping):
        raise TypeError("Dataset loader must return a mapping of arrays")
    return molecular_dataset_from_dict(raw, key_map=key_map)


def load_npz_dataset(path: str | Path) -> MolecularDataset:
    """Load a dataset stored in the NumPy ``.npz`` format."""

    return load_dataset(path, format="npz")


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
