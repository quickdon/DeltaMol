"""Utilities for loading raw molecular datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

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
    "coordinates": ("coordinates", "xyz", "positions", "R"),
    "energies": ("energies", "energy", "Etot", "total_energy", "E"),
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


def _broadcast_atomic_numbers(atoms: np.ndarray, target: int) -> np.ndarray:
    if atoms.dtype != object:
        atoms = np.array([np.asarray(atoms, dtype=int)], dtype=object)
    else:
        atoms = np.array([np.asarray(entry, dtype=int) for entry in atoms], dtype=object)
    length = len(atoms)
    if length == target:
        return atoms
    if length == 1:
        return np.array([atoms[0] for _ in range(target)], dtype=object)
    raise ValueError("Atoms must contain one entry per molecule or a single shared entry")


def _broadcast_optional_array(
    name: str, values: Optional[np.ndarray], target: int
) -> Optional[np.ndarray]:
    if values is None:
        return None
    if np.isscalar(values):
        return np.full(target, float(values), dtype=float)
    if isinstance(values, np.ndarray) and values.ndim == 0:
        return np.full(target, float(values), dtype=float)
    if isinstance(values, np.ndarray) and values.ndim == 2 and values.shape[-1] == 3:
        values = np.expand_dims(values, 0)
    length = len(values)
    if length == target:
        return values
    if length == 1:
        if values.dtype == object:
            return np.array([values[0] for _ in range(target)], dtype=object)
        return np.repeat(values, target, axis=0)
    raise ValueError(f"{name} must contain one entry per molecule or a single shared entry")


def molecular_dataset_from_dict(
    mapping: Mapping[str, Any],
    *,
    key_map: Optional[Mapping[str, str]] = None,
) -> MolecularDataset:
    data = dict(mapping)
    atoms = _coerce_atoms(_resolve_field(data, "atoms", key_map))
    coordinates = _coerce_coordinates(_resolve_field(data, "coordinates", key_map))
    if isinstance(coordinates, np.ndarray) and coordinates.ndim == 2 and coordinates.shape[-1] == 3:
        coordinates = np.expand_dims(coordinates, 0)
    num_entries = len(coordinates)
    atoms = _broadcast_atomic_numbers(atoms, num_entries)
    energies_raw = _resolve_field(data, "energies", key_map, required=False)
    forces_raw = _resolve_field(data, "forces", key_map, required=False)
    energies = _broadcast_optional_array(
        "Energies",
        np.asarray(energies_raw, dtype=float) if energies_raw is not None else None,
        num_entries,
    )
    forces = _broadcast_optional_array(
        "Forces", _coerce_optional(forces_raw) if forces_raw is not None else None, num_entries
    )
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


def _select_loader(dataset_path: Path, resolved_format: str) -> Callable[[Path], Dict[str, Any]]:
    if resolved_format == "auto":
        suffix = dataset_path.suffix.lstrip(".").lower()
        if suffix not in _LOADERS:
            raise ValueError(
                f"Could not infer dataset format from suffix '.{suffix}'. "
                "Provide --dataset-format to override."
            )
        return _LOADERS[suffix]
    if resolved_format not in _LOADERS:
        raise ValueError(
            f"Unsupported dataset format '{resolved_format}'. Supported formats: {sorted(_LOADERS)}"
        )
    return _LOADERS[resolved_format]


def _concat_datasets(datasets: Sequence[MolecularDataset]) -> MolecularDataset:
    if not datasets:
        raise ValueError("At least one dataset is required for concatenation")
    atoms = np.array([atom for ds in datasets for atom in ds.atoms], dtype=object)
    coordinates = np.array(
        [coords for ds in datasets for coords in ds.coordinates], dtype=object
    )
    energies = None
    forces = None
    metadata_entries = []
    if any(ds.energies is not None for ds in datasets):
        energies_list = [ds.energies for ds in datasets if ds.energies is not None]
        energies = np.concatenate(energies_list, axis=0)
    if any(ds.forces is not None for ds in datasets):
        forces_list = [ds.forces for ds in datasets if ds.forces is not None]
        forces = np.concatenate(forces_list, axis=0)
    for ds in datasets:
        if ds.metadata is not None:
            metadata_entries.append(ds.metadata)
    metadata = {"datasets": metadata_entries} if metadata_entries else None
    return MolecularDataset(
        atoms=atoms,
        coordinates=coordinates,
        energies=energies,
        forces=forces,
        metadata=metadata,
    )


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

    resolved_format = (format or "auto").lower()
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, Path)):
        datasets = [load_dataset(p, format=resolved_format, key_map=key_map) for p in path]
        return _concat_datasets(datasets)
    dataset_path = Path(path)
    if dataset_path.is_dir():
        datasets = []
        for file_path in sorted(dataset_path.iterdir()):
            if not file_path.is_file():
                continue
            try:
                loader = _select_loader(file_path, resolved_format)
            except ValueError:
                continue
            raw = loader(file_path)
            if not isinstance(raw, Mapping):
                raise TypeError("Dataset loader must return a mapping of arrays")
            datasets.append(molecular_dataset_from_dict(raw, key_map=key_map))
        if not datasets:
            raise ValueError(
                f"No datasets matching format '{resolved_format}' were found in {dataset_path}"
            )
        return _concat_datasets(datasets)
    loader = _select_loader(dataset_path, resolved_format)
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
