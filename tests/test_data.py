import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deltamol.data.io import MolecularDataset, load_dataset, load_npz_dataset
from deltamol.training.datasets import MolecularGraphDataset, collate_graphs


def test_load_npz_dataset(tmp_path):
    atoms = np.array([
        np.array([1, 8, 1]),
        np.array([6, 1]),
    ], dtype=object)
    coordinates = np.array([
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.75, -0.24]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32),
    ], dtype=object)
    energies = np.array([-76.4, -40.2], dtype=np.float32)
    extra = np.array([1, 2], dtype=np.int32)
    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, atoms=atoms, coordinates=coordinates, Etot=energies, split=extra)

    dataset = load_npz_dataset(dataset_path)

    assert dataset.atoms.shape[0] == 2
    assert dataset.coordinates.shape[0] == 2
    assert dataset.energies.tolist() == pytest.approx([-76.4, -40.2], abs=1e-5)
    assert dataset.metadata["split"].tolist() == [1, 2]


def test_load_dataset_from_json(tmp_path):
    atoms = [[1, 8, 1], [6, 1]]
    coordinates = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.75, -0.24]],
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
    ]
    energies = [-76.4, -40.2]
    payload = {
        "Z": atoms,
        "xyz": coordinates,
        "energy": energies,
        "labels": ["water", "methane"],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_dataset(dataset_path)

    assert dataset.forces is None
    assert dataset.energies.tolist() == pytest.approx(energies, abs=1e-5)
    assert dataset.metadata["labels"] == ["water", "methane"]


def test_load_dataset_with_key_overrides(tmp_path):
    atoms = np.array([
        np.array([1, 1]),
        np.array([8, 1]),
    ], dtype=object)
    coords = np.array([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ], dtype=object)
    energies = np.array([-1.1, -2.2], dtype=np.float32)
    forces = np.array([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ], dtype=object)
    payload = {
        "numbers": atoms,
        "coords": coords,
        "E": energies,
        "F_total": forces,
    }
    dataset_path = tmp_path / "dataset.pt"
    torch.save(payload, dataset_path)

    dataset = load_dataset(
        dataset_path,
        format="pt",
        key_map={
            "atoms": "numbers",
            "coordinates": "coords",
            "energies": "E",
            "forces": "F_total",
        },
    )

    assert dataset.forces.shape[0] == 2
    assert dataset.forces[1].shape == (2, 3)


def test_molecular_graph_dataset_and_collate():
    atoms = np.array([
        np.array([1, 8, 1]),
        np.array([6, 1]),
    ], dtype=object)
    coordinates = np.array([
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.9, -0.3]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32),
    ], dtype=object)
    energies = np.array([-76.4, -40.2], dtype=np.float32)
    forces = np.array([
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
    ], dtype=object)
    dataset = MolecularDataset(atoms=atoms, coordinates=coordinates, energies=energies, forces=forces)

    graph_dataset = MolecularGraphDataset(dataset, cutoff=2.5)
    batch = collate_graphs([graph_dataset[0], graph_dataset[1]])

    assert batch["node_indices"].shape == (2, 3)
    assert batch["positions"].shape[-1] == 3
    assert batch["mask"].dtype == torch.bool
    assert torch.allclose(batch["energies"], torch.tensor([-76.4, -40.2]))
    assert "forces" in batch
    assert batch["forces"].shape == (2, 3, 3)


def test_molecular_graph_dataset_accepts_string_dtype():
    atoms = np.array([np.array([1, 1])], dtype=object)
    coordinates = np.array([np.zeros((2, 3), dtype=np.float32)], dtype=object)
    energies = np.array([-1.23], dtype=np.float32)
    dataset = MolecularDataset(atoms=atoms, coordinates=coordinates, energies=energies, forces=None)

    graph_dataset = MolecularGraphDataset(dataset, cutoff=3.0, dtype="float64")

    graph = graph_dataset[0]
    assert graph.positions.dtype == torch.float64
