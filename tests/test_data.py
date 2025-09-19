import numpy as np
import pytest

from deltamol.data.io import load_npz_dataset


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
