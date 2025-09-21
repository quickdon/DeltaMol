import numpy as np
import pytest

pytest.importorskip("torch")

from deltamol.main import run_baseline_training


def test_run_baseline_training_creates_checkpoint(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    atoms = np.array([
        np.array([1, 1]),
        np.array([1, 8]),
    ], dtype=object)
    coordinates = np.array([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ], dtype=object)
    energies = np.array([-1.0, -2.0], dtype=np.float32)
    np.savez(dataset_path, atoms=atoms, coordinates=coordinates, Etot=energies)

    output_dir = tmp_path / "run"

    run_baseline_training(
        dataset_path,
        output_dir,
        epochs=1,
        batch_size=2,
        learning_rate=1e-2,
        validation_split=0.0,
    )

    assert (output_dir / "baseline.pt").exists()


def test_run_baseline_training_least_squares(tmp_path):
    dataset_path = tmp_path / "dataset_ls.npz"
    atoms = np.array([
        np.array([1, 6]),
        np.array([6, 8]),
        np.array([1, 8, 8]),
    ], dtype=object)
    coordinates = np.array([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
    ], dtype=object)
    energies = np.array([-1.0, -2.5, -3.0], dtype=np.float32)
    np.savez(dataset_path, atoms=atoms, coordinates=coordinates, Etot=energies)

    output_dir = tmp_path / "run_ls"

    run_baseline_training(
        dataset_path,
        output_dir,
        epochs=1,
        batch_size=3,
        learning_rate=1e-2,
        validation_split=0.0,
        solver="least_squares",
    )

    assert (output_dir / "baseline.pt").exists()
