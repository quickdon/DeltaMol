import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deltamol.models.baseline import build_formula_vector


def test_build_formula_vector_counts_species():
    atoms = np.array([1, 8, 8, 6])
    species = [1, 6, 8]
    vector = build_formula_vector(atoms, species=species)

    assert torch.allclose(vector, torch.tensor([1.0, 1.0, 2.0]))
