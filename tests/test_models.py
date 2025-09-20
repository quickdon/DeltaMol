import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deltamol.models.baseline import build_formula_vector
from deltamol.models.gcn import GCNConfig, GCNPotential
from deltamol.models.transformer import TransformerConfig, TransformerPotential


def test_build_formula_vector_counts_species():
    atoms = np.array([1, 8, 8, 6])
    species = [1, 6, 8]
    vector = build_formula_vector(atoms, species=species)

    assert torch.allclose(vector, torch.tensor([1.0, 1.0, 2.0]))


def test_gcn_forward_pass_runs():
    species = (1, 6, 8)
    config = GCNConfig(species=species, hidden_dim=16, num_layers=2, predict_forces=True)
    model = GCNPotential(config)
    node_indices = torch.tensor([[1, 2, 3, 0], [3, 1, 0, 0]], dtype=torch.long)
    positions = torch.randn(2, 4, 3)
    adjacency = torch.eye(4).repeat(2, 1, 1)
    mask = node_indices != 0

    output = model(node_indices, positions, adjacency, mask)

    assert output.energy.shape == (2,)
    assert output.forces is not None
    assert output.forces.shape == (2, 4, 3)


def test_transformer_forward_pass_runs():
    species = (1, 6)
    config = TransformerConfig(species=species, hidden_dim=8, num_layers=1, num_heads=2, predict_forces=True)
    model = TransformerPotential(config)
    node_indices = torch.tensor([[1, 2, 0], [2, 1, 1]], dtype=torch.long)
    positions = torch.randn(2, 3, 3)
    adjacency = torch.eye(3).repeat(2, 1, 1)
    mask = node_indices != 0

    output = model(node_indices, positions, adjacency, mask)

    assert output.energy.shape == (2,)
    assert output.forces is not None
    assert output.forces.shape == (2, 3, 3)
