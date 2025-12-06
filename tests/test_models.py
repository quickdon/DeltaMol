import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deltamol.models.baseline import build_formula_vector
from deltamol.models.hybrid import HybridPotential, HybridPotentialConfig
from deltamol.models.se3 import SE3TransformerConfig, SE3TransformerPotential


def test_build_formula_vector_counts_species():
    atoms = np.array([1, 8, 8, 6])
    species = [1, 6, 8]
    vector = build_formula_vector(atoms, species=species)

    assert torch.allclose(vector, torch.tensor([1.0, 1.0, 2.0]))


def test_hybrid_forward_pass_runs():
    torch.manual_seed(0)
    species = (1, 6, 8)
    config = HybridPotentialConfig(
        species=species,
        hidden_dim=16,
        gcn_layers=2,
        transformer_layers=1,
        num_heads=2,
        dropout=0.0,
        ffn_dim=32,
        predict_forces=True,
        soap_num_radial=6,
        soap_cutoff=3.0,
        soap_gaussian_width=0.6,
    )
    model = HybridPotential(config)
    node_indices = torch.tensor([[1, 2, 3, 0], [3, 1, 0, 0]], dtype=torch.long)
    positions = torch.randn(2, 4, 3)
    adjacency = torch.eye(4).repeat(2, 1, 1)
    mask = node_indices != 0

    output = model(node_indices, positions, adjacency, mask)

    assert output.energy.shape == (2,)
    assert output.forces is not None
    assert output.forces.shape == (2, 4, 3)


def test_hybrid_rejects_invalid_head_configuration():
    species = (1, 6)
    config = HybridPotentialConfig(
        species=species,
        hidden_dim=128,
        num_heads=5,
        gcn_layers=1,
        transformer_layers=1,
    )

    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        HybridPotential(config)


def test_hybrid_forces_match_finite_difference():
    torch.manual_seed(0)
    species = (1,)
    config = HybridPotentialConfig(
        species=species,
        hidden_dim=16,
        gcn_layers=1,
        transformer_layers=1,
        num_heads=2,
        dropout=0.0,
        ffn_dim=32,
        predict_forces=False,
        soap_num_radial=6,
        soap_cutoff=3.0,
        soap_gaussian_width=0.6,
    )
    model = HybridPotential(config).double()
    model.eval()

    node_indices = torch.tensor([[1, 1]], dtype=torch.long)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.4, -0.2]]], dtype=torch.double)
    positions.requires_grad_(True)
    adjacency = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.double)
    mask = node_indices != 0

    energy = model(node_indices, positions, adjacency, mask).energy.sum()
    grad = torch.autograd.grad(energy, positions)[0]
    forces = -grad

    eps = 1e-4
    numeric_grad = torch.zeros_like(positions)
    with torch.no_grad():
        for atom in range(positions.shape[1]):
            for coord in range(3):
                pos_plus = positions.detach().clone()
                pos_minus = positions.detach().clone()
                pos_plus[0, atom, coord] += eps
                pos_minus[0, atom, coord] -= eps
                e_plus = model(node_indices, pos_plus, adjacency, mask).energy.sum()
                e_minus = model(node_indices, pos_minus, adjacency, mask).energy.sum()
                numeric_grad[0, atom, coord] = (e_plus - e_minus) / (2 * eps)

    assert torch.max(torch.abs(forces)) > 0
    assert torch.allclose(forces, -numeric_grad, atol=1e-2, rtol=1e-2)


def test_se3_forward_pass_runs():
    torch.manual_seed(0)
    species = (1, 6, 8)
    config = SE3TransformerConfig(
        species=species,
        hidden_dim=24,
        num_layers=2,
        num_heads=4,
        ffn_dim=48,
        distance_embedding_dim=8,
        dropout=0.0,
        cutoff=3.5,
        predict_forces=True,
    )
    model = SE3TransformerPotential(config)
    node_indices = torch.tensor([[1, 2, 3, 0], [3, 1, 0, 0]], dtype=torch.long)
    positions = torch.randn(2, 4, 3)
    adjacency = torch.eye(4).repeat(2, 1, 1)
    mask = node_indices != 0

    output = model(node_indices, positions, adjacency, mask)

    assert output.energy.shape == (2,)
    assert output.forces is not None
    assert output.forces.shape == (2, 4, 3)


def test_se3_energy_dependent_on_coordinates():
    torch.manual_seed(0)
    species = (1,)
    config = SE3TransformerConfig(
        species=species,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=32,
        distance_embedding_dim=6,
        dropout=0.0,
        cutoff=2.5,
        predict_forces=False,
    )
    model = SE3TransformerPotential(config).double()
    model.eval()

    node_indices = torch.tensor([[1, 1]], dtype=torch.long)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.3, -0.1]]], dtype=torch.double)
    positions.requires_grad_(True)
    mask = node_indices != 0

    energy = model(node_indices, positions, None, mask).energy.sum()
    grad = torch.autograd.grad(energy, positions)[0]

    assert torch.max(torch.abs(grad)) > 0
