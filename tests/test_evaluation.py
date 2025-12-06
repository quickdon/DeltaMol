import numpy as np
import torch

from deltamol.data.io import MolecularDataset
from deltamol.evaluation.testing import (
    evaluate_baseline_model,
    evaluate_potential_model,
    plot_predictions_vs_targets,
)
from deltamol.models.baseline import LinearAtomicBaseline, LinearBaselineConfig
from deltamol.models.potential import PotentialOutput
from deltamol.training.datasets import MolecularGraphDataset


def test_evaluate_baseline_and_plot(tmp_path):
    model = LinearAtomicBaseline(LinearBaselineConfig(species=(1, 6)))
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([1.0, -1.0])
    )
    metrics, predictions, targets = evaluate_baseline_model(model, dataset)
    assert set(metrics) == {"mse", "rmse", "mae"}
    plot_path = tmp_path / "baseline.png"
    saved_path = plot_predictions_vs_targets(predictions, targets, plot_path)
    assert saved_path.exists()


def test_evaluate_potential_model(tmp_path):
    class DummyPotential(torch.nn.Module):
        def forward(self, node_indices, positions, adjacency, mask):
            atom_counts = mask.sum(dim=1).float()
            return PotentialOutput(energy=atom_counts)

    dataset = MolecularDataset(
        atoms=np.array([np.array([1, 1]), np.array([1])], dtype=object),
        coordinates=np.array(
            [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 0.0]])],
            dtype=object,
        ),
        energies=np.array([2.0, 1.0]),
    )
    graph_dataset = MolecularGraphDataset(dataset, species=(1,))
    model = DummyPotential()
    metrics, predictions, targets, force_predictions, force_targets = evaluate_potential_model(
        model, graph_dataset
    )
    assert metrics["mae"] < 1e-6
    assert force_predictions is None
    assert force_targets is None
    plot_path = tmp_path / "potential.png"
    saved_path = plot_predictions_vs_targets(predictions, targets, plot_path)
    assert saved_path.exists()


def test_evaluate_potential_model_with_forces():
    class DummyForcePotential(torch.nn.Module):
        def forward(self, node_indices, positions, adjacency, mask):
            atom_counts = mask.sum(dim=1).float()
            forces = torch.ones_like(positions) * mask.unsqueeze(-1)
            return PotentialOutput(energy=atom_counts, forces=forces)

    dataset = MolecularDataset(
        atoms=np.array([np.array([1, 1]), np.array([1])], dtype=object),
        coordinates=np.array(
            [
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([[0.0, 0.0, 0.0]]),
            ],
            dtype=object,
        ),
        energies=np.array([2.0, 1.0]),
        forces=np.array(
            [
                np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                np.array([[1.0, 1.0, 1.0]]),
            ],
            dtype=object,
        ),
    )
    graph_dataset = MolecularGraphDataset(dataset, species=(1,))
    model = DummyForcePotential()
    metrics, _, _, force_predictions, force_targets = evaluate_potential_model(
        model, graph_dataset
    )
    assert metrics["force_mae"] < 1e-6
    assert metrics["force_per_atom_mae"] < 1e-6
    assert force_predictions is not None
    assert force_targets is not None

