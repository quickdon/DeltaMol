import numpy as np
import torch

from deltamol.data.io import MolecularDataset
from deltamol.models.adapters import AdapterInputs, PotentialModelAdapter
from deltamol.models.potential import PotentialOutput
from deltamol.training.datasets import MolecularGraphDataset, collate_graphs
from deltamol.training.pipeline import PotentialTrainingConfig, train_potential_model


class _DummyExternal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, prepared: AdapterInputs):
        max_atoms = max(pos.size(0) for pos in prepared.positions)
        padded_forces = []
        energies = []
        for coords in prepared.positions:
            energies.append(coords.sum() + self.offset)
            pad = torch.zeros(max_atoms, 3, device=coords.device, dtype=coords.dtype)
            pad[: coords.size(0)] = 0.5
            padded_forces.append(pad)
        return {
            "energy": torch.stack(energies),
            "forces": torch.stack(padded_forces),
        }


def _build_small_dataset() -> MolecularGraphDataset:
    atoms = np.array([np.array([1, 1]), np.array([6, 1, 1])], dtype=object)
    coordinates = np.array(
        [np.zeros((2, 3), dtype=float), np.array([[0.0, 0.1, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])],
        dtype=object,
    )
    energies = np.array([0.0, 1.0], dtype=float)
    forces = np.array([np.zeros((2, 3), dtype=float), np.ones((3, 3), dtype=float)], dtype=object)
    dataset = MolecularDataset(atoms=atoms, coordinates=coordinates, energies=energies, forces=forces)
    return MolecularGraphDataset(dataset, species=(1, 6), cutoff=1.5)


def test_adapter_converts_external_outputs():
    dataset = _build_small_dataset()
    batch = collate_graphs([dataset[0], dataset[1]])
    adapter = PotentialModelAdapter(_DummyExternal())
    output = adapter(
        batch["node_indices"], batch["positions"], batch["adjacency"], batch["mask"]
    )
    assert isinstance(output, PotentialOutput)
    assert output.energy.shape[0] == 2
    assert output.forces is not None
    assert output.forces.shape == batch["positions"].shape


def test_adapter_supports_training_step(tmp_path):
    dataset = _build_small_dataset()
    model = PotentialModelAdapter(_DummyExternal())
    config = PotentialTrainingConfig(
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        log_every=1,
        log_every_steps=1,
        validation_split=0,
        tensorboard=False,
        force_weight=1.0,
        predict_forces_directly=True,
    )
    trainer = train_potential_model(dataset, model, config=config)
    assert trainer.last_checkpoint_path is not None
    assert trainer.history
