import json
import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader, Dataset

from deltamol.models.baseline import LinearAtomicBaseline, LinearBaselineConfig
from deltamol.models.potential import PotentialOutput
from deltamol.training.pipeline import (
    PotentialTrainer,
    PotentialTrainingConfig,
    TensorDataset,
    Trainer,
    TrainingConfig,
    train_baseline,
)


def test_trainer_persists_history(tmp_path):
    torch.manual_seed(0)
    inputs = torch.randn(8, 3)
    weights = torch.tensor([[1.0], [-2.0], [0.5]])
    targets = inputs @ weights
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    config = TrainingConfig(
        output_dir=tmp_path,
        epochs=3,
        learning_rate=1e-2,
        batch_size=4,
        log_every=1,
    )
    trainer = Trainer(model, config)

    history = trainer.train(loader)

    assert history == trainer.history
    final_key = f"train/{config.epochs}"
    assert final_key in history
    history_path = tmp_path / "history.json"
    assert history_path.exists()
    with history_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved == history
    assert trainer.best_checkpoint_path == tmp_path / config.best_checkpoint_name
    assert trainer.last_checkpoint_path == tmp_path / config.last_checkpoint_name
    assert trainer.best_checkpoint_path.exists()
    assert trainer.last_checkpoint_path.exists()


def test_trainer_supports_optimizer_and_scheduler(tmp_path):
    torch.manual_seed(1)
    inputs = torch.randn(16, 4)
    weights = torch.tensor([[0.5], [-1.0], [1.5], [0.25]])
    targets = inputs @ weights
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    config = TrainingConfig(
        output_dir=tmp_path,
        epochs=2,
        learning_rate=5e-3,
        batch_size=4,
        optimizer="adamw",
        weight_decay=0.01,
        scheduler="linear",
        warmup_steps=2,
        scheduler_total_steps=8,
        min_lr_ratio=0.2,
    )
    trainer = Trainer(model, config)

    trainer.train(loader)

    assert trainer.scheduler is not None
    final_lr = trainer.scheduler.get_last_lr()[0]
    expected_lr = config.learning_rate * config.min_lr_ratio
    assert pytest.approx(expected_lr, rel=1e-5) == final_lr
    lr_keys = [key for key in trainer.history if key.startswith("lr/")]
    assert lr_keys, "learning rate history should be recorded when scheduler is active"


def test_trainer_enables_mixed_precision_cpu(tmp_path):
    torch.manual_seed(0)
    inputs = torch.randn(8, 3)
    targets = torch.randn(8, 1)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    class _AutocastRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
            self.output_dtypes = []
            self.autocast_states = []

        def forward(self, x):
            out = self.model(x)
            self.output_dtypes.append(out.dtype)
            try:
                state = torch.is_autocast_enabled(x.device.type)
            except TypeError:  # pragma: no cover - compatibility fallback
                state = torch.is_autocast_enabled()
            self.autocast_states.append(state)
            return out

    model = _AutocastRecorder()
    config = TrainingConfig(
        output_dir=tmp_path,
        epochs=1,
        batch_size=4,
        mixed_precision=True,
        autocast_dtype="bfloat16",
    )
    trainer = Trainer(model, config)

    trainer.train(loader)

    assert torch.bfloat16 in model.output_dtypes
    assert any(model.autocast_states)
    assert trainer.scaler is None


def test_train_baseline_least_squares(tmp_path):
    torch.manual_seed(0)
    formula_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    true_weights = torch.tensor([1.5, -0.75, 0.25], dtype=torch.float32)
    energies = formula_vectors @ true_weights
    config = TrainingConfig(
        output_dir=tmp_path,
        solver="least_squares",
        validation_split=0.25,
    )
    trainer = train_baseline(
        formula_vectors,
        energies,
        species=[1, 6, 8],
        config=config,
    )
    learned = trainer.model.linear.weight.detach().cpu().squeeze(0)
    assert torch.allclose(learned, true_weights, atol=1e-6)
    assert pytest.approx(0.0, abs=1e-8) == trainer.history["train/1"]
    if "val/1" in trainer.history:
        assert pytest.approx(0.0, abs=1e-8) == trainer.history["val/1"]
    history_path = tmp_path / "history.json"
    assert history_path.exists()
    assert trainer.best_checkpoint_path == tmp_path / config.best_checkpoint_name
    assert trainer.last_checkpoint_path == tmp_path / config.last_checkpoint_name
    assert trainer.best_checkpoint_path.exists()
    assert trainer.last_checkpoint_path.exists()


def test_trainer_early_stopping_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    inputs = torch.randn(12, 2)
    targets = torch.randn(12, 1)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    config = TrainingConfig(
        output_dir=tmp_path,
        epochs=5,
        learning_rate=0.0,
        batch_size=4,
        log_every=1,
        early_stopping_patience=1,
        best_checkpoint_name="best.pt",
        last_checkpoint_name="last.pt",
    )
    trainer = Trainer(model, config)

    trainer.train(loader, val_loader=val_loader)

    assert "train/5" not in trainer.history
    assert trainer.best_checkpoint_path.exists()
    assert trainer.last_checkpoint_path.exists()
    assert trainer.best_checkpoint_path == tmp_path / "best.pt"
    assert trainer.last_checkpoint_path == tmp_path / "last.pt"


class _ToyPotentialDataset(Dataset):
    def __len__(self) -> int:  # pragma: no cover - trivial
        return 1

    def __getitem__(self, index: int):  # pragma: no cover - trivial
        return {
            "energies": torch.tensor([1.0], dtype=torch.float32),
            "formula_vectors": torch.tensor([[1.0]], dtype=torch.float32),
            "positions": torch.zeros(1, 1, 3, dtype=torch.float32),
            "mask": torch.tensor([[1.0]], dtype=torch.float32),
            "node_indices": torch.tensor([[1]], dtype=torch.long),
            "adjacency": torch.ones(1, 1, 1, dtype=torch.float32),
        }


class _ZeroPotential(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_indices, positions, adjacency, mask):  # pragma: no cover - simple
        batch_size = node_indices.size(0)
        energy = torch.zeros(batch_size, device=node_indices.device) + self.offset
        return PotentialOutput(energy=energy)


def test_potential_trainer_baseline_requires_grad(tmp_path):
    dataset = _ToyPotentialDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    baseline_config = LinearBaselineConfig(species=(1,))

    trainable_dir = tmp_path / "trainable"
    trainable_config = PotentialTrainingConfig(
        output_dir=trainable_dir,
        epochs=1,
        learning_rate=0.1,
        batch_size=1,
        validation_split=0.0,
    )
    baseline = LinearAtomicBaseline(baseline_config)
    baseline.linear.weight.data.zero_()
    trainer = PotentialTrainer(
        _ZeroPotential(),
        trainable_config,
        baseline=baseline,
        baseline_requires_grad=True,
    )
    trainer.train(loader)
    assert not torch.allclose(
        baseline.linear.weight.detach(), torch.zeros_like(baseline.linear.weight)
    )
    assert trainer.best_checkpoint_path == trainable_dir / trainable_config.best_checkpoint_name
    assert trainer.last_checkpoint_path == trainable_dir / trainable_config.last_checkpoint_name
    assert trainer.best_checkpoint_path.exists()
    assert trainer.last_checkpoint_path.exists()

    frozen_dir = tmp_path / "frozen"
    frozen_config = PotentialTrainingConfig(
        output_dir=frozen_dir,
        epochs=1,
        learning_rate=0.1,
        batch_size=1,
        validation_split=0.0,
    )
    frozen_baseline = LinearAtomicBaseline(baseline_config)
    frozen_baseline.linear.weight.data.zero_()
    trainer_frozen = PotentialTrainer(
        _ZeroPotential(),
        frozen_config,
        baseline=frozen_baseline,
        baseline_requires_grad=False,
    )
    trainer_frozen.train(loader)
    assert torch.allclose(
        frozen_baseline.linear.weight.detach(), torch.zeros_like(frozen_baseline.linear.weight)
    )


def test_potential_trainer_mixed_precision_cpu(tmp_path):
    dataset = _ToyPotentialDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])

    class _AutocastPotential(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.recorded_dtypes = []
            self.autocast_states = []

        def forward(self, node_indices, positions, adjacency, mask):
            pooled = mask.sum(dim=1, keepdim=True)
            energy = self.linear(pooled).squeeze(-1)
            self.recorded_dtypes.append(energy.dtype)
            try:
                state = torch.is_autocast_enabled(energy.device.type)
            except TypeError:  # pragma: no cover - compatibility fallback
                state = torch.is_autocast_enabled()
            self.autocast_states.append(state)
            return PotentialOutput(energy=energy)

    model = _AutocastPotential()
    config = PotentialTrainingConfig(
        output_dir=tmp_path,
        epochs=1,
        batch_size=1,
        validation_split=0.0,
        mixed_precision=True,
        autocast_dtype="bfloat16",
    )
    trainer = PotentialTrainer(model, config)

    trainer.train(loader)

    assert torch.bfloat16 in model.recorded_dtypes
    assert any(model.autocast_states)
    assert trainer.scaler is None
