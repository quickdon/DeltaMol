import json
import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from deltamol.training.pipeline import TensorDataset, Trainer, TrainingConfig, train_baseline


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
