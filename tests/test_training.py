import json
import pytest

torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from deltamol.training.pipeline import TensorDataset, Trainer, TrainingConfig


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
