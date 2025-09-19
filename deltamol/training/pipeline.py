"""High level training orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..models.baseline import LinearAtomicBaseline, LinearBaselineConfig


@dataclass
class TrainingConfig:
    output_dir: Path
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    log_every: int = 10
    device: str = "auto"
    validation_split: float = 0.1


class Trainer:
    """Simple trainer that optimizes a model with MSE loss."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def train(self, dataloader: DataLoader, *, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        history: Dict[str, float] = {}
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(dataloader, training=True)
            history[f"train/{epoch}"] = train_loss
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                history[f"val/{epoch}"] = val_loss
            if epoch % self.config.log_every == 0:
                print(f"Epoch {epoch:03d} | train: {train_loss:.4f}"
                      + (f" | val: {val_loss:.4f}" if val_loader is not None else ""))
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> float:
        self.model.train(mode=training)
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if training:
                self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            if training:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)


class TensorDataset(Dataset):
    """Tiny dataset wrapper around tensors."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


def train_baseline(formula_vectors: torch.Tensor, energies: torch.Tensor, *, species: Sequence[int], config: TrainingConfig) -> Trainer:
    dataset = TensorDataset(formula_vectors, energies)
    val_size = int(len(dataset) * config.validation_split)
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_dataset = dataset
        val_loader = None
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    baseline_config = LinearBaselineConfig(species=tuple(species))
    model = LinearAtomicBaseline(baseline_config)
    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader=val_loader)
    return trainer
