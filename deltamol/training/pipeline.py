"""High level training orchestration utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..models.baseline import LinearAtomicBaseline, LinearBaselineConfig
from ..models.potential import PotentialOutput
from .datasets import MolecularGraphDataset, collate_graphs

LOGGER = logging.getLogger(__name__)


def _emit_info(message: str) -> None:
    """Emit an informational message via logging with stdout fallback."""

    LOGGER.info(message)
    if not (LOGGER.hasHandlers() and LOGGER.isEnabledFor(logging.INFO)):
        print(message)


def _save_history(output_dir: Path, history: Dict[str, float]) -> Path:
    """Persist a training history dictionary to ``history.json``."""

    history_path = output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    _emit_info(f"Saved training history to {history_path}")
    return history_path


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
        self.history: Dict[str, float] = {}

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
        train_samples = len(getattr(dataloader, "dataset", []))
        val_samples = len(getattr(val_loader, "dataset", [])) if val_loader is not None else 0
        summary = f"Starting training for {self.config.epochs} epochs on {train_samples} samples"
        if val_loader is not None:
            summary += f" with {val_samples} validation samples"
        _emit_info(summary)
        log_interval = max(int(self.config.log_every), 1)
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(dataloader, training=True)
            history[f"train/{epoch}"] = train_loss
            val_loss: Optional[float] = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                history[f"val/{epoch}"] = val_loss
            if epoch == 1 or epoch == self.config.epochs or epoch % log_interval == 0:
                message = f"Epoch {epoch:03d} | train: {train_loss:.4f}"
                if val_loss is not None:
                    message += f" | val: {val_loss:.4f}"
                _emit_info(message)
        self.history = history
        _save_history(self.output_dir, history)
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


@dataclass
class PotentialTrainingConfig(TrainingConfig):
    """Configuration for potential energy/force training."""

    energy_weight: float = 1.0
    force_weight: float = 0.0
    predict_forces_directly: bool = False
    max_grad_norm: Optional[float] = None


class PotentialTrainer:
    """Trainer that optimizes potential models for energies and forces."""

    def __init__(
        self,
        model: nn.Module,
        config: PotentialTrainingConfig,
        *,
        baseline: Optional[LinearAtomicBaseline] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.energy_loss = nn.MSELoss()
        self.force_loss = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = baseline
        self.history: Dict[str, float] = {}
        if self.baseline is not None:
            self.baseline.to(self.device)
            self.baseline.eval()
            for param in self.baseline.parameters():
                param.requires_grad_(False)

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
        train_samples = len(getattr(dataloader, "dataset", []))
        val_samples = len(getattr(val_loader, "dataset", [])) if val_loader is not None else 0
        summary = (
            f"Starting potential training for {self.config.epochs} epochs on {train_samples} samples"
        )
        if val_loader is not None:
            summary += f" with {val_samples} validation samples"
        details = [f"energy weight={self.config.energy_weight}"]
        if self.config.force_weight > 0.0:
            details.append(f"force weight={self.config.force_weight}")
            if self.config.predict_forces_directly:
                details.append("predicting forces directly")
        if details:
            summary += " (" + ", ".join(details) + ")"
        _emit_info(summary)
        log_interval = max(int(self.config.log_every), 1)
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(dataloader, training=True)
            history[f"train/{epoch}"] = train_metrics["loss"]
            val_metrics: Optional[Dict[str, float]] = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False)
                history[f"val/{epoch}"] = val_metrics["loss"]
            if epoch == 1 or epoch == self.config.epochs or epoch % log_interval == 0:
                message = f"Epoch {epoch:03d} | train: {train_metrics['loss']:.4f}"
                if val_metrics is not None:
                    message += f" | val: {val_metrics['loss']:.4f}"
                _emit_info(message)
        self.history = history
        _save_history(self.output_dir, history)
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> Dict[str, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = {key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            energies = batch["energies"]
            formula_vectors = batch["formula_vectors"]
            baseline_energy = None
            if self.baseline is not None:
                with torch.no_grad():
                    baseline_energy = self.baseline(formula_vectors).detach()
                target_energy = energies - baseline_energy
            else:
                target_energy = energies
            requires_force_grad = (
                self.config.force_weight > 0.0
                and not self.config.predict_forces_directly
                and batch.get("forces") is not None
            )
            positions = batch["positions"]
            if requires_force_grad:
                positions = positions.clone().detach().requires_grad_(True)
                batch["positions"] = positions
            self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training or requires_force_grad):
                output = self._forward_model(batch)
                energy_pred = output.energy
                energy_loss = self.energy_loss(energy_pred, target_energy)
                loss = self.config.energy_weight * energy_loss
                force_loss_value = torch.tensor(0.0, device=self.device)
                if batch.get("forces") is not None and self.config.force_weight > 0.0:
                    if output.forces is not None and self.config.predict_forces_directly:
                        predicted_forces = output.forces
                    else:
                        grads = torch.autograd.grad(
                            energy_pred.sum(),
                            positions,
                            create_graph=training,
                            retain_graph=training,
                        )[0]
                        predicted_forces = -grads
                    mask = batch["mask"].unsqueeze(-1)
                    target_forces = batch["forces"] * mask
                    predicted_forces = predicted_forces * mask
                    force_loss_value = self.force_loss(predicted_forces, target_forces)
                    loss = loss + self.config.force_weight * force_loss_value
            if training:
                loss.backward()
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            total_loss += loss.item()
            total_energy_loss += energy_loss.item()
            total_force_loss += force_loss_value.item()
            n_batches += 1
        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "energy_loss": total_energy_loss / max(n_batches, 1),
        }
        if self.config.force_weight > 0.0:
            metrics["force_loss"] = total_force_loss / max(n_batches, 1)
        return metrics

    def _forward_model(self, batch: Dict[str, torch.Tensor]) -> PotentialOutput:
        return self.model(
            batch["node_indices"],
            batch["positions"],
            batch["adjacency"],
            batch["mask"],
        )

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)


def train_potential_model(
    dataset: MolecularGraphDataset,
    model: nn.Module,
    *,
    config: PotentialTrainingConfig,
    baseline: Optional[LinearAtomicBaseline] = None,
) -> PotentialTrainer:
    val_size = int(len(dataset) * config.validation_split)
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
    else:
        train_dataset = dataset
        val_loader = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    trainer = PotentialTrainer(model, config, baseline=baseline)
    trainer.train(train_loader, val_loader=val_loader)
    return trainer
