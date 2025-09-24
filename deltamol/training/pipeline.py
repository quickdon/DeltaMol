"""High level training orchestration utilities."""
from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split

from ..models.baseline import LinearAtomicBaseline, LinearBaselineConfig
from ..models.potential import PotentialOutput
from .datasets import MolecularGraphDataset, collate_graphs
from ..utils.distributed import DistributedConfig, DistributedState, init_distributed, is_main_process

LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - torch<2.0 fallback
    from torch.amp import GradScaler as _GradScalerBase
except ImportError:  # pragma: no cover - legacy path
    from torch.cuda.amp import GradScaler as _GradScalerBase  # type: ignore[attr-defined]


def _make_grad_scaler(enabled: bool) -> _GradScalerBase:
    """Instantiate a :class:`GradScaler` handling old and new signatures."""

    try:
        return _GradScalerBase(device_type="cuda", enabled=enabled)  # type: ignore[call-arg]
    except TypeError:  # pragma: no cover - torch<2.0 signature
        return _GradScalerBase(enabled=enabled)


def _emit_info(message: str) -> None:
    """Emit an informational message via logging with stdout fallback."""

    if not is_main_process():
        return
    LOGGER.info(message)
    if not (LOGGER.hasHandlers() and LOGGER.isEnabledFor(logging.INFO)):
        print(message)


def _save_history(output_dir: Path, history: Dict[str, float]) -> Path:
    """Persist a training history dictionary to ``history.json``."""

    history_path = output_dir / "history.json"
    if is_main_process():
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
    _emit_info(f"Saved training history to {history_path}")
    return history_path


def _resolve_autocast_settings(
    device: torch.device, config: "TrainingConfig"
) -> Tuple[bool, Optional[str], Optional[torch.dtype], bool]:
    """Determine whether autocast should be enabled for the trainer."""

    if not getattr(config, "mixed_precision", False):
        return False, None, None, False
    device_type = device.type
    dtype_name = str(getattr(config, "autocast_dtype", None) or "float16").lower()
    if device_type == "cuda":
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported autocast dtype '{dtype_name}' for CUDA mixed precision")
        dtype = mapping[dtype_name]
        scaler_enabled = bool(getattr(config, "grad_scaler", True)) and dtype == torch.float16
        return True, device_type, dtype, scaler_enabled
    if device_type == "cpu":
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.bfloat16,
            "fp16": torch.bfloat16,
            "half": torch.bfloat16,
        }
        dtype = mapping.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported autocast dtype '{dtype_name}' for CPU mixed precision")
        if dtype_name not in {"bfloat16", "bf16"}:
            _emit_info(
                "CPU mixed precision only supports bfloat16; falling back to bfloat16 autocast."
            )
        return True, device_type, torch.bfloat16, False
    _emit_info(
        "Mixed precision requested on device type '%s', but autocast is not supported; disabling mixed precision."
        % device_type
    )
    return False, None, None, False


def _autocast_context(
    enabled: bool, device_type: Optional[str], dtype: Optional[torch.dtype]
):
    if not enabled or device_type is None or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


class WarmupDecayScheduler:
    """Learning rate scheduler with warmup and configurable decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        strategy: str,
        min_lr_ratio: float,
        gamma: float,
        step_size: int,
    ) -> None:
        self.optimizer = optimizer
        self.strategy = strategy
        self.warmup_steps = max(int(warmup_steps), 0)
        inferred_total = max(int(total_steps), 1)
        self.total_steps = max(inferred_total, self.warmup_steps + 1)
        self.min_lr_ratio = max(float(min_lr_ratio), 0.0)
        self.gamma = float(gamma)
        self.step_size = max(int(step_size), 1)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0
        self.last_lrs = list(self.base_lrs)
        self._apply_lrs()

    def state_dict(self) -> Dict[str, object]:
        return {
            "current_step": self.current_step,
            "last_lrs": list(self.last_lrs),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.current_step = int(state_dict.get("current_step", 0))
        self._apply_lrs()

    def step(self) -> None:
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
        self._apply_lrs()

    def _apply_lrs(self) -> None:
        factor = self._compute_factor(self.current_step)
        self.last_lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = base_lr * factor
            group["lr"] = new_lr
            self.last_lrs.append(new_lr)

    def _compute_factor(self, step: int) -> float:
        if self.total_steps <= 1:
            return 1.0
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 1:
            return 1.0
        progress = (step - self.warmup_steps) / max(decay_steps - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        if self.strategy == "linear":
            value = 1.0 - (1.0 - self.min_lr_ratio) * progress
        elif self.strategy == "cosine":
            value = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        elif self.strategy == "exponential":
            exponent = max(step - self.warmup_steps, 0)
            value = self.gamma**exponent
        elif self.strategy == "step":
            exponent = max(step - self.warmup_steps, 0) // self.step_size
            value = self.gamma**exponent
        elif self.strategy == "constant":
            value = 1.0
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"Unknown scheduler strategy: {self.strategy}")
        return max(value, self.min_lr_ratio)

    def get_last_lr(self) -> Sequence[float]:
        return list(self.last_lrs)


class _PotentialDDPWrapper(nn.Module):
    """Wrapper that exposes baseline parameters to DDP while delegating forwards."""

    def __init__(self, model: nn.Module, baseline: Optional[nn.Module]):
        super().__init__()
        self.model = model
        if baseline is not None:
            self.baseline = baseline

    def forward(self, node_indices, positions, adjacency, mask):  # pragma: no cover - delegate
        return self.model(node_indices, positions, adjacency, mask)


def _build_optimizer(parameters, config: TrainingConfig) -> Optimizer:
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad,
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'")


def _maybe_build_scheduler(
    optimizer: Optimizer, config: TrainingConfig, steps_per_epoch: Optional[int]
) -> Optional[WarmupDecayScheduler]:
    strategy = (config.scheduler or "").lower()
    if not strategy:
        return None
    supported = {"linear", "cosine", "exponential", "step", "constant"}
    if strategy not in supported:
        raise ValueError(f"Unsupported scheduler '{config.scheduler}'")
    if config.scheduler_total_steps is not None:
        total_steps = int(config.scheduler_total_steps)
    elif steps_per_epoch is not None:
        total_steps = steps_per_epoch * max(config.epochs, 1)
    else:
        return None
    if total_steps <= 0:
        return None
    return WarmupDecayScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        strategy=strategy,
        min_lr_ratio=config.min_lr_ratio,
        gamma=config.scheduler_gamma,
        step_size=config.scheduler_step_size,
    )


@dataclass
class TrainingConfig:
    output_dir: Path
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 0
    update_frequency: int = 1
    log_every: int = 1
    device: str = "auto"
    mixed_precision: bool = False
    autocast_dtype: str = "float16"
    grad_scaler: bool = True
    validation_split: float = 0.1
    optimizer: str = "adam"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    momentum: float = 0.9
    nesterov: bool = False
    solver: str = "optimizer"
    scheduler: Optional[str] = None
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    scheduler_gamma: float = 0.1
    scheduler_step_size: int = 1000
    scheduler_total_steps: Optional[int] = None
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


class Trainer:
    """Simple trainer that optimizes a model with MSE loss."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        base_device = self._resolve_device(config.device)
        self.distributed: DistributedState = init_distributed(config.distributed, base_device)
        self.device = self.distributed.device
        self.model.to(self.device)
        self.ddp_model: Optional[nn.parallel.DistributedDataParallel]
        if self.distributed.enabled:
            ddp_kwargs = {
                "find_unused_parameters": config.distributed.find_unused_parameters,
                "broadcast_buffers": config.distributed.broadcast_buffers,
            }
            if self.device.type == "cuda":
                device_index = self.device.index if self.device.index is not None else 0
                ddp_kwargs["device_ids"] = [device_index]
                ddp_kwargs["output_device"] = device_index
            self.ddp_model = nn.parallel.DistributedDataParallel(self.model, **ddp_kwargs)
            param_source = self.ddp_model.parameters()
        else:
            self.ddp_model = None
            param_source = self.model.parameters()
        self.optimizer = _build_optimizer(param_source, config)
        (
            self._amp_enabled,
            self._autocast_device,
            self._autocast_dtype,
            scaler_enabled,
        ) = _resolve_autocast_settings(self.device, config)
        self.scaler: Optional[_GradScalerBase]
        if self._amp_enabled and self._autocast_device == "cuda":
            self.scaler = _make_grad_scaler(enabled=scaler_enabled)
        else:
            self.scaler = None
        self.scheduler: Optional[WarmupDecayScheduler] = None
        self.criterion = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, float] = {}
        self.best_checkpoint_path: Optional[Path] = None
        self.last_checkpoint_path: Optional[Path] = None
        self._best_metric: Optional[float] = None
        self._early_stop_counter = 0
        self._update_frequency = max(int(self.config.update_frequency), 1)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def train(
        self,
        dataloader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        train_sampler=None,
    ) -> Dict[str, float]:
        history: Dict[str, float] = {}
        train_samples = len(getattr(dataloader, "dataset", []))
        val_samples = len(getattr(val_loader, "dataset", [])) if val_loader is not None else 0
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        self.scheduler = _maybe_build_scheduler(self.optimizer, self.config, steps_per_epoch)
        summary = f"Starting training for {self.config.epochs} epochs on {train_samples} samples"
        if val_loader is not None:
            summary += f" with {val_samples} validation samples"
        summary += f" | optimizer={self.config.optimizer}"
        if self.scheduler is not None:
            summary += f", scheduler={self.config.scheduler}"
            if self.config.warmup_steps > 0:
                summary += f" (warmup={self.config.warmup_steps})"
        if self.distributed.enabled and self.distributed.world_size > 1:
            summary += f", world size={self.distributed.world_size}"
        if self._update_frequency > 1:
            summary += f", update frequency={self._update_frequency}"
        effective_batch = (
            self.config.batch_size
            * max(self.distributed.world_size, 1)
            * max(self._update_frequency, 1)
        )
        if effective_batch != self.config.batch_size:
            summary += f", effective batch={effective_batch}"
        _emit_info(summary)
        log_interval = max(int(self.config.log_every), 1)
        for epoch in range(1, self.config.epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
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
            if self.scheduler is not None:
                lr_value = float(self.scheduler.get_last_lr()[0])
                history[f"lr/{epoch}"] = lr_value
            self._update_checkpoints(train_loss, val_loss)
            if self._should_stop_early(val_loss):
                _emit_info(
                    "Early stopping triggered after epoch %03d (best %.4f)"
                    % (epoch, self._best_metric if self._best_metric is not None else train_loss)
                )
                break
        self.history = history
        _save_history(self.output_dir, history)
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> float:
        module = self.ddp_model if self.ddp_model is not None else self.model
        module.train(mode=training)
        total_loss = 0.0
        n_batches = 0
        pending_update = False
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        for step, batch in enumerate(dataloader, start=1):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if training and ((step - 1) % self._update_frequency == 0):
                self.optimizer.zero_grad(set_to_none=True)
            with _autocast_context(
                self._amp_enabled, self._autocast_device, self._autocast_dtype
            ):
                outputs = module(inputs)
                raw_loss = self.criterion(outputs, targets)
            loss_value = raw_loss.detach().item()
            loss = raw_loss
            if training and self._update_frequency > 1:
                loss = loss / float(self._update_frequency)
            if training:
                if self.scaler is not None:
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                else:
                    loss.backward()
                pending_update = True
                should_step = (step % self._update_frequency == 0)
                if total_batches is not None and step == total_batches:
                    should_step = True
                if should_step:
                    self._apply_optimizer_step()
                    pending_update = False
            total_loss += loss_value
            n_batches += 1
        if training and pending_update:
            self._apply_optimizer_step()
        reduce_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        stats = torch.tensor(
            [total_loss, float(n_batches)],
            device=reduce_device,
            dtype=torch.float64,
        )
        stats = self.distributed.all_reduce(stats)
        total_loss = float(stats[0].item())
        n_batches = int(stats[1].item())
        return total_loss / max(n_batches, 1)

    def _apply_optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def save_checkpoint(self, path: Path) -> None:
        if self.distributed.is_main_process():
            self._save_checkpoint(path)

    def _checkpoint_state(self) -> Dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "config": self.config,
        }

    def _save_checkpoint(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), path)
        return path

    def _resolve_checkpoint_name(self, filename: str) -> Path:
        return self.output_dir / filename

    def _update_checkpoints(self, train_loss: float, val_loss: Optional[float]) -> None:
        monitor = val_loss if val_loss is not None else train_loss
        if monitor is None:
            return
        improved = False
        if self._best_metric is None or (
            monitor < self._best_metric - float(self.config.early_stopping_min_delta)
        ):
            self._best_metric = monitor
            self._early_stop_counter = 0
            if self.config.best_checkpoint_name:
                path = self._resolve_checkpoint_name(self.config.best_checkpoint_name)
                self.best_checkpoint_path = path
                if self.distributed.is_main_process():
                    self.best_checkpoint_path = self._save_checkpoint(path)
                    _emit_info(
                        "New best checkpoint saved to %s (loss=%.4f)" % (path, float(monitor))
                    )
            improved = True
        if not improved and val_loss is not None and self.config.early_stopping_patience > 0:
            self._early_stop_counter += 1
        if self.config.last_checkpoint_name:
            path = self._resolve_checkpoint_name(self.config.last_checkpoint_name)
            self.last_checkpoint_path = path
            if self.distributed.is_main_process():
                self.last_checkpoint_path = self._save_checkpoint(path)

    def _should_stop_early(self, val_loss: Optional[float]) -> bool:
        if val_loss is None or self.config.early_stopping_patience <= 0:
            return False
        return self._early_stop_counter >= self.config.early_stopping_patience


class TensorDataset(Dataset):
    """Tiny dataset wrapper around tensors."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


def _solve_least_squares(
    model: LinearAtomicBaseline,
    dataset: TensorDataset,
    *,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    output_dir: Path,
) -> Dict[str, float]:
    train_indices = list(train_indices)
    val_indices = list(val_indices)
    if not train_indices:
        raise ValueError("Least squares solver requires at least one training sample")
    train_inputs = dataset.inputs[train_indices]
    train_targets = dataset.targets[train_indices]
    _emit_info(
        "Fitting baseline with closed-form least squares on %d samples%s"
        % (
            len(train_indices),
            f" and {len(val_indices)} validation samples" if val_indices else "",
        )
    )
    with torch.no_grad():
        solution = torch.linalg.lstsq(
            train_inputs.to(torch.float64), train_targets.to(torch.float64).unsqueeze(-1)
        ).solution.squeeze(-1)
        model.linear.weight.data.copy_(solution.to(train_inputs.dtype).unsqueeze(0))
    history: Dict[str, float] = {}
    with torch.no_grad():
        train_predictions = model(train_inputs)
        train_loss = torch.mean((train_predictions - train_targets) ** 2).item()
        history["train/1"] = train_loss
        if val_indices:
            val_inputs = dataset.inputs[val_indices]
            val_targets = dataset.targets[val_indices]
            val_predictions = model(val_inputs)
            val_loss = torch.mean((val_predictions - val_targets) ** 2).item()
            history["val/1"] = val_loss
    message = f"Least squares fit | train: {train_loss:.4f}"
    if "val/1" in history:
        message += f" | val: {history['val/1']:.4f}"
    _emit_info(message)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_history(output_dir, history)
    return history


def train_baseline(
    formula_vectors: torch.Tensor,
    energies: torch.Tensor,
    *,
    species: Sequence[int],
    config: TrainingConfig,
) -> Trainer:
    dataset = TensorDataset(formula_vectors, energies)
    val_size = int(len(dataset) * config.validation_split)
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_indices = list(train_dataset.indices)  # type: ignore[attr-defined]
        val_indices = list(val_dataset.indices)  # type: ignore[attr-defined]
    else:
        train_dataset = dataset
        val_dataset = None
        train_indices = list(range(len(dataset)))
        val_indices = []
    baseline_config = LinearBaselineConfig(species=tuple(species))
    model = LinearAtomicBaseline(baseline_config)
    solver = getattr(config, "solver", "optimizer").lower()
    trainer = Trainer(model, config)
    if solver in {"least_squares", "ols", "linear_regression"}:
        if trainer.distributed.enabled and trainer.distributed.world_size > 1:
            if trainer.distributed.is_main_process():
                history = _solve_least_squares(
                    model,
                    dataset,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    output_dir=config.output_dir,
                )
            else:
                history = {}
            trainer.distributed.sync_module_state(trainer.model)
            history = trainer.distributed.broadcast_object(history)
        else:
            history = _solve_least_squares(
                model,
                dataset,
                train_indices=train_indices,
                val_indices=val_indices,
                output_dir=config.output_dir,
            )
        trainer.history = history
        train_loss_value = float(history.get("train/1", 0.0))
        val_loss_value = history.get("val/1")
        trainer._update_checkpoints(train_loss_value, val_loss_value)
        return trainer
    train_sampler = trainer.distributed.build_sampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    if val_dataset is not None:
        val_sampler = trainer.distributed.build_sampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
        )
    else:
        val_loader = None
    trainer.train(train_loader, val_loader=val_loader, train_sampler=train_sampler)
    return trainer


@dataclass
class PotentialTrainingConfig(TrainingConfig):
    """Configuration for potential energy/force training."""

    energy_weight: float = 1.0
    force_weight: float = 0.0
    predict_forces_directly: bool = False
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.best_checkpoint_name == "best.pt":
            self.best_checkpoint_name = "potential_best.pt"
        if self.last_checkpoint_name == "last.pt":
            self.last_checkpoint_name = "potential_last.pt"


class PotentialTrainer:
    """Trainer that optimizes potential models for energies and forces."""

    def __init__(
        self,
        model: nn.Module,
        config: PotentialTrainingConfig,
        *,
        baseline: Optional[LinearAtomicBaseline] = None,
        baseline_requires_grad: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        base_device = self._resolve_device(config.device)
        self.distributed: DistributedState = init_distributed(config.distributed, base_device)
        self.device = self.distributed.device
        self.model.to(self.device)
        self.scheduler: Optional[WarmupDecayScheduler] = None
        self.energy_loss = nn.MSELoss()
        self.force_loss = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = baseline
        self.baseline_trainable = baseline is not None and baseline_requires_grad
        self.history: Dict[str, float] = {}
        self.best_checkpoint_path: Optional[Path] = None
        self.last_checkpoint_path: Optional[Path] = None
        self._best_metric: Optional[float] = None
        self._early_stop_counter = 0
        if self.baseline is not None:
            self.baseline.to(self.device)
            if self.baseline_trainable:
                for param in self.baseline.parameters():
                    param.requires_grad_(True)
            else:
                self.baseline.eval()
                for param in self.baseline.parameters():
                    param.requires_grad_(False)
        self.ddp_model: Optional[nn.parallel.DistributedDataParallel]
        if self.distributed.enabled:
            wrapper = _PotentialDDPWrapper(self.model, self.baseline)
            ddp_kwargs = {
                "find_unused_parameters": config.distributed.find_unused_parameters,
                "broadcast_buffers": config.distributed.broadcast_buffers,
            }
            if self.device.type == "cuda":
                device_index = self.device.index if self.device.index is not None else 0
                ddp_kwargs["device_ids"] = [device_index]
                ddp_kwargs["output_device"] = device_index
            self.ddp_model = nn.parallel.DistributedDataParallel(wrapper, **ddp_kwargs)
            param_source = self.ddp_model.parameters()
        else:
            self.ddp_model = None
            param_source = list(self.model.parameters())
            if self.baseline is not None and self.baseline_trainable:
                param_source += list(self.baseline.parameters())
        self.optimizer = _build_optimizer(param_source, config)
        (
            self._amp_enabled,
            self._autocast_device,
            self._autocast_dtype,
            scaler_enabled,
        ) = _resolve_autocast_settings(self.device, config)
        self.scaler: Optional[_GradScalerBase]
        if self._amp_enabled and self._autocast_device == "cuda":
            self.scaler = _make_grad_scaler(enabled=scaler_enabled)
        else:
            self.scaler = None
        self._update_frequency = max(int(self.config.update_frequency), 1)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def train(
        self,
        dataloader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        train_sampler=None,
    ) -> Dict[str, float]:
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
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        self.scheduler = _maybe_build_scheduler(self.optimizer, self.config, steps_per_epoch)
        details.append(f"optimizer={self.config.optimizer}")
        if self.scheduler is not None:
            schedule_msg = f"scheduler={self.config.scheduler}"
            if self.config.warmup_steps > 0:
                schedule_msg += f" (warmup={self.config.warmup_steps})"
            details.append(schedule_msg)
        if self.distributed.enabled and self.distributed.world_size > 1:
            details.append(f"world size={self.distributed.world_size}")
        if self._update_frequency > 1:
            details.append(f"update frequency={self._update_frequency}")
        effective_batch = (
            self.config.batch_size
            * max(self.distributed.world_size, 1)
            * max(self._update_frequency, 1)
        )
        if effective_batch != self.config.batch_size:
            details.append(f"effective batch={effective_batch}")
        if details:
            summary += " (" + ", ".join(details) + ")"
        _emit_info(summary)
        log_interval = max(int(self.config.log_every), 1)
        for epoch in range(1, self.config.epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
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
            if self.scheduler is not None:
                history[f"lr/{epoch}"] = float(self.scheduler.get_last_lr()[0])
            val_loss = val_metrics["loss"] if val_metrics is not None else None
            self._update_checkpoints(train_metrics["loss"], val_loss)
            if self._should_stop_early(val_loss):
                _emit_info(
                    "Early stopping triggered after epoch %03d (best %.4f)"
                    % (
                        epoch,
                        self._best_metric if self._best_metric is not None else train_metrics["loss"],
                    )
                )
                break
        self.history = history
        _save_history(self.output_dir, history)
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> Dict[str, float]:
        module = self.ddp_model if self.ddp_model is not None else self.model
        module.train(mode=training)
        if self.baseline is not None:
            if self.baseline_trainable:
                self.baseline.train(mode=training)
            else:
                self.baseline.eval()
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0
        n_batches = 0
        pending_update = False
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        for step, batch in enumerate(dataloader, start=1):
            batch = {
                key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)
            }
            energies = batch["energies"]
            formula_vectors = batch["formula_vectors"]
            requires_force_grad = (
                self.config.force_weight > 0.0
                and not self.config.predict_forces_directly
                and batch.get("forces") is not None
            )
            positions = batch["positions"]
            if requires_force_grad:
                positions = positions.clone().detach().requires_grad_(True)
                batch["positions"] = positions
            if training and ((step - 1) % self._update_frequency == 0):
                self.optimizer.zero_grad(set_to_none=True)
            force_loss_tensor = torch.tensor(0.0, device=self.device)
            with torch.set_grad_enabled(training or requires_force_grad):
                with _autocast_context(
                    self._amp_enabled, self._autocast_device, self._autocast_dtype
                ):
                    baseline_energy = None
                    if self.baseline is not None:
                        baseline_training = self.baseline_trainable and training
                        baseline_ctx = nullcontext() if baseline_training else torch.no_grad()
                        with baseline_ctx:
                            baseline_energy = self.baseline(formula_vectors)
                        if baseline_energy is not None and not baseline_training:
                            baseline_energy = baseline_energy.detach()
                        target_energy = energies - baseline_energy
                    else:
                        target_energy = energies
                    output = self._forward_model(batch)
                    energy_pred = output.energy
                    energy_loss = self.energy_loss(energy_pred, target_energy)
                    raw_loss = self.config.energy_weight * energy_loss
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
                        force_loss_tensor = self.force_loss(predicted_forces, target_forces)
                        raw_loss = raw_loss + self.config.force_weight * force_loss_tensor
            energy_loss_value = float(energy_loss.detach().item())
            force_loss_value = float(force_loss_tensor.detach().item())
            loss_value = float(raw_loss.detach().item())
            loss = raw_loss
            if training and self._update_frequency > 1:
                loss = loss / float(self._update_frequency)
            if training:
                if self.scaler is not None:
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                else:
                    loss.backward()
                pending_update = True
                should_step = (step % self._update_frequency == 0)
                if total_batches is not None and step == total_batches:
                    should_step = True
                if should_step:
                    self._apply_optimizer_step()
                    pending_update = False
            total_loss += loss_value
            total_energy_loss += energy_loss_value
            total_force_loss += force_loss_value
            n_batches += 1
        if training and pending_update:
            self._apply_optimizer_step()
        reduce_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        stats = torch.tensor(
            [total_loss, total_energy_loss, total_force_loss, float(n_batches)],
            device=reduce_device,
            dtype=torch.float64,
        )
        stats = self.distributed.all_reduce(stats)
        total_loss = float(stats[0].item())
        total_energy_loss = float(stats[1].item())
        total_force_loss = float(stats[2].item())
        n_batches = int(stats[3].item())
        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "energy_loss": total_energy_loss / max(n_batches, 1),
        }
        if self.config.force_weight > 0.0:
            metrics["force_loss"] = total_force_loss / max(n_batches, 1)
        return metrics

    def _apply_optimizer_step(self) -> None:
        if self.config.max_grad_norm is not None:
            params_to_clip = list(self.model.parameters())
            if self.baseline is not None and self.baseline_trainable:
                params_to_clip += list(self.baseline.parameters())
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _forward_model(self, batch: Dict[str, torch.Tensor]) -> PotentialOutput:
        args = (
            batch["node_indices"],
            batch["positions"],
            batch["adjacency"],
            batch["mask"],
        )
        if self.ddp_model is not None:
            return self.ddp_model(*args)
        return self.model(*args)

    def save_checkpoint(self, path: Path) -> None:
        if self.distributed.is_main_process():
            self._save_checkpoint(path)

    def _checkpoint_state(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "model_state": self.model.state_dict(),
            "config": self.config,
        }
        if self.baseline is not None:
            state["baseline_state"] = self.baseline.state_dict()
            state["baseline_trainable"] = self.baseline_trainable
        return state

    def _save_checkpoint(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), path)
        return path

    def _resolve_checkpoint_name(self, filename: str) -> Path:
        return self.output_dir / filename

    def _update_checkpoints(self, train_loss: float, val_loss: Optional[float]) -> None:
        monitor = val_loss if val_loss is not None else train_loss
        if monitor is None:
            return
        improved = False
        if self._best_metric is None or (
            monitor < self._best_metric - float(self.config.early_stopping_min_delta)
        ):
            self._best_metric = monitor
            self._early_stop_counter = 0
            if self.config.best_checkpoint_name:
                path = self._resolve_checkpoint_name(self.config.best_checkpoint_name)
                self.best_checkpoint_path = path
                if self.distributed.is_main_process():
                    self.best_checkpoint_path = self._save_checkpoint(path)
                    _emit_info(
                        "New best checkpoint saved to %s (loss=%.4f)" % (path, float(monitor))
                    )
            improved = True
        if not improved and val_loss is not None and self.config.early_stopping_patience > 0:
            self._early_stop_counter += 1
        if self.config.last_checkpoint_name:
            path = self._resolve_checkpoint_name(self.config.last_checkpoint_name)
            self.last_checkpoint_path = path
            if self.distributed.is_main_process():
                self.last_checkpoint_path = self._save_checkpoint(path)

    def _should_stop_early(self, val_loss: Optional[float]) -> bool:
        if val_loss is None or self.config.early_stopping_patience <= 0:
            return False
        return self._early_stop_counter >= self.config.early_stopping_patience


def train_potential_model(
    dataset: MolecularGraphDataset,
    model: nn.Module,
    *,
    config: PotentialTrainingConfig,
    baseline: Optional[LinearAtomicBaseline] = None,
    baseline_requires_grad: bool = True,
) -> PotentialTrainer:
    val_size = int(len(dataset) * config.validation_split)
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None
    trainer = PotentialTrainer(
        model,
        config,
        baseline=baseline,
        baseline_requires_grad=baseline_requires_grad,
    )
    train_sampler = trainer.distributed.build_sampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_graphs,
        num_workers=config.num_workers,
    )
    if val_dataset is not None:
        val_sampler = trainer.distributed.build_sampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_graphs,
            num_workers=config.num_workers,
        )
    else:
        val_loader = None
    trainer.train(train_loader, val_loader=val_loader, train_sampler=train_sampler)
    return trainer


@dataclass
class PotentialTrainingConfig(TrainingConfig):
    """Configuration for potential energy/force training."""

    energy_weight: float = 1.0
    force_weight: float = 0.0
    predict_forces_directly: bool = False
    max_grad_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.best_checkpoint_name == "best.pt":
            self.best_checkpoint_name = "potential_best.pt"
        if self.last_checkpoint_name == "last.pt":
            self.last_checkpoint_name = "potential_last.pt"


class PotentialTrainer:
    """Trainer that optimizes potential models for energies and forces."""

    def __init__(
        self,
        model: nn.Module,
        config: PotentialTrainingConfig,
        *,
        baseline: Optional[LinearAtomicBaseline] = None,
        baseline_requires_grad: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        parameters = list(self.model.parameters())
        self.scheduler: Optional[WarmupDecayScheduler] = None
        self.energy_loss = nn.MSELoss()
        self.force_loss = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = baseline
        self.baseline_trainable = baseline is not None and baseline_requires_grad
        self.history: Dict[str, float] = {}
        self.best_checkpoint_path: Optional[Path] = None
        self.last_checkpoint_path: Optional[Path] = None
        self._best_metric: Optional[float] = None
        self._early_stop_counter = 0
        if self.baseline is not None:
            self.baseline.to(self.device)
            if self.baseline_trainable:
                for param in self.baseline.parameters():
                    param.requires_grad_(True)
                parameters.extend(self.baseline.parameters())
            else:
                self.baseline.eval()
                for param in self.baseline.parameters():
                    param.requires_grad_(False)
        self.optimizer = _build_optimizer(parameters, config)
        (
            self._amp_enabled,
            self._autocast_device,
            self._autocast_dtype,
            scaler_enabled,
        ) = _resolve_autocast_settings(self.device, config)
        self.scaler: Optional[_GradScalerBase]
        if self._amp_enabled and self._autocast_device == "cuda":
            self.scaler = _make_grad_scaler(enabled=scaler_enabled)
        else:
            self.scaler = None

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
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        self.scheduler = _maybe_build_scheduler(self.optimizer, self.config, steps_per_epoch)
        details.append(f"optimizer={self.config.optimizer}")
        if self.scheduler is not None:
            schedule_msg = f"scheduler={self.config.scheduler}"
            if self.config.warmup_steps > 0:
                schedule_msg += f" (warmup={self.config.warmup_steps})"
            details.append(schedule_msg)
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
            if self.scheduler is not None:
                history[f"lr/{epoch}"] = float(self.scheduler.get_last_lr()[0])
            val_loss = val_metrics["loss"] if val_metrics is not None else None
            self._update_checkpoints(train_metrics["loss"], val_loss)
            if self._should_stop_early(val_loss):
                _emit_info(
                    "Early stopping triggered after epoch %03d (best %.4f)"
                    % (
                        epoch,
                        self._best_metric if self._best_metric is not None else train_metrics["loss"],
                    )
                )
                break
        self.history = history
        _save_history(self.output_dir, history)
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> Dict[str, float]:
        self.model.train(mode=training)
        if self.baseline is not None:
            if self.baseline_trainable:
                self.baseline.train(mode=training)
            else:
                self.baseline.eval()
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            batch = {key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            energies = batch["energies"]
            formula_vectors = batch["formula_vectors"]
            requires_force_grad = (
                self.config.force_weight > 0.0
                and not self.config.predict_forces_directly
                and batch.get("forces") is not None
            )
            positions = batch["positions"]
            if requires_force_grad:
                positions = positions.clone().detach().requires_grad_(True)
                batch["positions"] = positions
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            force_loss_value = torch.tensor(0.0, device=self.device)
            with torch.set_grad_enabled(training or requires_force_grad):
                with _autocast_context(
                    self._amp_enabled, self._autocast_device, self._autocast_dtype
                ):
                    baseline_energy = None
                    if self.baseline is not None:
                        baseline_training = self.baseline_trainable and training
                        baseline_ctx = nullcontext() if baseline_training else torch.no_grad()
                        with baseline_ctx:
                            baseline_energy = self.baseline(formula_vectors)
                        if baseline_energy is not None and not baseline_training:
                            baseline_energy = baseline_energy.detach()
                        target_energy = energies - baseline_energy
                    else:
                        target_energy = energies
                    output = self._forward_model(batch)
                    energy_pred = output.energy
                    energy_loss = self.energy_loss(energy_pred, target_energy)
                    loss = self.config.energy_weight * energy_loss
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
                if self.scaler is not None:
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                    if self.config.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        params_to_clip = list(self.model.parameters())
                        if self.baseline is not None and self.baseline_trainable:
                            params_to_clip += list(self.baseline.parameters())
                        nn.utils.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.max_grad_norm is not None:
                        params_to_clip = list(self.model.parameters())
                        if self.baseline is not None and self.baseline_trainable:
                            params_to_clip += list(self.baseline.parameters())
                        nn.utils.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
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
        self._save_checkpoint(path)

    def _checkpoint_state(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "model_state": self.model.state_dict(),
            "config": self.config,
        }
        if self.baseline is not None:
            state["baseline_state"] = self.baseline.state_dict()
            state["baseline_trainable"] = self.baseline_trainable
        return state

    def _save_checkpoint(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), path)
        return path

    def _resolve_checkpoint_name(self, filename: str) -> Path:
        return self.output_dir / filename

    def _update_checkpoints(self, train_loss: float, val_loss: Optional[float]) -> None:
        monitor = val_loss if val_loss is not None else train_loss
        if monitor is None:
            return
        improved = False
        if self._best_metric is None or (
            monitor < self._best_metric - float(self.config.early_stopping_min_delta)
        ):
            self._best_metric = monitor
            self._early_stop_counter = 0
            if self.config.best_checkpoint_name:
                path = self._resolve_checkpoint_name(self.config.best_checkpoint_name)
                self.best_checkpoint_path = self._save_checkpoint(path)
                _emit_info(
                    "New best checkpoint saved to %s (loss=%.4f)" % (path, float(monitor))
                )
            improved = True
        if not improved and val_loss is not None and self.config.early_stopping_patience > 0:
            self._early_stop_counter += 1
        if self.config.last_checkpoint_name:
            path = self._resolve_checkpoint_name(self.config.last_checkpoint_name)
            self.last_checkpoint_path = self._save_checkpoint(path)

    def _should_stop_early(self, val_loss: Optional[float]) -> bool:
        if val_loss is None or self.config.early_stopping_patience <= 0:
            return False
        return self._early_stop_counter >= self.config.early_stopping_patience


def train_potential_model(
    dataset: MolecularGraphDataset,
    model: nn.Module,
    *,
    config: PotentialTrainingConfig,
    baseline: Optional[LinearAtomicBaseline] = None,
    baseline_requires_grad: bool = True,
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
    trainer = PotentialTrainer(
        model,
        config,
        baseline=baseline,
        baseline_requires_grad=baseline_requires_grad,
    )
    trainer.train(train_loader, val_loader=val_loader)
    return trainer
