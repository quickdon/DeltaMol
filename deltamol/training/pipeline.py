"""High level training orchestration utilities."""
from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import init as nn_init

try:  # pragma: no cover - optional dependency import guard
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard may be unavailable
    SummaryWriter = None  # type: ignore[assignment]

from ..models.baseline import LinearAtomicBaseline, LinearBaselineConfig
from ..models.potential import PotentialOutput
from .datasets import MolecularGraphDataset, collate_graphs
from ..utils.distributed import DistributedConfig, DistributedState, init_distributed, is_main_process
from ..utils.random import seed_everything, seed_worker

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


def _describe_device(device: torch.device) -> str:
    """Return a human readable representation of a torch device."""

    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        try:  # pragma: no cover - defensive against runtime CUDA issues
            name = torch.cuda.get_device_name(index)
        except Exception:  # pragma: no cover - fallback when querying fails
            name = "CUDA device"
        return f"cuda:{index} ({name})"
    if device.type == "mps":
        return "mps (Metal Performance Shaders)"
    return str(device)


def _format_int(value: int) -> str:
    """Format an integer with thousands separators."""

    return f"{value:,}"


def _format_lrs(optimizer: Optimizer) -> str:
    """Format the learning rates of all parameter groups."""

    if not optimizer.param_groups:
        return "n/a"
    return "/".join(f"{group['lr']:.6g}" for group in optimizer.param_groups)


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


def _initialise_parameters(module: nn.Module, strategy: Optional[str]) -> None:
    """Apply the requested parameter initialisation scheme to ``module``."""

    if not strategy:
        return
    strategy = strategy.lower()
    if strategy in {"default", "none"}:
        return
    supported = {
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "orthogonal",
        "zeros",
    }
    if strategy not in supported:
        raise ValueError(f"Unsupported parameter initialisation '{strategy}'")

    def initialise_module(submodule: nn.Module) -> None:
        if isinstance(submodule, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            weight = submodule.weight
            if strategy == "xavier_uniform":
                nn_init.xavier_uniform_(weight)
            elif strategy == "xavier_normal":
                nn_init.xavier_normal_(weight)
            elif strategy == "kaiming_uniform":
                nn_init.kaiming_uniform_(weight, nonlinearity="relu")
            elif strategy == "kaiming_normal":
                nn_init.kaiming_normal_(weight, nonlinearity="relu")
            elif strategy == "orthogonal":
                nn_init.orthogonal_(weight)
            elif strategy == "zeros":
                nn_init.zeros_(weight)
            if submodule.bias is not None:
                nn_init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Embedding):
            weight = submodule.weight
            if strategy == "xavier_uniform":
                nn_init.xavier_uniform_(weight)
            elif strategy == "xavier_normal":
                nn_init.xavier_normal_(weight)
            elif strategy == "kaiming_uniform":
                nn_init.kaiming_uniform_(weight, nonlinearity="relu")
            elif strategy == "kaiming_normal":
                nn_init.kaiming_normal_(weight, nonlinearity="relu")
            elif strategy == "orthogonal":
                nn_init.orthogonal_(weight)
            elif strategy == "zeros":
                nn_init.zeros_(weight)
        elif isinstance(submodule, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(submodule, "weight") and submodule.weight is not None:
                nn_init.ones_(submodule.weight)
            if hasattr(submodule, "bias") and submodule.bias is not None:
                nn_init.zeros_(submodule.bias)

    module.apply(initialise_module)


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
    log_every_steps: int = 100
    device: str = "auto"
    mixed_precision: bool = False
    autocast_dtype: str = "float16"
    grad_scaler: bool = True
    validation_split: float = 0.1
    seed: Optional[int] = None
    optimizer: str = "adam"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    momentum: float = 0.9
    nesterov: bool = False
    solver: str = "optimizer"
    parameter_init: Optional[str] = None
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
    resume_from: Optional[Path] = None
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    tensorboard: bool = True


class Trainer:
    """Simple trainer that optimizes a model with MSE loss."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        base_device = self._resolve_device(config.device)
        self.distributed: DistributedState = init_distributed(config.distributed, base_device)
        self.device = self.distributed.device
        self._seed_generator: Optional[torch.Generator]
        self._worker_init_fn: Optional[Callable[[int], None]]
        if config.seed is not None:
            self._seed_generator = seed_everything(config.seed, rank=self.distributed.rank)
            self._worker_init_fn = seed_worker
            _emit_info(
                "Seeded RNGs with base seed %d (rank offset %d)"
                % (int(config.seed), self.distributed.rank)
            )
        else:
            self._seed_generator = None
            self._worker_init_fn = None
        self.model.to(self.device)
        if config.parameter_init:
            _initialise_parameters(self.model, config.parameter_init)
            if self.distributed.enabled:
                self.distributed.sync_module_state(self.model)
            _emit_info(f"Applied parameter initialisation: {config.parameter_init}")
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
        self._global_batches = 0
        self._optimizer_steps = 0
        self._eval_batches = 0
        self._current_epoch = 0
        self._start_epoch = 0
        self._pending_scheduler_state: Optional[Dict[str, object]] = None
        self._summary_writer: Optional[SummaryWriter]
        self._tensorboard_dir: Optional[Path] = None
        if self.config.tensorboard and self.distributed.is_main_process():
            if SummaryWriter is None:
                _emit_info(
                    "TensorBoard logging requested but torch.utils.tensorboard is unavailable."
                )
                self._summary_writer = None
            else:
                log_dir = self.output_dir / "tensorboard"
                try:
                    self._summary_writer = SummaryWriter(log_dir=str(log_dir))
                except Exception as exc:  # pragma: no cover - defensive
                    self._summary_writer = None
                    _emit_info(f"Failed to initialise TensorBoard writer: {exc}")
                else:
                    self._tensorboard_dir = log_dir
                    _emit_info(f"TensorBoard events will be written to {log_dir}")
        else:
            self._summary_writer = None
        device_msg = f"Initialised trainer on {_describe_device(self.device)}"
        if self.distributed.enabled:
            device_msg += (
                f" | rank={self.distributed.rank} of {self.distributed.world_size}"
            )
        _emit_info(device_msg)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        _emit_info(
            "Model parameter counts | total=%s, trainable=%s"
            % (_format_int(total_params), _format_int(trainable_params))
        )
        schedule_name = self.config.scheduler if self.config.scheduler else "none"
        amp_state = "enabled" if self._amp_enabled else "disabled"
        _emit_info(
            "Hyperparameters | epochs=%d, batch_size=%d, lr=%g, update_freq=%d, "
            "optimizer=%s, scheduler=%s, mixed_precision=%s"
            % (
                self.config.epochs,
                self.config.batch_size,
                self.config.learning_rate,
                self._update_frequency,
                self.config.optimizer,
                schedule_name,
                amp_state,
            )
        )
        self._maybe_resume_from_checkpoint()

    @property
    def data_loader_generator(self) -> Optional[torch.Generator]:
        return self._seed_generator

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        return self._worker_init_fn

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
        _emit_info(
            "Dataloader setup | train_workers=%d, val_workers=%d"
            % (
                getattr(dataloader, "num_workers", getattr(dataloader, "_num_workers", 0)),
                getattr(val_loader, "num_workers", getattr(val_loader, "_num_workers", 0))
                if val_loader is not None
                else 0,
            )
        )
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        self.scheduler = _maybe_build_scheduler(self.optimizer, self.config, steps_per_epoch)
        if self._pending_scheduler_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(self._pending_scheduler_state)
            except Exception:  # pragma: no cover - defensive
                _emit_info("Scheduler state in checkpoint is incompatible; skipping scheduler resume")
            self._pending_scheduler_state = None
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
        if self._start_epoch > 0:
            summary += f", resuming from epoch {self._start_epoch}"
        _emit_info(summary)
        _emit_info("Entering training loop")
        start_time = perf_counter()
        last_train_loss = math.nan
        last_val_loss: Optional[float] = None
        log_interval = max(int(self.config.log_every), 1)
        start_epoch = int(self._start_epoch) + 1
        if start_epoch > self.config.epochs:
            _emit_info("All configured epochs have already been completed; skipping training loop")
            return self.history
        for epoch in range(start_epoch, self.config.epochs + 1):
            self._current_epoch = epoch
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_start = perf_counter()
            train_loss = self._run_epoch(dataloader, training=True, epoch=epoch)
            history[f"train/{epoch}"] = train_loss
            last_train_loss = train_loss
            val_loss: Optional[float] = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False, epoch=epoch)
                history[f"val/{epoch}"] = val_loss
                last_val_loss = val_loss
            epoch_duration = perf_counter() - epoch_start
            history[f"time/{epoch}"] = float(epoch_duration)
            self._log_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                self._log_scalar("loss/val", val_loss, epoch)
            self._log_scalar("time/epoch", float(epoch_duration), epoch)
            if epoch == 1 or epoch == self.config.epochs or epoch % log_interval == 0:
                message = f"Epoch {epoch:03d} | time={epoch_duration:.2f}s | train: {train_loss:.4f}"
                if val_loss is not None:
                    message += f" | val: {val_loss:.4f}"
                _emit_info(message)
            if self.scheduler is not None:
                lr_value = float(self.scheduler.get_last_lr()[0])
                history[f"lr/{epoch}"] = lr_value
                self._log_scalar("lr", lr_value, epoch)
            self._update_checkpoints(train_loss, val_loss)
            if self._should_stop_early(val_loss):
                _emit_info(
                    "Early stopping triggered after epoch %03d (best %.4f)"
                    % (epoch, self._best_metric if self._best_metric is not None else train_loss)
                )
                break
        self.history = history
        _save_history(self.output_dir, history)
        elapsed = perf_counter() - start_time
        duration = timedelta(seconds=float(elapsed))
        final_message = (
            "Training completed in %s | final train loss=%.4f"
            % (duration, last_train_loss if not math.isnan(last_train_loss) else float("nan"))
        )
        if last_val_loss is not None:
            final_message += f", final val loss={last_val_loss:.4f}"
        _emit_info(final_message)
        if self.best_checkpoint_path is not None:
            _emit_info(f"Best checkpoint: {self.best_checkpoint_path}")
        if self.last_checkpoint_path is not None:
            _emit_info(f"Last checkpoint: {self.last_checkpoint_path}")
        if self._summary_writer is not None:
            self._summary_writer.flush()
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool, epoch: int) -> float:
        module = self.ddp_model if self.ddp_model is not None else self.model
        module.train(mode=training)
        total_loss = 0.0
        n_batches = 0
        pending_update = False
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        log_interval_steps = max(int(self.config.log_every_steps), 1)
        for step, batch in enumerate(dataloader, start=1):
            batch_start = perf_counter()
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = int(targets.shape[0]) if hasattr(targets, "shape") else len(targets)
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
            optimizer_step = False
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
                    optimizer_step = True
            total_loss += loss_value
            n_batches += 1
            batch_time = perf_counter() - batch_start
            lr_message = _format_lrs(self.optimizer)
            accum_step = (step - 1) % self._update_frequency + 1
            total_batches_str = f"/{total_batches:04d}" if total_batches is not None else ""
            phase = "train" if training else "eval"
            should_log = (
                step == 1
                or (total_batches is not None and step == total_batches)
                or (step % log_interval_steps == 0)
            )
            message = (
                f"[Epoch {epoch:03d}][{phase}][Batch {step:04d}{total_batches_str}] "
                f"loss={loss_value:.4f}, lr={lr_message}, batch_size={batch_size}, "
                f"time={batch_time:.3f}s"
            )
            if training:
                train_step_index = self._global_batches + 1
                self._log_scalar("loss/train_step", loss_value, train_step_index)
                message += (
                    f", accum={accum_step}/{self._update_frequency}, "
                    f"optimizer_steps={self._optimizer_steps}, "
                    f"global_batch={train_step_index}"
                )
                message += ", optimizer_step=yes" if optimizer_step else ", optimizer_step=no"
            else:
                eval_step_index = self._eval_batches + 1
                self._log_scalar("loss/val_step", loss_value, eval_step_index)
            if should_log:
                _emit_info(message)
            if training:
                self._global_batches += 1
            else:
                self._eval_batches += 1
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

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._summary_writer is None:
            return
        try:
            self._summary_writer.add_scalar(tag, value, step)
        except Exception:  # pragma: no cover - defensive guard
            pass

    def _apply_optimizer_step(self) -> None:
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._optimizer_steps += 1

    def save_checkpoint(self, path: Path) -> None:
        if self.distributed.is_main_process():
            self._save_checkpoint(path)

    def _checkpoint_state(self) -> Dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "config": self.config,
            "epoch": self._current_epoch,
            "global_batches": self._global_batches,
            "optimizer_steps": self._optimizer_steps,
            "eval_batches": self._eval_batches,
            "best_metric": self._best_metric,
            "early_stop_counter": self._early_stop_counter,
            "history": dict(self.history),
        }

    def _save_checkpoint(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), path)
        return path

    def _resolve_checkpoint_name(self, filename: str) -> Path:
        return self.output_dir / filename

    def _resolve_resume_checkpoint(self, resume_from: Path) -> Path:
        path = Path(resume_from)
        if path.is_file():
            return path
        if not path.exists():
            raise FileNotFoundError(f"Resume path '{resume_from}' does not exist")
        candidates = []
        if self.config.last_checkpoint_name:
            candidates.append(path / self.config.last_checkpoint_name)
        if self.config.best_checkpoint_name:
            candidates.append(path / self.config.best_checkpoint_name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint found in '{resume_from}'; checked {[c.name for c in candidates]}"
        )

    def _apply_checkpoint_state(self, state: Dict[str, object], *, load_scheduler: bool = True) -> None:
        model_state = state.get("model_state")
        if model_state is not None:
            self.model.load_state_dict(model_state)  # type: ignore[arg-type]
            if self.distributed.enabled:
                self.distributed.sync_module_state(self.model)
        if self.baseline is not None:
            baseline_state = state.get("baseline_state")
            if baseline_state is not None:
                try:
                    self.baseline.load_state_dict(baseline_state)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - defensive
                    _emit_info("Baseline state in checkpoint is incompatible; skipping baseline resume")
            self.baseline_trainable = bool(state.get("baseline_trainable", self.baseline_trainable))
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]
            except ValueError:
                _emit_info("Optimizer state in checkpoint is incompatible; skipping optimizer resume")
        scaler_state = state.get("scaler_state")
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                _emit_info("AMP scaler state in checkpoint is incompatible; skipping scaler resume")
        scheduler_state = state.get("scheduler_state")
        if load_scheduler and scheduler_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                _emit_info("Scheduler state in checkpoint is incompatible; skipping scheduler resume")
        else:
            self._pending_scheduler_state = scheduler_state if scheduler_state is not None else None
        self._start_epoch = int(state.get("epoch", 0))
        self._current_epoch = self._start_epoch
        self._global_batches = int(state.get("global_batches", self._global_batches))
        self._optimizer_steps = int(state.get("optimizer_steps", self._optimizer_steps))
        self._eval_batches = int(state.get("eval_batches", self._eval_batches))
        self._best_metric = state.get("best_metric")  # type: ignore[assignment]
        self._early_stop_counter = int(state.get("early_stop_counter", self._early_stop_counter))
        history = state.get("history")
        if isinstance(history, dict):
            self.history.update(history)
        if self.config.best_checkpoint_name:
            best_candidate = self.output_dir / self.config.best_checkpoint_name
            if best_candidate.exists():
                self.best_checkpoint_path = best_candidate
        if self.config.last_checkpoint_name:
            last_candidate = self.output_dir / self.config.last_checkpoint_name
            if last_candidate.exists():
                self.last_checkpoint_path = last_candidate

    def _maybe_resume_from_checkpoint(self) -> None:
        if self.config.resume_from is None:
            return
        resume_path = self._resolve_resume_checkpoint(self.config.resume_from)
        state = torch.load(resume_path, map_location=self.device)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint at {resume_path} is invalid or corrupted")
        self._apply_checkpoint_state(state, load_scheduler=False)
        _emit_info(
            f"Resuming potential training from {resume_path} at epoch {int(self._start_epoch)}"
        )

    def _resolve_resume_checkpoint(self, resume_from: Path) -> Path:
        path = Path(resume_from)
        if path.is_file():
            return path
        if not path.exists():
            raise FileNotFoundError(f"Resume path '{resume_from}' does not exist")
        candidates = []
        if self.config.last_checkpoint_name:
            candidates.append(path / self.config.last_checkpoint_name)
        if self.config.best_checkpoint_name:
            candidates.append(path / self.config.best_checkpoint_name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint found in '{resume_from}'; checked {[c.name for c in candidates]}"
        )

    def _apply_checkpoint_state(self, state: Dict[str, object], *, load_scheduler: bool = True) -> None:
        model_state = state.get("model_state")
        if model_state is not None:
            self.model.load_state_dict(model_state)  # type: ignore[arg-type]
            if self.distributed.enabled:
                self.distributed.sync_module_state(self.model)
        optimizer_state = state.get("optimizer_state")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)  # type: ignore[arg-type]
            except ValueError:
                _emit_info("Optimizer state in checkpoint is incompatible; skipping optimizer resume")
        scaler_state = state.get("scaler_state")
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                _emit_info("AMP scaler state in checkpoint is incompatible; skipping scaler resume")
        scheduler_state = state.get("scheduler_state")
        if load_scheduler and scheduler_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                _emit_info("Scheduler state in checkpoint is incompatible; skipping scheduler resume")
        else:
            self._pending_scheduler_state = scheduler_state if scheduler_state is not None else None
        self._start_epoch = int(state.get("epoch", 0))
        self._current_epoch = self._start_epoch
        self._global_batches = int(state.get("global_batches", self._global_batches))
        self._optimizer_steps = int(state.get("optimizer_steps", self._optimizer_steps))
        self._eval_batches = int(state.get("eval_batches", self._eval_batches))
        self._best_metric = state.get("best_metric")  # type: ignore[assignment]
        self._early_stop_counter = int(state.get("early_stop_counter", self._early_stop_counter))
        history = state.get("history")
        if isinstance(history, dict):
            self.history.update(history)
        if self.config.best_checkpoint_name:
            best_candidate = self.output_dir / self.config.best_checkpoint_name
            if best_candidate.exists():
                self.best_checkpoint_path = best_candidate
        if self.config.last_checkpoint_name:
            last_candidate = self.output_dir / self.config.last_checkpoint_name
            if last_candidate.exists():
                self.last_checkpoint_path = last_candidate

    def _maybe_resume_from_checkpoint(self) -> None:
        if self.config.resume_from is None:
            return
        resume_path = self._resolve_resume_checkpoint(self.config.resume_from)
        state = torch.load(resume_path, map_location=self.device)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint at {resume_path} is invalid or corrupted")
        self._apply_checkpoint_state(state, load_scheduler=False)
        _emit_info(
            f"Resuming training from {resume_path} at epoch {int(self._start_epoch)}"
        )

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
    split_generator: Optional[torch.Generator] = None
    if config.seed is not None:
        split_generator = torch.Generator()
        split_generator.manual_seed(int(config.seed))
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=split_generator
        )
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
    if solver in {"least_squares", "ols", "linear_regression"}:
        if config.device.lower() != "cpu":
            _emit_info(
                "Least squares solver requested; overriding device '%s' with CPU"
                % config.device
            )
            config.device = "cpu"
        if getattr(config, "mixed_precision", False):
            _emit_info("Disabling mixed precision for least squares solver")
            config.mixed_precision = False
        if config.tensorboard:
            _emit_info("TensorBoard logging disabled for least squares solver")
            config.tensorboard = False
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
        worker_init_fn=trainer.worker_init_fn,
        generator=trainer.data_loader_generator,
    )
    if val_dataset is not None:
        val_sampler = trainer.distributed.build_sampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            worker_init_fn=trainer.worker_init_fn,
            generator=trainer.data_loader_generator,
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
        residual_mode: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.residual_mode = residual_mode
        base_device = self._resolve_device(config.device)
        self.distributed: DistributedState = init_distributed(config.distributed, base_device)
        self.device = self.distributed.device
        self._seed_generator: Optional[torch.Generator]
        self._worker_init_fn: Optional[Callable[[int], None]]
        if config.seed is not None:
            self._seed_generator = seed_everything(config.seed, rank=self.distributed.rank)
            self._worker_init_fn = seed_worker
            _emit_info(
                "Seeded RNGs with base seed %d (rank offset %d)"
                % (int(config.seed), self.distributed.rank)
            )
        else:
            self._seed_generator = None
            self._worker_init_fn = None
        self.model.to(self.device)
        if config.parameter_init:
            _initialise_parameters(self.model, config.parameter_init)
            if self.distributed.enabled:
                self.distributed.sync_module_state(self.model)
            _emit_info(f"Applied parameter initialisation: {config.parameter_init}")
        self.scheduler: Optional[WarmupDecayScheduler] = None
        self.energy_loss = nn.MSELoss()
        self.force_loss = nn.MSELoss()
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline = baseline
        self.baseline_trainable = baseline is not None and baseline_requires_grad
        self._residual_mode_logged = False
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
        self._global_batches = 0
        self._optimizer_steps = 0
        self._eval_batches = 0
        self._current_epoch = 0
        self._start_epoch = 0
        self._pending_scheduler_state: Optional[Dict[str, object]] = None
        self._summary_writer: Optional[SummaryWriter]
        self._tensorboard_dir: Optional[Path] = None
        if self.config.tensorboard and self.distributed.is_main_process():
            if SummaryWriter is None:
                _emit_info(
                    "TensorBoard logging requested but torch.utils.tensorboard is unavailable."
                )
                self._summary_writer = None
            else:
                log_dir = self.output_dir / "tensorboard"
                try:
                    self._summary_writer = SummaryWriter(log_dir=str(log_dir))
                except Exception as exc:  # pragma: no cover - defensive
                    self._summary_writer = None
                    _emit_info(f"Failed to initialise TensorBoard writer: {exc}")
                else:
                    self._tensorboard_dir = log_dir
                    _emit_info(f"TensorBoard events will be written to {log_dir}")
        else:
            self._summary_writer = None
        device_msg = f"Initialised potential trainer on {_describe_device(self.device)}"
        if self.distributed.enabled:
            device_msg += (
                f" | rank={self.distributed.rank} of {self.distributed.world_size}"
            )
        _emit_info(device_msg)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        _emit_info(
            "Potential model parameters | total=%s, trainable=%s"
            % (_format_int(total_params), _format_int(trainable_params))
        )
        if self.baseline is not None:
            baseline_total = sum(p.numel() for p in self.baseline.parameters())
            baseline_trainable = sum(
                p.numel() for p in self.baseline.parameters() if p.requires_grad
            )
            mode = "trainable" if self.baseline_trainable else "frozen"
            _emit_info(
                "Baseline parameters (%s) | total=%s, trainable=%s"
                % (mode, _format_int(baseline_total), _format_int(baseline_trainable))
            )
        schedule_name = self.config.scheduler if self.config.scheduler else "none"
        amp_state = "enabled" if self._amp_enabled else "disabled"
        _emit_info(
            "Hyperparameters | epochs=%d, batch_size=%d, lr=%g, update_freq=%d, "
            "optimizer=%s, scheduler=%s, mixed_precision=%s"
            % (
                self.config.epochs,
                self.config.batch_size,
                self.config.learning_rate,
                self._update_frequency,
                self.config.optimizer,
                schedule_name,
                amp_state,
            )
        )
        self._maybe_resume_from_checkpoint()

    @property
    def data_loader_generator(self) -> Optional[torch.Generator]:
        return self._seed_generator

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        return self._worker_init_fn

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
        _emit_info(
            "Dataloader setup | train_workers=%d, val_workers=%d"
            % (
                getattr(dataloader, "num_workers", getattr(dataloader, "_num_workers", 0)),
                getattr(val_loader, "num_workers", getattr(val_loader, "_num_workers", 0))
                if val_loader is not None
                else 0,
            )
        )
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
        if self.baseline is not None:
            mode_label = (
                "residual energies (E - baseline)" if self.residual_mode else "absolute energies"
            )
            baseline_label = "trainable" if self.baseline_trainable else "frozen"
            details.append(f"baseline={baseline_label}, targets={mode_label}")
        try:
            steps_per_epoch = len(dataloader)
        except TypeError:
            steps_per_epoch = None
        self.scheduler = _maybe_build_scheduler(self.optimizer, self.config, steps_per_epoch)
        if self._pending_scheduler_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(self._pending_scheduler_state)
            except Exception:  # pragma: no cover - defensive
                _emit_info("Scheduler state in checkpoint is incompatible; skipping scheduler resume")
            self._pending_scheduler_state = None
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
        if self._start_epoch > 0:
            details.append(f"resuming from epoch {self._start_epoch}")
        if details:
            summary += " (" + ", ".join(details) + ")"
        _emit_info(summary)
        _emit_info("Entering training loop")
        start_time = perf_counter()
        last_train_metrics: Dict[str, float] = {}
        last_val_metrics: Optional[Dict[str, float]] = None
        log_interval = max(int(self.config.log_every), 1)
        start_epoch = int(self._start_epoch) + 1
        if start_epoch > self.config.epochs:
            _emit_info("All configured epochs have already been completed; skipping training loop")
            return self.history
        for epoch in range(start_epoch, self.config.epochs + 1):
            self._current_epoch = epoch
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_start = perf_counter()
            train_metrics = self._run_epoch(dataloader, training=True, epoch=epoch)
            history[f"train/{epoch}"] = train_metrics["loss"]
            if "energy_loss" in train_metrics:
                history[f"train_energy/{epoch}"] = train_metrics["energy_loss"]
            if "force_loss" in train_metrics:
                history[f"train_force/{epoch}"] = train_metrics["force_loss"]
            last_train_metrics = train_metrics
            val_metrics: Optional[Dict[str, float]] = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
                history[f"val/{epoch}"] = val_metrics["loss"]
                if "energy_loss" in val_metrics:
                    history[f"val_energy/{epoch}"] = val_metrics["energy_loss"]
                if "force_loss" in val_metrics:
                    history[f"val_force/{epoch}"] = val_metrics["force_loss"]
                last_val_metrics = val_metrics
            epoch_duration = perf_counter() - epoch_start
            history[f"time/{epoch}"] = float(epoch_duration)
            self._log_scalar("loss/train", train_metrics["loss"], epoch)
            if "energy_loss" in train_metrics:
                self._log_scalar("loss/train_energy", train_metrics["energy_loss"], epoch)
            if "force_loss" in train_metrics:
                self._log_scalar("loss/train_force", train_metrics["force_loss"], epoch)
            if val_metrics is not None:
                self._log_scalar("loss/val", val_metrics["loss"], epoch)
                if "energy_loss" in val_metrics:
                    self._log_scalar("loss/val_energy", val_metrics["energy_loss"], epoch)
                if "force_loss" in val_metrics:
                    self._log_scalar("loss/val_force", val_metrics["force_loss"], epoch)
            self._log_scalar("time/epoch", float(epoch_duration), epoch)
            if epoch == 1 or epoch == self.config.epochs or epoch % log_interval == 0:
                message = (
                    f"Epoch {epoch:03d} | time={epoch_duration:.2f}s | "
                    f"train: {train_metrics['loss']:.4f}"
                )
                if val_metrics is not None:
                    message += f" | val: {val_metrics['loss']:.4f}"
                _emit_info(message)
            if self.scheduler is not None:
                lr_value = float(self.scheduler.get_last_lr()[0])
                history[f"lr/{epoch}"] = lr_value
                self._log_scalar("lr", lr_value, epoch)
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
        elapsed = perf_counter() - start_time
        duration = timedelta(seconds=float(elapsed))
        summary_msg = (
            "Training completed in %s | final train loss=%.4f"
            % (duration, last_train_metrics.get("loss", float("nan")))
        )
        if last_val_metrics is not None:
            summary_msg += f", final val loss={last_val_metrics['loss']:.4f}"
        if "energy_loss" in last_train_metrics:
            summary_msg += f", energy_loss={last_train_metrics['energy_loss']:.4f}"
        if "force_loss" in last_train_metrics:
            summary_msg += f", force_loss={last_train_metrics['force_loss']:.4f}"
        _emit_info(summary_msg)
        if self.best_checkpoint_path is not None:
            _emit_info(f"Best checkpoint: {self.best_checkpoint_path}")
        if self.last_checkpoint_path is not None:
            _emit_info(f"Last checkpoint: {self.last_checkpoint_path}")
        if self._summary_writer is not None:
            self._summary_writer.flush()
        return history

    def _run_epoch(self, dataloader: DataLoader, *, training: bool, epoch: int) -> Dict[str, float]:
        module = self.ddp_model if self.ddp_model is not None else self.model
        module.train(mode=training)
        if self.baseline is not None:
            if self.baseline_trainable:
                self.baseline.train(mode=training)
            else:
                self.baseline.eval()
        if self.baseline is not None and not self._residual_mode_logged:
            mode_msg = (
                "Using residual targets: energies - baseline"
                if self.residual_mode
                else "Using absolute energy targets without subtracting the baseline"
            )
            _emit_info(mode_msg)
            self._residual_mode_logged = True
        total_loss = 0.0
        total_energy_loss = 0.0
        total_force_loss = 0.0
        n_batches = 0
        pending_update = False
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        log_interval_steps = max(int(self.config.log_every_steps), 1)
        for step, batch in enumerate(dataloader, start=1):
            batch_start = perf_counter()
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
                positions = positions.clone().requires_grad_(True)
                batch["positions"] = positions
            if training and ((step - 1) % self._update_frequency == 0):
                self.optimizer.zero_grad(set_to_none=True)
            force_loss_tensor = torch.tensor(0.0, device=self.device)
            with torch.set_grad_enabled(training or requires_force_grad):
                with _autocast_context(
                    self._amp_enabled, self._autocast_device, self._autocast_dtype
                ):
                    baseline_energy = None
                    if self.baseline is not None and self.residual_mode:
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
            optimizer_step = False
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
                    optimizer_step = True
            total_loss += loss_value
            total_energy_loss += energy_loss_value
            total_force_loss += force_loss_value
            n_batches += 1
            batch_time = perf_counter() - batch_start
            lr_message = _format_lrs(self.optimizer)
            accum_step = (step - 1) % self._update_frequency + 1
            total_batches_str = f"/{total_batches:04d}" if total_batches is not None else ""
            phase = "train" if training else "eval"
            batch_size = int(energies.shape[0])
            should_log = (
                step == 1
                or (total_batches is not None and step == total_batches)
                or (step % log_interval_steps == 0)
            )
            message = (
                f"[Epoch {epoch:03d}][{phase}][Batch {step:04d}{total_batches_str}] "
                f"loss={loss_value:.4f}, energy_loss={energy_loss_value:.4f}, "
                f"lr={lr_message}, batch_size={batch_size}, time={batch_time:.3f}s"
            )
            if self.config.force_weight > 0.0:
                message += f", force_loss={force_loss_value:.4f}"
            if training:
                train_step_index = self._global_batches + 1
                self._log_scalar("loss/train_step", loss_value, train_step_index)
                self._log_scalar("loss/train_energy_step", energy_loss_value, train_step_index)
                if self.config.force_weight > 0.0:
                    self._log_scalar("loss/train_force_step", force_loss_value, train_step_index)
                message += (
                    f", accum={accum_step}/{self._update_frequency}, "
                    f"optimizer_steps={self._optimizer_steps}, "
                    f"global_batch={train_step_index}"
                )
                message += ", optimizer_step=yes" if optimizer_step else ", optimizer_step=no"
            else:
                eval_step_index = self._eval_batches + 1
                self._log_scalar("loss/val_step", loss_value, eval_step_index)
                self._log_scalar("loss/val_energy_step", energy_loss_value, eval_step_index)
                if self.config.force_weight > 0.0:
                    self._log_scalar("loss/val_force_step", force_loss_value, eval_step_index)
            if should_log:
                _emit_info(message)
            if training:
                self._global_batches += 1
            else:
                self._eval_batches += 1
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

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._summary_writer is None:
            return
        try:
            self._summary_writer.add_scalar(tag, value, step)
        except Exception:  # pragma: no cover - defensive guard
            pass

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
        self._optimizer_steps += 1

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
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "config": self.config,
            "epoch": self._current_epoch,
            "global_batches": self._global_batches,
            "optimizer_steps": self._optimizer_steps,
            "eval_batches": self._eval_batches,
            "best_metric": self._best_metric,
            "early_stop_counter": self._early_stop_counter,
            "history": dict(self.history),
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
    residual_mode: bool = True,
) -> PotentialTrainer:
    val_size = int(len(dataset) * config.validation_split)
    split_generator: Optional[torch.Generator] = None
    if config.seed is not None:
        split_generator = torch.Generator()
        split_generator.manual_seed(int(config.seed))
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=split_generator
        )
    else:
        train_dataset = dataset
        val_dataset = None
    trainer = PotentialTrainer(
        model,
        config,
        baseline=baseline,
        baseline_requires_grad=baseline_requires_grad,
        residual_mode=residual_mode,
    )
    train_sampler = trainer.distributed.build_sampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_graphs,
        num_workers=config.num_workers,
        worker_init_fn=trainer.worker_init_fn,
        generator=trainer.data_loader_generator,
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
            worker_init_fn=trainer.worker_init_fn,
            generator=trainer.data_loader_generator,
        )
    else:
        val_loader = None
    trainer.train(train_loader, val_loader=val_loader, train_sampler=train_sampler)
    return trainer
