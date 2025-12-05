"""High level entry points for DeltaMol."""
from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import TensorDataset

from .config.manager import save_config
from .data.io import load_dataset
from .models.baseline import build_formula_vector
from .evaluation.testing import evaluate_baseline_model, plot_predictions_vs_targets
from .training.pipeline import TrainingConfig, train_baseline
from .utils import is_main_process
from .utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _resolve_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
    """Normalise dtype inputs to a torch.dtype instance."""

    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    aliases = {"fp16": "float16", "fp32": "float32", "fp64": "float64", "bf16": "bfloat16"}
    resolved_name = aliases.get(name, name)
    resolved = getattr(torch, resolved_name, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown dtype '{dtype}'")
    return resolved


def run_baseline_training(
    dataset_path: Path,
    output_dir: Path,
    *,
    dataset_format: str | None = None,
    dataset_key_map: dict[str, str] | None = None,
    test_dataset: Path | None = None,
    test_dataset_format: str | None = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    update_frequency: Optional[int] = None,
    num_workers: Optional[int] = None,
    learning_rate: Optional[float] = None,
    validation_split: Optional[float] = None,
    test_split: Optional[float] = None,
    solver: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: Optional[float] = None,
    best_checkpoint_name: Optional[str] = None,
    last_checkpoint_name: Optional[str] = None,
    mixed_precision: Optional[bool] = None,
    autocast_dtype: Optional[str] = None,
    grad_scaler: Optional[bool] = None,
    dtype: torch.dtype | str | None = None,
    log_every_steps: Optional[int] = None,
    tensorboard: Optional[bool] = None,
    seed: Optional[int] = None,
    parameter_init: Optional[str] = None,
    resume_from: Optional[Path] = None,
    config: TrainingConfig | None = None,
) -> None:
    """Train the linear atomic baseline on a dataset."""

    configure_logging(output_dir)
    data_dtype = _resolve_dtype(dtype)
    dataset = load_dataset(dataset_path, format=dataset_format, key_map=dataset_key_map)
    if dataset.energies is None:
        raise ValueError("Baseline training requires energies in the dataset")
    species = sorted({int(z) for atoms in dataset.atoms for z in atoms})
    test_dataset_tensors: TensorDataset | None = None
    if test_dataset is not None:
        test_data = load_dataset(
            test_dataset,
            format=test_dataset_format or dataset_format,
            key_map=dataset_key_map,
        )
        if test_data.energies is None:
            raise ValueError("Test dataset must include energies for evaluation")
        unknown_species = {int(z) for atoms in test_data.atoms for z in atoms} - set(species)
        if unknown_species:
            raise ValueError(
                f"Test dataset contains species {sorted(unknown_species)} not seen in training"
            )
        test_formula_vectors = torch.stack(
            [build_formula_vector(atoms, species=species) for atoms in test_data.atoms]
        ).to(data_dtype)
        test_energies_tensor = torch.as_tensor(test_data.energies, dtype=data_dtype).squeeze(-1)
        test_dataset_tensors = TensorDataset(test_formula_vectors, test_energies_tensor)
    if is_main_process():
        LOGGER.info(
            "Starting baseline training on %d molecules with %d species",
            len(dataset.energies),
            len(species),
        )
    formula_vectors = torch.stack(
        [build_formula_vector(atoms, species=species) for atoms in dataset.atoms]
    ).to(data_dtype)
    energies = torch.as_tensor(dataset.energies, dtype=data_dtype).squeeze(-1)
    override_kwargs = {}
    if mixed_precision is not None:
        override_kwargs["mixed_precision"] = mixed_precision
    if autocast_dtype is not None:
        override_kwargs["autocast_dtype"] = autocast_dtype
    if grad_scaler is not None:
        override_kwargs["grad_scaler"] = grad_scaler
    if update_frequency is not None:
        override_kwargs["update_frequency"] = update_frequency
    if num_workers is not None:
        override_kwargs["num_workers"] = num_workers
    if log_every_steps is not None:
        override_kwargs["log_every_steps"] = log_every_steps
    if tensorboard is not None:
        override_kwargs["tensorboard"] = tensorboard
    if seed is not None:
        override_kwargs["seed"] = seed
    if parameter_init is not None:
        override_kwargs["parameter_init"] = parameter_init
    if resume_from is not None:
        override_kwargs["resume_from"] = resume_from
    if config is None:
        config = TrainingConfig(
            output_dir=output_dir,
            epochs=epochs if epochs is not None else 200,
            learning_rate=learning_rate if learning_rate is not None else 1e-2,
            batch_size=batch_size if batch_size is not None else 128,
            update_frequency=update_frequency if update_frequency is not None else 1,
            num_workers=num_workers if num_workers is not None else 0,
            validation_split=validation_split if validation_split is not None else 0.1,
            test_split=test_split if test_split is not None else 0.0,
            solver=solver if solver is not None else "optimizer",
            early_stopping_patience=(
                early_stopping_patience if early_stopping_patience is not None else 0
            ),
            early_stopping_min_delta=(
                early_stopping_min_delta if early_stopping_min_delta is not None else 0.0
            ),
            best_checkpoint_name=(best_checkpoint_name or "baseline_best.pt"),
            last_checkpoint_name=(last_checkpoint_name or "baseline_last.pt"),
            seed=seed if seed is not None else None,
            parameter_init=parameter_init,
            **override_kwargs,
        )
    else:
        effective_best_checkpoint = (
            best_checkpoint_name
            if best_checkpoint_name is not None
            else config.best_checkpoint_name
        )
        effective_last_checkpoint = (
            last_checkpoint_name
            if last_checkpoint_name is not None
            else config.last_checkpoint_name
        )
        if effective_best_checkpoint in {None, "", "best.pt"}:
            effective_best_checkpoint = "baseline_best.pt"
        if effective_last_checkpoint in {None, "", "last.pt"}:
            effective_last_checkpoint = "baseline_last.pt"

        config = replace(
            config,
            output_dir=output_dir,
            epochs=epochs if epochs is not None else config.epochs,
            learning_rate=learning_rate if learning_rate is not None else config.learning_rate,
            batch_size=batch_size if batch_size is not None else config.batch_size,
            update_frequency=(
                update_frequency if update_frequency is not None else config.update_frequency
            ),
            num_workers=num_workers if num_workers is not None else config.num_workers,
            validation_split=(
                validation_split
                if validation_split is not None
                else config.validation_split
            ),
            test_split=(test_split if test_split is not None else config.test_split),
            solver=solver if solver is not None else config.solver,
            early_stopping_patience=(
                early_stopping_patience
                if early_stopping_patience is not None
                else config.early_stopping_patience
            ),
            early_stopping_min_delta=(
                early_stopping_min_delta
                if early_stopping_min_delta is not None
                else config.early_stopping_min_delta
            ),
            best_checkpoint_name=effective_best_checkpoint,
            last_checkpoint_name=effective_last_checkpoint,
            seed=seed if seed is not None else config.seed,
            parameter_init=(
                parameter_init if parameter_init is not None else config.parameter_init
            ),
            resume_from=resume_from if resume_from is not None else config.resume_from,
            **override_kwargs,
        )
    trainer = train_baseline(formula_vectors, energies, species=species, config=config)
    if test_dataset_tensors is not None:
        test_metrics, predictions, targets = evaluate_baseline_model(
            trainer.model,
            test_dataset_tensors,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=trainer.device,
        )
        trainer.history.update({f"test/{name}": value for name, value in test_metrics.items()})
        plot_path = config.output_dir / "baseline_test_predictions.png"
        try:
            plot_predictions_vs_targets(
                predictions,
                targets,
                plot_path,
                title="Baseline predictions vs targets (test)",
            )
            if is_main_process():
                LOGGER.info("Saved baseline test scatter plot to %s", plot_path)
        except Exception as exc:  # pragma: no cover - best effort plotting
            if is_main_process():
                LOGGER.warning("Failed to save baseline test plot: %s", exc)
    if trainer.distributed.is_main_process():
        best_path = trainer.best_checkpoint_path
        last_path = trainer.last_checkpoint_path
        if best_path is not None:
            LOGGER.info("Best baseline checkpoint saved to %s", best_path)
        if last_path is not None and last_path != best_path:
            LOGGER.info("Last baseline checkpoint saved to %s", last_path)
        alias_source = best_path or last_path
        checkpoint_path = output_dir / "baseline.pt"
        if alias_source is not None:
            if Path(alias_source).resolve() != checkpoint_path.resolve():
                shutil.copy2(alias_source, checkpoint_path)
                LOGGER.info("Copied %s to %s", alias_source, checkpoint_path)
            else:
                LOGGER.info("Best baseline checkpoint already stored at %s", checkpoint_path)
        else:
            trainer.save_checkpoint(checkpoint_path)
            LOGGER.info("Saved baseline checkpoint to %s", checkpoint_path)
        try:
            config_path = output_dir / "config.yaml"
            save_config(config, config_path)
            LOGGER.info("Saved training configuration to %s", config_path)
        except ImportError as exc:
            LOGGER.warning("Skipping config serialization: %s", exc)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point that dispatches to :mod:`deltamol.cli`."""

    from .cli import main as cli_main

    cli_main(list(argv) if argv is not None else None)


__all__ = ["main", "run_baseline_training"]


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()
