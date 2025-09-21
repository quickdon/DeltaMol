"""High level entry points for DeltaMol."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import torch

from .config.manager import save_config
from .data.io import load_npz_dataset
from .models.baseline import build_formula_vector
from .training.pipeline import TrainingConfig, train_baseline
from .utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_baseline_training(
    dataset_path: Path,
    output_dir: Path,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    validation_split: Optional[float] = None,
    config: TrainingConfig | None = None,
) -> None:
    """Train the linear atomic baseline on a dataset."""

    configure_logging(output_dir)
    dataset = load_npz_dataset(dataset_path)
    species = sorted({int(z) for atoms in dataset.atoms for z in atoms})
    LOGGER.info(
        "Starting baseline training on %d molecules with %d species",
        len(dataset.energies),
        len(species),
    )
    formula_vectors = torch.stack(
        [build_formula_vector(atoms, species=species) for atoms in dataset.atoms]
    )
    energies = torch.tensor(dataset.energies, dtype=torch.float32)
    if config is None:
        config = TrainingConfig(
            output_dir=output_dir,
            epochs=epochs if epochs is not None else 200,
            learning_rate=learning_rate if learning_rate is not None else 1e-2,
            batch_size=batch_size if batch_size is not None else 128,
            validation_split=validation_split if validation_split is not None else 0.1,
        )
    else:
        config = replace(
            config,
            output_dir=output_dir,
            epochs=epochs if epochs is not None else config.epochs,
            learning_rate=learning_rate if learning_rate is not None else config.learning_rate,
            batch_size=batch_size if batch_size is not None else config.batch_size,
            validation_split=(
                validation_split
                if validation_split is not None
                else config.validation_split
            ),
        )
    trainer = train_baseline(formula_vectors, energies, species=species, config=config)
    checkpoint_path = output_dir / "baseline.pt"
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
