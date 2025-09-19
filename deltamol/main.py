"""High level entry points for DeltaMol."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .config.manager import save_config
from .data.io import load_npz_dataset
from .models.baseline import build_formula_vector
from .training.pipeline import TrainingConfig, train_baseline


def run_baseline_training(
    dataset_path: Path,
    output_dir: Path,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-2,
    validation_split: float = 0.1,
) -> None:
    """Train the linear atomic baseline on a dataset."""

    dataset = load_npz_dataset(dataset_path)
    species = sorted({int(z) for atoms in dataset.atoms for z in atoms})
    formula_vectors = torch.stack(
        [build_formula_vector(atoms, species=species) for atoms in dataset.atoms]
    )
    energies = torch.tensor(dataset.energies, dtype=torch.float32)
    config = TrainingConfig(
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=validation_split,
    )
    trainer = train_baseline(formula_vectors, energies, species=species, config=config)
    trainer.save_checkpoint(output_dir / "baseline.pt")
    try:
        save_config(config, output_dir / "config.yaml")
    except ImportError as exc:
        print(f"Skipping config serialization: {exc}")


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point that dispatches to :mod:`deltamol.cli`."""

    from .cli import main as cli_main

    cli_main(list(argv) if argv is not None else None)


__all__ = ["main", "run_baseline_training"]
