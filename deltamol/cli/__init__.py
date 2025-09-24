"""Command line interface for DeltaMol workflows."""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from ..config.manager import load_config, save_config
from ..data.io import MolecularDataset, cache_descriptor_matrix, load_npz_dataset
from ..descriptors.acsf import build_acsf_descriptor
from ..descriptors.fchl19 import build_fchl19_descriptor
from ..descriptors.lmbtr import build_lmbtr_descriptor
from ..descriptors.slatm import build_slatm_descriptor
from ..descriptors.soap import build_soap_descriptor
from ..main import run_baseline_training
from ..models import (
    GCNConfig,
    GCNPotential,
    LinearAtomicBaseline,
    LinearBaselineConfig,
    TransformerConfig,
    TransformerPotential,
)
from ..training.configs import BaselineConfig, ModelConfig, PotentialExperimentConfig
from ..training.datasets import MolecularGraphDataset
from ..training.pipeline import TrainingConfig, train_potential_model
from ..utils.logging import configure_logging

_DESCRIPTOR_BUILDERS: Dict[str, Callable] = {
    "acsf": build_acsf_descriptor,
    "soap": build_soap_descriptor,
    "slatm": build_slatm_descriptor,
    "lmbtr": build_lmbtr_descriptor,
    "fchl19": build_fchl19_descriptor,
}

LOGGER = logging.getLogger(__name__)


def _resolve_species(dataset: MolecularDataset, explicit: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if explicit:
        return tuple(int(z) for z in explicit)
    species = {int(z) for atoms in dataset.atoms for z in atoms}
    return tuple(sorted(species))


def _build_potential_model(model_cfg: ModelConfig, species: Sequence[int]):
    species_tuple = tuple(int(z) for z in species)
    name = model_cfg.name.lower()
    if name == "gcn":
        config = GCNConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            use_coordinate_features=model_cfg.use_coordinate_features,
            predict_forces=model_cfg.predict_forces,
        )
        return GCNPotential(config)
    if name in {"transformer", "transformer-potential"}:
        config = TransformerConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            num_heads=model_cfg.num_heads,
            dropout=model_cfg.dropout,
            ffn_dim=model_cfg.ffn_dim,
            use_coordinate_features=model_cfg.use_coordinate_features,
            predict_forces=model_cfg.predict_forces,
        )
        return TransformerPotential(config)
    raise ValueError(f"Unsupported potential model '{model_cfg.name}'")


def _load_baseline(
    baseline_cfg: Optional[BaselineConfig], species: Sequence[int]
) -> Tuple[Optional[LinearAtomicBaseline], bool]:
    if baseline_cfg is None or baseline_cfg.checkpoint is None:
        return None, True
    resolved_species = tuple(
        int(z) for z in (baseline_cfg.species if baseline_cfg.species else species)
    )
    model = LinearAtomicBaseline(LinearBaselineConfig(species=resolved_species))
    checkpoint = torch.load(baseline_cfg.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model, bool(baseline_cfg.requires_grad)


def _train_baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config, TrainingConfig) if args.config else None
    grad_scaler = False if args.no_grad_scaler else None
    run_baseline_training(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
        solver=args.solver,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        best_checkpoint_name=args.best_checkpoint_name,
        last_checkpoint_name=args.last_checkpoint_name,
        mixed_precision=True if args.mixed_precision else None,
        autocast_dtype=args.precision_dtype,
        grad_scaler=grad_scaler,
        config=config,
    )
    config_path = training_cfg.output_dir / "experiment.yaml"
    save_config(resolved_experiment, config_path)
    LOGGER.info("Saved experiment configuration to %s", config_path)


def _train_potential(args: argparse.Namespace) -> None:
    experiment = load_config(args.config, PotentialExperimentConfig)
    dataset_path = args.dataset or experiment.dataset.path
    if dataset_path is None:
        raise ValueError("A dataset path must be provided via the CLI or configuration file")
    dataset_path = Path(dataset_path)
    dataset = load_npz_dataset(dataset_path)
    species = _resolve_species(dataset, experiment.dataset.species)
    graph_dataset = MolecularGraphDataset(
        dataset,
        species=species,
        cutoff=experiment.dataset.cutoff,
        dtype=experiment.dataset.dtype,
    )
    training_cfg = experiment.training
    overrides = {}
    if args.output is not None:
        overrides["output_dir"] = args.output
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.validation_split is not None:
        overrides["validation_split"] = args.validation_split
    if args.mixed_precision:
        overrides["mixed_precision"] = True
    if args.precision_dtype is not None:
        overrides["autocast_dtype"] = args.precision_dtype
    if args.no_grad_scaler:
        overrides["grad_scaler"] = False
    if overrides:
        if "output_dir" in overrides and not isinstance(overrides["output_dir"], Path):
            overrides["output_dir"] = Path(overrides["output_dir"])
        training_cfg = replace(training_cfg, **overrides)
    elif not isinstance(training_cfg.output_dir, Path):
        training_cfg = replace(training_cfg, output_dir=Path(training_cfg.output_dir))
    if training_cfg.output_dir is None:
        raise ValueError("Potential training configuration must define an output directory")
    configure_logging(training_cfg.output_dir)
    LOGGER.info("Training potential model using dataset at %s", dataset_path)
    model = _build_potential_model(experiment.model, species)
    baseline, baseline_trainable = _load_baseline(experiment.baseline, species)
    if baseline is not None and experiment.baseline is not None:
        LOGGER.info("Loaded baseline checkpoint from %s", experiment.baseline.checkpoint)
        if not baseline_trainable:
            LOGGER.info("Baseline parameters will remain frozen during potential training")
    trainer = train_potential_model(
        graph_dataset,
        model,
        config=training_cfg,
        baseline=baseline,
        baseline_requires_grad=baseline_trainable,
    )
    checkpoint_path = training_cfg.output_dir / "potential.pt"
    best_path = trainer.best_checkpoint_path
    last_path = trainer.last_checkpoint_path
    if best_path is not None:
        LOGGER.info("Best potential checkpoint saved to %s", best_path)
    if last_path is not None and last_path != best_path:
        LOGGER.info("Last potential checkpoint saved to %s", last_path)
    alias_source = last_path or best_path
    if alias_source is not None:
        if Path(alias_source).resolve() != checkpoint_path.resolve():
            shutil.copy2(alias_source, checkpoint_path)
            LOGGER.info("Copied %s to %s", alias_source, checkpoint_path)
        else:
            LOGGER.info("Best potential checkpoint already stored at %s", checkpoint_path)
    else:
        trainer.save_checkpoint(checkpoint_path)
        LOGGER.info("Saved potential checkpoint to %s", checkpoint_path)
    resolved_dataset_cfg = replace(experiment.dataset, path=dataset_path, species=species)
    resolved_model_cfg = replace(experiment.model)
    resolved_baseline_cfg = (
        replace(experiment.baseline, species=tuple(experiment.baseline.species or species))
        if experiment.baseline is not None
        else None
    )
    resolved_experiment = PotentialExperimentConfig(
        training=training_cfg,
        model=resolved_model_cfg,
        dataset=resolved_dataset_cfg,
        baseline=resolved_baseline_cfg,
    )
    config_path = training_cfg.output_dir / "experiment.yaml"
    save_config(resolved_experiment, config_path)
    LOGGER.info("Saved experiment configuration to %s", config_path)


def _cache_descriptors(args: argparse.Namespace) -> None:
    dataset = load_npz_dataset(args.dataset)
    builder = _DESCRIPTOR_BUILDERS[args.descriptor]
    descriptor = builder(dataset.atoms)
    try:
        from ase import Atoms
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("ase is required for descriptor generation") from exc

    processed = 0
    progress_mod = max(args.progress, 1)
    for idx, (numbers, coords) in enumerate(zip(dataset.atoms, dataset.coordinates)):
        atoms = Atoms(numbers=numbers, positions=coords)
        matrix = descriptor.create(atoms)
        cache_descriptor_matrix(args.cache, str(idx), matrix)
        processed += 1
        if processed % progress_mod == 0:
            print(f"Cached descriptors for {processed} molecules")
    print(f"Descriptor cache written to {args.cache}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltamol", description="DeltaMol command line interface")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train-baseline", help="Train the linear atomic baseline")
    train_parser.add_argument("dataset", type=Path, help="Path to the NPZ dataset")
    train_parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    train_parser.add_argument("--config", type=Path, help="YAML file with training overrides")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 200)")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 128)")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-2)")
    train_parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable automatic mixed precision during baseline optimisation",
    )
    train_parser.add_argument(
        "--precision-dtype",
        choices=["float16", "bfloat16", "fp16", "bf16"],
        default=None,
        help="Autocast dtype for mixed precision (default: float16 on CUDA, bfloat16 on CPU)",
    )
    train_parser.add_argument(
        "--no-grad-scaler",
        action="store_true",
        help="Disable gradient scaling when mixed precision is enabled",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Validation fraction (default: 0.1)",
    )
    train_parser.add_argument(
        "--solver",
        choices=["optimizer", "least_squares"],
        default=None,
        help="Solver for the baseline weights (default: optimizer)",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training after N epochs without validation improvement",
    )
    train_parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum change in validation loss to count as an improvement",
    )
    train_parser.add_argument(
        "--best-checkpoint-name",
        type=str,
        default=None,
        help="Filename for the best baseline checkpoint (default: baseline_best.pt)",
    )
    train_parser.add_argument(
        "--last-checkpoint-name",
        type=str,
        default=None,
        help="Filename for the most recent baseline checkpoint (default: baseline_last.pt)",
    )
    train_parser.set_defaults(func=_train_baseline)

    potential_parser = subcommands.add_parser(
        "train-potential", help="Train a neural potential with configurable settings"
    )
    potential_parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Path to the NPZ dataset (overrides dataset.path in the config)",
    )
    potential_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file describing dataset, model, and training parameters",
    )
    potential_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output directory defined in the config",
    )
    potential_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    potential_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    potential_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    potential_parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Override validation split",
    )
    potential_parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable automatic mixed precision for potential training",
    )
    potential_parser.add_argument(
        "--precision-dtype",
        choices=["float16", "bfloat16", "fp16", "bf16"],
        default=None,
        help="Autocast dtype for mixed precision (default: float16 on CUDA, bfloat16 on CPU)",
    )
    potential_parser.add_argument(
        "--no-grad-scaler",
        action="store_true",
        help="Disable gradient scaling during mixed precision runs",
    )
    potential_parser.set_defaults(func=_train_potential)

    descriptor_parser = subcommands.add_parser(
        "cache-descriptors", help="Generate and cache atomic descriptors"
    )
    descriptor_parser.add_argument("dataset", type=Path, help="Path to the NPZ dataset")
    descriptor_parser.add_argument(
        "--descriptor",
        choices=sorted(_DESCRIPTOR_BUILDERS.keys()),
        required=True,
        help="Descriptor family to generate",
    )
    descriptor_parser.add_argument(
        "--cache", type=Path, default=Path("descriptor_cache.h5"), help="Output HDF5 file"
    )
    descriptor_parser.add_argument(
        "--progress", type=int, default=100, help="Print a message every N processed molecules"
    )
    descriptor_parser.set_defaults(func=_cache_descriptors)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]
