"""Command line interface for DeltaMol workflows."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import TensorDataset

from ..config.manager import load_config, save_config
from ..data.io import (
    MolecularDataset,
    cache_descriptor_matrix,
    load_dataset,
    load_npz_dataset,
)
from ..descriptors.acsf import build_acsf_descriptor
from ..descriptors.fchl19 import build_fchl19_descriptor
from ..descriptors.lmbtr import build_lmbtr_descriptor
from ..descriptors.slatm import build_slatm_descriptor
from ..descriptors.soap import build_soap_descriptor
from ..main import _resolve_dtype, run_baseline_training
from ..models import (
    PotentialModelAdapter,
    HybridPotential,
    HybridPotentialConfig,
    LinearAtomicBaseline,
    LinearBaselineConfig,
    SE3TransformerConfig,
    SE3TransformerPotential,
    build_formula_vector,
    load_external_model,
)
from ..evaluation.testing import (
    evaluate_baseline_model,
    evaluate_potential_model,
    plot_predictions_vs_targets,
    save_force_predictions_and_targets,
    save_predictions_and_targets,
)
from ..training.configs import BaselineConfig, ModelConfig, PotentialExperimentConfig
from ..training.datasets import MolecularGraphDataset
from ..training.pipeline import TrainingConfig, train_potential_model
from ..utils import is_main_process
from ..utils.logging import configure_logging

_DESCRIPTOR_BUILDERS: Dict[str, Callable] = {
    "acsf": build_acsf_descriptor,
    "soap": build_soap_descriptor,
    "slatm": build_slatm_descriptor,
    "lmbtr": build_lmbtr_descriptor,
    "fchl19": build_fchl19_descriptor,
}

_DATASET_FORMAT_CHOICES = (
    "auto",
    "npz",
    "npy",
    "json",
    "yaml",
    "yml",
    "pt",
    "pth",
)

LOGGER = logging.getLogger(__name__)


def _resolve_species(dataset: MolecularDataset, explicit: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if explicit:
        return tuple(int(z) for z in explicit)
    species = {int(z) for atoms in dataset.atoms for z in atoms}
    return tuple(sorted(species))


def _build_potential_model(model_cfg: ModelConfig, species: Sequence[int]):
    species_tuple = tuple(int(z) for z in species)
    name = model_cfg.name.lower()
    if name in {"transformer", "hybrid", "hybrid-potential", "soap-transformer"}:
        config = HybridPotentialConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            gcn_layers=model_cfg.gcn_layers,
            transformer_layers=model_cfg.transformer_layers,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            use_coordinate_features=model_cfg.use_coordinate_features,
            soap_num_radial=model_cfg.soap_num_radial,
            soap_cutoff=model_cfg.soap_cutoff,
            soap_gaussian_width=model_cfg.soap_gaussian_width,
            predict_forces=model_cfg.predict_forces,
        )
        return HybridPotential(config)
    if name in {"se3", "se3-transformer", "equivariant"}:
        config = SE3TransformerConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.se3_layers or model_cfg.transformer_layers,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            distance_embedding_dim=model_cfg.se3_distance_embedding,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return SE3TransformerPotential(config)
    if name == "gcn":
        config = HybridPotentialConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            gcn_layers=model_cfg.gcn_layers,
            transformer_layers=0,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            use_coordinate_features=model_cfg.use_coordinate_features,
            soap_num_radial=model_cfg.soap_num_radial,
            soap_cutoff=model_cfg.soap_cutoff,
            soap_gaussian_width=model_cfg.soap_gaussian_width,
            predict_forces=model_cfg.predict_forces,
        )
        return HybridPotential(config)
    if name == "external":
        if not model_cfg.adapter:
            raise ValueError("External model requires an 'adapter' path to be provided")
        try:
            external_model = load_external_model(model_cfg.adapter)
        except ImportError as exc:
            raise ImportError(
                "Failed to load external model. Install its dependencies or update the adapter path."
            ) from exc
        if model_cfg.adapter_weights is not None:
            checkpoint = torch.load(
                model_cfg.adapter_weights,
                map_location="cpu",
                weights_only=False,
            )
            try:
                external_model.load_state_dict(checkpoint)
            except Exception as exc:  # pragma: no cover - defensive for custom loaders
                raise RuntimeError(
                    f"Unable to load weights from {model_cfg.adapter_weights}: {exc}"
                ) from exc
        return PotentialModelAdapter(
            external_model,
            neighbor_strategy=model_cfg.neighbor_strategy,
            neighbor_cutoff=model_cfg.neighbor_cutoff or model_cfg.cutoff,
        )
    raise ValueError(f"Unsupported potential model '{model_cfg.name}'")


def _parse_dataset_key_overrides(values: Optional[Sequence[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not values:
        return mapping
    allowed = {"atoms", "coordinates", "energies", "forces"}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"Invalid dataset key override '{raw}'. Expected format CANONICAL=SOURCE"
            )
        canonical, source = raw.split("=", 1)
        canonical = canonical.strip().lower()
        source = source.strip()
        if canonical not in allowed:
            raise ValueError(
                f"Unsupported canonical key '{canonical}'. Choose from {sorted(allowed)}"
            )
        if not source:
            raise ValueError(f"Dataset key override '{raw}' is missing a source field")
        mapping[canonical] = source
    return mapping


def _load_checkpoint_state(path: Path) -> Dict[str, object]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return {"model_state": state}
    return state


def _save_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, output_dir: Path, prefix: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_predictions.pt"
    torch.save({"predictions": predictions, "targets": targets}, output_path)
    return output_path


def _save_force_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, output_dir: Path, prefix: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_force_predictions.pt"
    torch.save({"predictions": predictions, "targets": targets}, output_path)
    return output_path


def _load_baseline(
    baseline_cfg: Optional[BaselineConfig], species: Sequence[int]
) -> Tuple[Optional[LinearAtomicBaseline], bool]:
    if baseline_cfg is None or baseline_cfg.checkpoint is None:
        return None, True
    resolved_species = tuple(
        int(z) for z in (baseline_cfg.species if baseline_cfg.species else species)
    )
    model = LinearAtomicBaseline(LinearBaselineConfig(species=resolved_species))
    checkpoint = torch.load(
        baseline_cfg.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return model, bool(baseline_cfg.requires_grad)


def _predict_baseline(args: argparse.Namespace) -> None:
    try:
        dataset_key_map = _parse_dataset_key_overrides(args.dataset_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = load_dataset(
        args.dataset,
        format=args.dataset_format,
        key_map=dataset_key_map or None,
    )
    if dataset.energies is None:
        raise ValueError("Prediction dataset must include energies for evaluation and plotting")
    species = tuple(int(z) for z in (args.species or _resolve_species(dataset, None)))
    output_dir = args.output or args.checkpoint.parent
    configure_logging(output_dir)
    data_dtype = _resolve_dtype(args.dtype)
    formula_vectors = torch.stack(
        [build_formula_vector(atoms, species=species) for atoms in dataset.atoms]
    ).to(data_dtype)
    energies = torch.as_tensor(dataset.energies, dtype=data_dtype).squeeze(-1)
    checkpoint_state = _load_checkpoint_state(args.checkpoint)
    model_state = checkpoint_state.get("model_state", checkpoint_state)
    model = LinearAtomicBaseline(LinearBaselineConfig(species=species))
    model.load_state_dict(model_state)  # type: ignore[arg-type]
    device = torch.device(args.device) if args.device is not None else None
    if device is not None:
        model = model.to(device=device, dtype=data_dtype)
    dataset_tensors = TensorDataset(formula_vectors, energies)
    metrics, predictions, targets = evaluate_baseline_model(
        model,
        dataset_tensors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    metrics_path = output_dir / f"baseline_metrics_{args.dataset.stem}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_path = output_dir / f"baseline_predictions_{args.dataset.stem}.png"
    plot_predictions_vs_targets(
        predictions,
        targets,
        plot_path,
        title=f"Baseline predictions vs targets ({args.dataset.stem})",
    )
    predictions_path = _save_predictions(
        predictions, targets, output_dir, f"baseline_{args.dataset.stem}"
    )
    if is_main_process():
        LOGGER.info("Saved baseline metrics to %s", metrics_path)
        LOGGER.info("Saved baseline predictions to %s", predictions_path)
        LOGGER.info("Saved baseline scatter plot to %s", plot_path)


def _predict_potential(args: argparse.Namespace) -> None:
    experiment_path = args.experiment
    if experiment_path is None:
        default_config = args.checkpoint.parent / "experiment.yaml"
        if default_config.exists():
            experiment_path = default_config
    if experiment_path is None:
        raise ValueError("An experiment configuration is required to load the potential model")
    experiment = load_config(experiment_path, PotentialExperimentConfig)
    dataset_path = args.dataset
    if dataset_path is None:
        raise ValueError("A dataset path must be provided for prediction")
    config_key_map = dict(experiment.dataset.key_map or {})
    try:
        cli_key_map = _parse_dataset_key_overrides(args.dataset_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config_key_map.update(cli_key_map)
    dataset_format = (
        args.dataset_format if args.dataset_format is not None else experiment.dataset.format
    )
    dataset = load_dataset(
        dataset_path,
        format=dataset_format,
        key_map=config_key_map or None,
    )
    if dataset.energies is None:
        raise ValueError("Prediction dataset must include energies for evaluation and plotting")
    species = _resolve_species(dataset, experiment.dataset.species)
    graph_dataset = MolecularGraphDataset(
        dataset,
        species=species,
        cutoff=experiment.dataset.cutoff,
        dtype=experiment.dataset.dtype,
    )
    output_dir = args.output or args.checkpoint.parent
    configure_logging(output_dir)
    baseline_cfg = experiment.baseline
    if (
        baseline_cfg is not None
        and baseline_cfg.checkpoint is not None
        and not Path(baseline_cfg.checkpoint).is_absolute()
    ):
        baseline_cfg = replace(
            baseline_cfg, checkpoint=Path(experiment_path).parent / baseline_cfg.checkpoint
        )
    baseline, _ = _load_baseline(baseline_cfg, species)
    model = _build_potential_model(experiment.model, species)
    checkpoint_state = _load_checkpoint_state(args.checkpoint)
    model_state = checkpoint_state.get("model_state", checkpoint_state)
    model.load_state_dict(model_state)  # type: ignore[arg-type]
    device = torch.device(args.device) if args.device is not None else None
    if device is not None:
        model = model.to(device)
        if baseline is not None:
            baseline = baseline.to(device)
    batch_size = args.batch_size or experiment.training.batch_size
    num_workers = args.num_workers or experiment.training.num_workers
    residual_mode = experiment.model.residual_mode
    (
        metrics,
        predictions,
        targets,
        force_predictions,
        force_targets,
    ) = evaluate_potential_model(
        model,
        graph_dataset,
        baseline=baseline,
        residual_mode=residual_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    metrics_path = output_dir / f"potential_metrics_{dataset_path.stem}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_path = output_dir / f"potential_predictions_{dataset_path.stem}.png"
    plot_predictions_vs_targets(
        predictions,
        targets,
        plot_path,
        title=f"Potential predictions vs targets ({dataset_path.stem})",
    )
    predictions_path = _save_predictions(
        predictions, targets, output_dir, f"potential_{dataset_path.stem}"
    )
    if force_predictions is not None and force_targets is not None:
        _save_force_predictions(
            force_predictions, force_targets, output_dir, f"potential_{dataset_path.stem}"
        )
        save_force_predictions_and_targets(
            force_predictions,
            force_targets,
            output_dir / f"potential_force_results_{dataset_path.stem}.npz",
        )
    if is_main_process():
        LOGGER.info("Saved potential metrics to %s", metrics_path)
        LOGGER.info("Saved potential predictions to %s", predictions_path)
        LOGGER.info("Saved potential scatter plot to %s", plot_path)
        if force_predictions is not None and force_targets is not None:
            LOGGER.info(
                "Saved potential force predictions to %s and %s",
                output_dir / f"potential_{dataset_path.stem}_force_predictions.pt",
                output_dir / f"potential_force_results_{dataset_path.stem}.npz",
            )


def _train_baseline(args: argparse.Namespace) -> None:
    config = load_config(args.config, TrainingConfig) if args.config else None
    grad_scaler = False if args.no_grad_scaler else None
    parameter_init = None
    if args.parameter_init and args.parameter_init.lower() not in {"default", "none"}:
        parameter_init = args.parameter_init
    try:
        dataset_key_map = _parse_dataset_key_overrides(args.dataset_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = args.output
    if output_dir is None:
        if config is not None:
            output_dir = config.output_dir
        else:
            output_dir = Path("runs/baseline")
    if args.resume_from is not None and args.output is None:
        resume_path = Path(args.resume_from)
        output_dir = resume_path.parent if resume_path.is_file() else resume_path
    run_baseline_training(
        args.dataset,
        output_dir,
        dataset_format=args.dataset_format,
        dataset_key_map=dataset_key_map or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
        test_split=args.test_split,
        test_dataset=args.test_dataset,
        solver=args.solver,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        best_checkpoint_name=args.best_checkpoint_name,
        last_checkpoint_name=args.last_checkpoint_name,
        mixed_precision=True if args.mixed_precision else None,
        autocast_dtype=args.precision_dtype,
        grad_scaler=grad_scaler,
        dtype=args.dtype,
        update_frequency=args.update_frequency,
        num_workers=args.num_workers,
        log_every_steps=args.log_every_steps,
        tensorboard=False if args.no_tensorboard else None,
        seed=args.seed,
        parameter_init=parameter_init,
        resume_from=args.resume_from,
        config=config,
    )


def _train_potential(args: argparse.Namespace) -> None:
    experiment = load_config(args.config, PotentialExperimentConfig)
    dataset_path = args.dataset or experiment.dataset.path
    if dataset_path is None:
        raise ValueError("A dataset path must be provided via the CLI or configuration file")
    dataset_path = Path(dataset_path)
    config_key_map = dict(experiment.dataset.key_map or {})
    try:
        cli_key_map = _parse_dataset_key_overrides(args.dataset_key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config_key_map.update(cli_key_map)
    dataset_format = (
        args.dataset_format
        if args.dataset_format not in {None}
        else experiment.dataset.format
    )
    dataset = load_dataset(
        dataset_path,
        format=dataset_format,
        key_map=config_key_map or None,
    )
    species = _resolve_species(dataset, experiment.dataset.species)
    graph_dataset = MolecularGraphDataset(
        dataset,
        species=species,
        cutoff=experiment.dataset.cutoff,
        dtype=experiment.dataset.dtype,
    )
    test_dataset_path = args.test_dataset or experiment.dataset.test_path
    test_graph_dataset: MolecularGraphDataset | None = None
    if test_dataset_path is not None:
        test_format = args.test_dataset_format or experiment.dataset.test_format or dataset_format
        test_raw = load_dataset(
            test_dataset_path,
            format=test_format,
            key_map=config_key_map or None,
        )
        unseen_species = {int(z) for atoms in test_raw.atoms for z in atoms} - set(species)
        if unseen_species:
            raise ValueError(
                f"Test dataset contains species {sorted(unseen_species)} not present in training data"
            )
        test_graph_dataset = MolecularGraphDataset(
            test_raw,
            species=species,
            cutoff=experiment.dataset.cutoff,
            dtype=experiment.dataset.dtype,
        )
    training_cfg = experiment.training
    model_overrides = {}
    overrides = {}
    if args.output is not None:
        overrides["output_dir"] = args.output
    elif args.resume_from is not None:
        resume_path = Path(args.resume_from)
        overrides["output_dir"] = resume_path.parent if resume_path.is_file() else resume_path
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.update_frequency is not None:
        overrides["update_frequency"] = args.update_frequency
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.validation_split is not None:
        overrides["validation_split"] = args.validation_split
    if args.test_split is not None:
        overrides["test_split"] = args.test_split
    if args.mixed_precision:
        overrides["mixed_precision"] = True
    if args.precision_dtype is not None:
        overrides["autocast_dtype"] = args.precision_dtype
    if args.no_grad_scaler:
        overrides["grad_scaler"] = False
    if args.log_every_steps is not None:
        overrides["log_every_steps"] = args.log_every_steps
    if args.no_tensorboard:
        overrides["tensorboard"] = False
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.parameter_init is not None:
        if args.parameter_init.lower() in {"default", "none"}:
            overrides["parameter_init"] = None
        else:
            overrides["parameter_init"] = args.parameter_init
    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
    if args.model_name is not None:
        name = args.model_name
        if name in {"hybrid-potential", "soap-transformer"}:
            name = "hybrid"
        model_overrides["name"] = name
    if args.hidden_dim is not None:
        model_overrides["hidden_dim"] = args.hidden_dim
    if args.transformer_layers is not None:
        model_overrides["transformer_layers"] = args.transformer_layers
    if args.gcn_layers is not None:
        model_overrides["gcn_layers"] = args.gcn_layers
    if args.se3_layers is not None:
        model_overrides["se3_layers"] = args.se3_layers
    if args.se3_distance_embedding is not None:
        model_overrides["se3_distance_embedding"] = args.se3_distance_embedding
    if args.num_heads is not None:
        model_overrides["num_heads"] = args.num_heads
    if args.ffn_dim is not None:
        model_overrides["ffn_dim"] = args.ffn_dim
    if args.dropout is not None:
        model_overrides["dropout"] = args.dropout
    if args.cutoff is not None:
        model_overrides["cutoff"] = args.cutoff
    if args.use_coordinate_features is not None:
        model_overrides["use_coordinate_features"] = args.use_coordinate_features
    if args.predict_forces is not None:
        model_overrides["predict_forces"] = args.predict_forces
    if args.residual_mode is not None:
        experiment = replace(experiment, model=replace(experiment.model, residual_mode=args.residual_mode))
    if model_overrides:
        experiment = replace(experiment, model=replace(experiment.model, **model_overrides))
    if overrides:
        if "output_dir" in overrides and not isinstance(overrides["output_dir"], Path):
            overrides["output_dir"] = Path(overrides["output_dir"])
        training_cfg = replace(training_cfg, **overrides)
    elif not isinstance(training_cfg.output_dir, Path):
        training_cfg = replace(training_cfg, output_dir=Path(training_cfg.output_dir))
    if experiment.model.predict_forces and not training_cfg.predict_forces_directly:
        training_cfg = replace(training_cfg, predict_forces_directly=True)
    if training_cfg.output_dir is None:
        raise ValueError("Potential training configuration must define an output directory")
    configure_logging(training_cfg.output_dir)
    if is_main_process():
        LOGGER.info("Training potential model using dataset at %s", dataset_path)
    model = _build_potential_model(experiment.model, species)
    baseline, baseline_trainable = _load_baseline(experiment.baseline, species)
    if (
        baseline is not None
        and experiment.baseline is not None
        and is_main_process()
    ):
        LOGGER.info("Loaded baseline checkpoint from %s", experiment.baseline.checkpoint)
        if not baseline_trainable:
            LOGGER.info("Baseline parameters will remain frozen during potential training")
    trainer = train_potential_model(
        graph_dataset,
        model,
        config=training_cfg,
        baseline=baseline,
        baseline_requires_grad=baseline_trainable,
        residual_mode=experiment.model.residual_mode,
    )
    if test_graph_dataset is not None:
        (
            test_metrics,
            predictions,
            targets,
            force_predictions,
            force_targets,
        ) = evaluate_potential_model(
            trainer.model,
            test_graph_dataset,
            baseline=baseline,
            residual_mode=experiment.model.residual_mode,
            batch_size=training_cfg.batch_size,
            num_workers=training_cfg.num_workers,
            device=trainer.device,
        )
        trainer.history.update({f"test/{name}": value for name, value in test_metrics.items()})
        results_path = training_cfg.output_dir / "potential_test_results.npz"
        try:
            save_predictions_and_targets(predictions, targets, results_path)
            if is_main_process():
                LOGGER.info("Saved potential test predictions to %s", results_path)
        except Exception as exc:  # pragma: no cover - best effort persistence
            if is_main_process():
                LOGGER.warning("Failed to save potential test predictions: %s", exc)
        if force_predictions is not None and force_targets is not None:
            force_results_path = training_cfg.output_dir / "potential_test_forces.npz"
            try:
                save_force_predictions_and_targets(
                    force_predictions, force_targets, force_results_path
                )
                if is_main_process():
                    LOGGER.info("Saved potential test force predictions to %s", force_results_path)
            except Exception as exc:  # pragma: no cover - best effort persistence
                if is_main_process():
                    LOGGER.warning("Failed to save potential test force predictions: %s", exc)
        plot_path = training_cfg.output_dir / "potential_test_predictions.png"
        try:
            plot_predictions_vs_targets(
                predictions,
                targets,
                plot_path,
                title="Potential predictions vs targets (test)",
            )
            if is_main_process():
                LOGGER.info("Saved potential test scatter plot to %s", plot_path)
        except Exception as exc:  # pragma: no cover - plotting best effort
            if is_main_process():
                LOGGER.warning("Failed to save potential test plot: %s", exc)
    if trainer.distributed.is_main_process():
        checkpoint_path = training_cfg.output_dir / "potential.pt"
        best_path = trainer.best_checkpoint_path
        last_path = trainer.last_checkpoint_path
        if best_path is not None:
            LOGGER.info("Best potential checkpoint saved to %s", best_path)
        if last_path is not None and last_path != best_path:
            LOGGER.info("Last potential checkpoint saved to %s", last_path)
        alias_source = best_path or last_path
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
        if dataset_format is not None:
            resolved_dataset_cfg = replace(resolved_dataset_cfg, format=dataset_format)
        if test_dataset_path is not None:
            resolved_dataset_cfg = replace(resolved_dataset_cfg, test_path=test_dataset_path)
        if args.test_dataset_format is not None:
            resolved_dataset_cfg = replace(
                resolved_dataset_cfg, test_format=args.test_dataset_format
            )
        if config_key_map:
            resolved_dataset_cfg = replace(resolved_dataset_cfg, key_map=dict(config_key_map))
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
    train_parser.add_argument("dataset", type=Path, help="Path to the dataset file")
    train_parser.add_argument(
        "--dataset-format",
        choices=_DATASET_FORMAT_CHOICES,
        default="auto",
        help="Format of the dataset file (default: infer from extension)",
    )
    train_parser.add_argument(
        "--dataset-key",
        action="append",
        default=None,
        metavar="CANON=FIELD",
        help="Map dataset fields to canonical keys (repeatable)",
    )
    train_parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype to load baseline training data with (default: float32)",
    )
    train_parser.add_argument("--output", type=Path, default=None)
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a checkpoint file or output directory to resume baseline training",
    )
    train_parser.add_argument("--config", type=Path, help="YAML file with training overrides")
    train_parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 200)")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 128)")
    train_parser.add_argument(
        "--update-frequency",
        type=int,
        default=None,
        help="Accumulate gradients over N batches before stepping the optimizer",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: 0)",
    )
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
        "--test-split",
        type=float,
        default=None,
        help="Test fraction to hold out from the training dataset (default: 0.0)",
    )
    train_parser.add_argument(
        "--test-dataset",
        type=Path,
        default=None,
        help="Optional dataset used only for testing after training",
    )
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    train_parser.add_argument(
        "--solver",
        choices=["optimizer", "least_squares"],
        default=None,
        help="Solver for the baseline weights (default: optimizer)",
    )
    train_parser.add_argument(
        "--log-every-steps",
        type=int,
        default=None,
        help="Print batch metrics every N steps (default: 100)",
    )
    train_parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging for baseline training",
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
        "--parameter-init",
        choices=[
            "default",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
            "zeros",
        ],
        default="default",
        help="Initialisation strategy for trainable parameters",
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

    baseline_predictor = subcommands.add_parser(
        "predict-baseline", help="Run inference and plotting with a trained baseline"
    )
    baseline_predictor.add_argument("dataset", type=Path, help="Dataset containing energies")
    baseline_predictor.add_argument("checkpoint", type=Path, help="Path to the baseline checkpoint")
    baseline_predictor.add_argument(
        "--dataset-format",
        choices=_DATASET_FORMAT_CHOICES,
        default="auto",
        help="Format of the dataset file (default: infer from extension)",
    )
    baseline_predictor.add_argument(
        "--dataset-key",
        action="append",
        default=None,
        metavar="CANON=FIELD",
        help="Map dataset fields to canonical keys (repeatable)",
    )
    baseline_predictor.add_argument(
        "--species",
        type=int,
        action="append",
        default=None,
        help="Explicit species ordering used during baseline training",
    )
    baseline_predictor.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Torch dtype to load baseline prediction data with (default: float32)",
    )
    baseline_predictor.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )
    baseline_predictor.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers to use (default: 0)",
    )
    baseline_predictor.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (default: model device)",
    )
    baseline_predictor.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store predictions, metrics, and plots (default: checkpoint directory)",
    )
    baseline_predictor.set_defaults(func=_predict_baseline)

    potential_parser = subcommands.add_parser(
        "train-potential", help="Train a neural potential with configurable settings"
    )
    potential_parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Path to the dataset file (overrides dataset.path in the config)",
    )
    potential_parser.add_argument(
        "--dataset-format",
        choices=_DATASET_FORMAT_CHOICES,
        default=None,
        help="Override the dataset format defined in the config",
    )
    potential_parser.add_argument(
        "--dataset-key",
        action="append",
        default=None,
        metavar="CANON=FIELD",
        help="Override dataset key mappings defined in the config",
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
    potential_parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a checkpoint file or output directory to resume potential training",
    )
    potential_parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    potential_parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    potential_parser.add_argument(
        "--update-frequency",
        type=int,
        default=None,
        help="Accumulate gradients for N batches before each optimizer step",
    )
    potential_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers to use (default: 0)",
    )
    potential_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    potential_parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Override validation split",
    )
    potential_parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Hold out a fraction of the dataset for testing",
    )
    potential_parser.add_argument(
        "--test-dataset",
        type=Path,
        default=None,
        help="Path to a dedicated test dataset",
    )
    potential_parser.add_argument(
        "--test-dataset-format",
        choices=_DATASET_FORMAT_CHOICES,
        default=None,
        help="Override the format used to read the test dataset",
    )
    potential_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible runs"
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
    potential_parser.add_argument(
        "--model",
        dest="model_name",
        choices=["transformer", "hybrid", "hybrid-potential", "soap-transformer", "se3", "gcn"],
        default=None,
        help="Override the architecture defined in the config (options include transformer, hybrid, and se3)",
    )
    potential_parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Hidden dimension for the potential backbone"
    )
    potential_parser.add_argument(
        "--transformer-layers",
        type=int,
        default=None,
        help="Number of transformer layers for hybrid or SE(3) architectures",
    )
    potential_parser.add_argument(
        "--gcn-layers", type=int, default=None, help="Number of GCN layers for the hybrid model"
    )
    potential_parser.add_argument(
        "--se3-layers",
        type=int,
        default=None,
        help="Number of SE(3) attention blocks (defaults to transformer_layers when omitted)",
    )
    potential_parser.add_argument(
        "--se3-distance-embedding",
        type=int,
        default=None,
        help="Distance embedding dimension for SE(3) attention biases",
    )
    potential_parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    potential_parser.add_argument(
        "--ffn-dim",
        type=int,
        default=None,
        help="Transformer feedforward hidden dimension for hybrid or SE(3) models",
    )
    potential_parser.add_argument(
        "--dropout", type=float, default=None, help="Dropout probability applied throughout the model"
    )
    potential_parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Neighbourhood cutoff (Å) for graph construction and distance embeddings",
    )
    coord_group = potential_parser.add_mutually_exclusive_group()
    coord_group.add_argument(
        "--use-coordinate-features",
        dest="use_coordinate_features",
        action="store_true",
        default=None,
        help="Include coordinate-derived features in the hybrid potential",
    )
    coord_group.add_argument(
        "--no-coordinate-features",
        dest="use_coordinate_features",
        action="store_false",
        help="Disable coordinate-derived features in the hybrid potential",
    )
    predict_group = potential_parser.add_mutually_exclusive_group()
    predict_group.add_argument(
        "--predict-forces",
        dest="predict_forces",
        action="store_true",
        default=None,
        help="Enable direct force prediction heads when supported by the architecture",
    )
    predict_group.add_argument(
        "--no-predict-forces",
        dest="predict_forces",
        action="store_false",
        help="Disable direct force prediction even if the config enables it",
    )
    residual_group = potential_parser.add_mutually_exclusive_group()
    residual_group.add_argument(
        "--residual-mode",
        dest="residual_mode",
        action="store_true",
        help="Train the potential on residual energies (target = E - baseline)",
    )
    residual_group.add_argument(
        "--absolute-mode",
        dest="residual_mode",
        action="store_false",
        help="Train the potential on absolute energies without subtracting the baseline",
    )
    residual_group.set_defaults(residual_mode=None)
    potential_parser.add_argument(
        "--log-every-steps",
        type=int,
        default=None,
        help="Print batch metrics every N steps (default: 100)",
    )
    potential_parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging for potential training",
    )
    potential_parser.add_argument(
        "--parameter-init",
        choices=[
            "default",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "orthogonal",
            "zeros",
        ],
        default="default",
        help="Initialisation strategy for the potential weights",
    )
    potential_parser.set_defaults(func=_train_potential)

    potential_predictor = subcommands.add_parser(
        "predict-potential", help="Run inference and plotting with a trained potential"
    )
    potential_predictor.add_argument("dataset", type=Path, help="Dataset containing energies")
    potential_predictor.add_argument("checkpoint", type=Path, help="Path to the potential checkpoint")
    potential_predictor.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Path to the experiment YAML (defaults to <checkpoint_dir>/experiment.yaml)",
    )
    potential_predictor.add_argument(
        "--dataset-format",
        choices=_DATASET_FORMAT_CHOICES,
        default=None,
        help="Override the dataset format defined in the experiment",
    )
    potential_predictor.add_argument(
        "--dataset-key",
        action="append",
        default=None,
        metavar="CANON=FIELD",
        help="Override dataset key mappings defined in the config",
    )
    potential_predictor.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: training batch size)",
    )
    potential_predictor.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers to use (default: training workers)",
    )
    potential_predictor.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (default: model device)",
    )
    potential_predictor.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store predictions, metrics, and plots (default: checkpoint directory)",
    )
    potential_predictor.set_defaults(func=_predict_potential)

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
