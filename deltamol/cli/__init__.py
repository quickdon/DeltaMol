"""Command line interface for DeltaMol workflows."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

from ..data.io import cache_descriptor_matrix, load_npz_dataset
from ..descriptors.acsf import build_acsf_descriptor
from ..descriptors.fchl19 import build_fchl19_descriptor
from ..descriptors.lmbtr import build_lmbtr_descriptor
from ..descriptors.slatm import build_slatm_descriptor
from ..descriptors.soap import build_soap_descriptor
from ..main import run_baseline_training

_DESCRIPTOR_BUILDERS: Dict[str, Callable] = {
    "acsf": build_acsf_descriptor,
    "soap": build_soap_descriptor,
    "slatm": build_slatm_descriptor,
    "lmbtr": build_lmbtr_descriptor,
    "fchl19": build_fchl19_descriptor,
}


def _train_baseline(args: argparse.Namespace) -> None:
    run_baseline_training(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_split=args.validation_split,
    )


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
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--lr", type=float, default=1e-2)
    train_parser.add_argument("--validation-split", type=float, default=0.1)
    train_parser.set_defaults(func=_train_baseline)

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
