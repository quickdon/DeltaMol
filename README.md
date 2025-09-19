# DeltaMol

DeltaMol is a modular research toolkit for building delta-learning pipelines for
molecular energy and force models. The project combines a lightweight linear
atomic baseline with neural corrections, wraps several descriptor families
behind a unified API, and exposes command line tools that make it easy to stage
datasets, cache descriptors, train models, and evaluate results.

## Project goals

* Provide a **reproducible pipeline** for training molecular energy and force
  models on standard `.npz` archives.
* Support multiple descriptor families (ACSF, SOAP, SLATM, LMBTR, FCHL19) behind
  a common abstraction.
* Combine a linear atomic baseline with graph neural networks and transformer
  style message passing to model energy corrections.
* Offer ready-to-use tooling for dataset preparation, model training,
  checkpointing, and evaluation.

## Repository structure

```
deltamol/
├── cli/             # Command line entry points
├── config/          # Configuration serialization helpers
├── data/            # Dataset IO, caching and splitting utilities
├── descriptors/     # Descriptor abstraction layer and concrete wrappers
├── evaluation/      # Metrics and validation helpers
├── models/          # Baseline and graph neural network models
├── training/        # Trainers, pipelines and reusable routines
└── utils/           # Miscellaneous helpers
```

Additional top level directories include:

* `docs/` – human readable guides and architecture notes.
* `tests/` – unit tests covering key utilities.
* `requirements.txt` – base runtime dependencies for quick environment setup.

## Quickstart

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional descriptor backends (``dscribe`` and ``qmllib``) can be installed using
`pip install .[descriptors]`.

### Baseline training

DeltaMol exposes a CLI that orchestrates the baseline training loop:

```bash
python -m deltamol.main train-baseline path/to/dataset.npz \
    --output runs/baseline \
    --epochs 500 \
    --batch-size 256 \
    --lr 5e-3
```

The command will:

1. Load the dataset and build atomic formula vectors.
2. Train the linear baseline with mean squared error.
3. Save the trained model checkpoint and configuration into the output folder.

### Descriptor caching

Use the `cache-descriptors` subcommand to precompute descriptor matrices:

```bash
python -m deltamol.main cache-descriptors path/to/dataset.npz \
    --descriptor soap \
    --cache caches/soap.h5
```

This iterates over the dataset, generates the requested descriptor, and stores
it as individual datasets in an HDF5 cache.

## Documentation and tests

The `docs/` directory contains a growing set of guides on design decisions,
workflow automation, and future roadmap items. Unit tests live under `tests/`
and can be executed with `pytest` once the development extras are installed:

```bash
pip install .[dev]
pytest
```

Contributions and suggestions are welcome!
