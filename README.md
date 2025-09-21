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

When the baseline behaves like a simple linear regression, you can bypass
iterative optimisation and recover the atomic energy coefficients with a single
least-squares solve:

```bash
python -m deltamol.main train-baseline data.npz --solver least_squares
```

The same `solver` option can be stored inside a YAML configuration alongside
other training overrides.

Advanced options such as the optimizer family, scheduler, or device can be
specified via a YAML file and passed to the command with `--config`. Any
arguments provided on the CLI continue to override the values defined in the
configuration file, so the most common tweaks remain one flag away:

```yaml
# baseline.yaml
output_dir: runs/baseline-adamw
optimizer: adamw
weight_decay: 0.01
scheduler: cosine
warmup_steps: 500
min_lr_ratio: 0.2
device: cuda
```

```bash
python -m deltamol.main train-baseline data.npz --config baseline.yaml --epochs 300
```

### Potential training with configuration files

The `train-potential` subcommand consumes a structured YAML file that describes
dataset preprocessing, model architecture, and training parameters. Only the
most important options such as the dataset path and output directory need to be
specified on the CLI; all other values live alongside the experiment
definition.

```yaml
# configs/potential.yaml
dataset:
  path: datasets/DFT_uniques.npz
  cutoff: 6.0
  dtype: float32
model:
  name: transformer
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  predict_forces: true
training:
  output_dir: runs/potential-transformer
  epochs: 80
  batch_size: 16
  learning_rate: 5.0e-4
  optimizer: adamw
  scheduler: cosine
  warmup_steps: 1000
  force_weight: 0.5
baseline:
  checkpoint: runs/baseline/baseline.pt
```

Launch the run with:

```bash
python -m deltamol.main train-potential --config configs/potential.yaml
```

The trainer will configure logging, build the requested model, run the
experiment, and persist an `experiment.yaml` file in the output directory that
captures the resolved dataset, model, and training parameters.

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

## License

This project is licensed under the terms of the MIT license. See LICENSE for additional details.
