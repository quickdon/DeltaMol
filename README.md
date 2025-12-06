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
* Combine a linear atomic baseline with a hybrid SOAP-guided graph/transformer
  potential to model energy corrections, and optionally test an
  SE(3)-Transformer architecture for equivariant attention over molecular
  geometries.
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
├── models/          # Baseline and hybrid potential models
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

If you prefer Conda, create an environment with:

```bash
conda create -n deltamol python=3.10
conda activate deltamol
pip install -r requirements.txt
```

The toolkit also ships standard packaging metadata so it can be installed as a
library from a source checkout:

```bash
pip install .
# or, for development
pip install -e .[dev]
```

Optional descriptor backends (``dscribe`` and ``qmllib``) can be installed using
`pip install .[descriptors]`.

### Baseline training

The recommended way to fit the baseline is to skip iterative optimisation and
recover the atomic energy coefficients with a single least-squares solve:

```bash
python -m deltamol.main train-baseline data.npz --solver least_squares
```

DeltaMol also exposes a CLI that orchestrates the baseline training loop with
an optimizer:

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
4. Persist both `baseline_best.pt` (best validation loss) and `baseline_last.pt`
   (latest weights) so you can transfer the best model into potential training
   while keeping a checkpoint that is ready to resume optimisation.

Resume an interrupted run by pointing `--resume-from` to either the previous
output directory or a specific checkpoint file:

```bash
python -m deltamol.main train-baseline data.npz \
    --output runs/baseline \
    --resume-from runs/baseline
```

When an output directory is provided the trainer will auto-discover
`baseline_last.pt` and continue training from the saved epoch, restoring the
optimizer, scheduler, and (when enabled) gradient scaler states.

DeltaMol standardises raw datasets by looking for the canonical fields
`atoms`, `coordinates`, `energies`, and (optionally) `forces`. Any additional
information remains available via the dataset metadata and the forces field can
be omitted when a source dataset only provides energies. Files stored as NPZ,
NPY, JSON, YAML, or Torch checkpoints are recognised automatically based on the
extension, and the CLI exposes `--dataset-format` to override auto-detection
when necessary. The loader understands MD-style `.npz` archives where a single
atomic-number array or per-atom force frame is shared across a trajectory; the
array is broadcast across all coordinate frames so you can load MD trajectories
without reshaping them. Dataset paths can point to either a single file, a
directory of supported files, or a space-separated list of paths; all matching
datasets are loaded and concatenated to simplify training across multiple
molecules at once. If a dataset uses different field names you can map them to
the canonical keys directly on the command line:

```bash
python -m deltamol.main train-baseline data.json \
    --dataset-format json \
    --dataset-key atoms=Z \
    --dataset-key coordinates=xyz \
    --dataset-key energies=E_total \
    --dataset-key forces=gradients
```

The same mapping controls propagate to potential training, so once a dataset is
standardised you can reuse the configuration across models.

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

Add `--early-stopping-patience` and `--early-stopping-min-delta` to stop the
run once the validation loss plateaus, and customise the checkpoint filenames
with `--best-checkpoint-name` and `--last-checkpoint-name` if the defaults do
not suit your workflow. The CLI mirrors these values inside the saved
`config.yaml` so downstream tasks can discover which files contain the best and
latest weights.

When GPU memory is at a premium you can add `--mixed-precision` to enable
automatic mixed precision. CUDA devices default to float16 autocasting while
CPUs fall back to bfloat16; override the behaviour with `--precision-dtype` and
use `--no-grad-scaler` to disable gradient scaling for edge cases that require
manual control.

Large batch experiments can also rely on gradient accumulation via
`--update-frequency`. Gradients are accumulated for the requested number of
mini-batches before a single optimiser update is applied, effectively running
with a global batch size of ``batch_size × world_size × update_frequency`` when
distributed training is enabled. Combine the flag with `--num-workers` to spin
up multi-process data loading on both CPU and GPU hosts.

Reproducible runs can set `--seed` to initialise Python, NumPy, and Torch RNGs
on every rank. The trainers also expose deterministic parameter initialisation
via `--parameter-init`, supporting schemes such as `xavier_uniform`,
`xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `orthogonal`, and `zeros`
in addition to the default PyTorch initialisers.

Use `--log-every-steps` to control how often batch-level metrics are printed to
the terminal (the default is every 100 steps). Every epoch summary reports its
elapsed time and both the training and validation losses. Runs also emit
TensorBoard event files under ``<output>/tensorboard`` capturing the train/val
loss curves, learning-rate schedule, and other tracked scalars; pass
`--no-tensorboard` if you prefer to skip event generation.

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
  format: npz
  cutoff: 6.0
  dtype: float32
  key_map:
    energies: Etot
model:
  name: hybrid
  hidden_dim: 256
  gcn_layers: 3
  transformer_layers: 4
  num_heads: 8
  ffn_dim: 512
  dropout: 0.1
  predict_forces: true
  cutoff: 6.0
  soap_num_radial: 8
  soap_max_angular: 4
  soap_cutoff: 6.0
  soap_gaussian_width: 0.5
  residual_mode: true
training:
  output_dir: runs/potential-hybrid
  epochs: 80
  batch_size: 16
  learning_rate: 5.0e-4
  optimizer: adamw
  scheduler: cosine
  warmup_steps: 1000
  force_weight: 0.5
  mixed_precision: true
  autocast_dtype: float16
baseline:
  checkpoint: runs/baseline/baseline.pt
  requires_grad: false
```

Launch the run with:

```bash
python -m deltamol.main train-potential --config configs/potential.yaml
```

The trainer will configure logging, build the requested model, run the
experiment, and persist an `experiment.yaml` file in the output directory that
captures the resolved dataset, model, and training parameters. In addition to
the copied `potential.pt` convenience file, the run directory will contain
`potential_best.pt` and `potential_last.pt` so you can evaluate the best model
and resume from the final optimiser state with minimal effort. Resume runs by
adding `--resume-from` on the CLI or setting `training.resume_from` in the
YAML; both forms accept either a full checkpoint path or an output directory
that contains `potential_last.pt`. Mixed precision
can be toggled directly in the YAML file via the `mixed_precision`,
`autocast_dtype`, and `grad_scaler` fields or overridden on the CLI with
`--mixed-precision`, `--precision-dtype`, and `--no-grad-scaler`. When a
baseline is present, `residual_mode` controls whether training targets subtract
the baseline prediction (the default) or optimise absolute energies instead;
override the YAML with `--residual-mode` or `--absolute-mode` on the command
line.

Architectures can be swapped without editing the YAML by passing
`--model transformer`, `--model hybrid`, or `--model se3` on the CLI; the flag
respects the same residual training flow used by the baseline so SE(3)
Transformer runs participate in delta-learning alongside the hybrid potential.
Additional overrides for depth (`--transformer-layers`, `--gcn-layers`,
`--se3-layers`), width (`--hidden-dim`, `--ffn-dim`, `--num-heads`), and SE(3)
distance embeddings (`--se3-distance-embedding`) keep experiments flexible
without needing to duplicate configuration files. When `predict_forces` is
enabled in either the YAML or via `--predict-forces`, the trainer automatically
switches to direct force supervision rather than deriving gradients from the
predicted energies.

Dataset sections inside the experiment configuration accept the same `format`
and `key_map` fields exposed on the CLI. Each `key_map` entry follows the
`canonical: source` syntax and determines where to find the atomic numbers,
coordinates, energies, or forces within the raw file. Because these mappings are
stored alongside the rest of the experiment description, you can reuse the same
configuration when swapping between datasets that share the same layout.

The potential workflow mirrors the logging controls offered by the baseline
trainer. Adjust batch-level verbosity with `--log-every-steps` and disable
TensorBoard event generation via `--no-tensorboard` when desired. Each run
writes train/validation energy and force loss curves, along with the learning
rate and epoch durations, to ``<output>/tensorboard`` for convenient monitoring
in TensorBoard.

Potential experiments honour the same `seed` and `parameter_init` options so
multi-rank training stays reproducible and model initialisation is fully
controlled from the configuration file or CLI overrides.

### Evaluation and testing

Both trainers support automated testing on held-out data. Add `--test-split` to
reserve a fraction of the training dataset for final evaluation, or provide a
dedicated file via `--test-dataset` (optionally overriding auto-detection with
`--test-dataset-format`). After training, the best baseline or potential
checkpoint is evaluated on the test data, logging MSE, RMSE, and MAE values,
writing the full prediction/target arrays to `baseline_test_results.npz` or
`potential_test_results.npz`, and saving a `baseline_test_predictions.png` or
`potential_test_predictions.png` scatter plot that compares predictions against
the true energies. When forces are evaluated, the test report additionally logs
force metrics (component-wise and averaged per atom) and writes the full force
prediction/target arrays to `potential_test_forces.npz`.

Trained models can also be reused for stand-alone evaluation or application.
Use `deltamol predict-baseline DATASET CHECKPOINT` to load a baseline
checkpoint, compute metrics on the provided dataset, and write predictions plus
`baseline_predictions_<dataset>.png` and `baseline_metrics_<dataset>.json`
artifacts alongside the checkpoint (or in a custom `--output` directory). For
potential models, `deltamol predict-potential DATASET CHECKPOINT --experiment
experiment.yaml` rebuilds the architecture from the saved experiment
description, evaluates the checkpoint on the supplied dataset, and saves an
equivalent set of metrics, serialized predictions, and scatter plots for rapid
model assessment.

Example invocations:

```bash
# Evaluate a trained baseline on a fresh dataset
deltamol predict-baseline data/aspirin_test.npz runs/baseline/baseline_best.pt \
  --output runs/baseline/application

# Apply a potential checkpoint using the stored experiment description
deltamol predict-potential data/aspirin_test.npz runs/potential/potential_best.pt \
  --experiment runs/potential/experiment.yaml \
  --output runs/potential/application
```

Distributed training is fully supported: launch the CLI with `torchrun` to
perform single-node multi-GPU optimisation, or combine `--nproc_per_node` with
`--nnodes` to scale across machines. The trainer automatically detects the
world size, keeps logging confined to the designated main process, and averages
metrics across ranks. Advanced settings such as backend selection, explicit
`WORLD_SIZE`, or a fixed main-process rank can be expressed inside the training
configuration via the nested `distributed` block. Rendezvous defaults are set
automatically (`MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29500`) when you do not
provide an `init_method`, so single-node jobs no longer need extra environment
variables. GPU selection now accepts either an explicit list or an `all`
shorthand to match the number of visible CUDA devices:

```yaml
training:
  output_dir: runs/potential-hybrid
  batch_size: 16
  update_frequency: 4
  num_workers: 8
  distributed:
    enabled: true
    backend: nccl
    main_process: 0
    # Use all detected GPUs on the host; alternatively, list explicit device IDs
    world_size: all
    devices: [0, 2, 3]
```

With this configuration each optimiser step sees the equivalent of
``16 × 4 × 4 = 256`` samples on a four-GPU node. The same `update_frequency`
and `num_workers` controls are available on the baseline CLI, and the trainer
automatically freezes logging and checkpoint writes on non-main ranks.

#### Torchrun quick start

A minimal two-GPU launch looks like:

```bash
torchrun --nproc_per_node=2 -m deltamol.main train-potential --config configs/potential-ddp.yaml
```

When `torchrun` spawns multiple processes it sets `OMP_NUM_THREADS=1` for each
rank to avoid oversubscribing CPU cores with OpenMP threads. Set this
environment variable explicitly if you want heavier CPU-side work (for example,
larger dataloader transforms or CPU-bound descriptor steps) to use more
threads:

```bash
OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 -m deltamol.main train-potential \
    --config configs/potential-ddp.yaml
```

Override `--config` as needed for other experiments, and append additional
`torchrun` arguments such as `--nnodes` when scaling across hosts.

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
