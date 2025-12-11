# DeltaMol documentation

DeltaMol aims to streamline the construction of machine learned potential energy
surfaces by embracing three core ideas:

1. **Delta-learning** – approximate the total molecular energy as a sum of
   atomic contributions and learn the residual using neural networks.
2. **Descriptor modularity** – expose a common wrapper around descriptor
   families so that datasets can be cached once and reused across experiments.
3. **Automated workflows** – ship command line tools that chain dataset loading,
   descriptor generation, model training, and evaluation.

## Workflow overview

1. Prepare a dataset that exposes the canonical ``atoms``, ``coordinates``, and
   ``energies`` fields (``forces`` is optional). Files stored as NumPy ``.npz``
   or ``.npy`` archives, JSON/YAML blobs, or Torch checkpoints are all
   supported. MD-style ``.npz`` trajectories with a single atomic-number array
   or a shared per-atom force frame are broadcast across all coordinate frames
   during loading, so you can point the loader at raw MD outputs without
   manually reshaping them. Pass either a single file, a directory, or multiple
   paths to load and concatenate several molecules at once.
2. Cache descriptors using ``python -m deltamol.main cache-descriptors``. The
   cache is stored as an HDF5 file.
3. Train the baseline using ``python -m deltamol.main train-baseline``.
4. Train the hybrid SOAP-guided potential or the SE(3)-Transformer variant
   using the cached descriptors and the `train-potential` command. The
   architecture can be selected in the experiment YAML via `model.name` or
   overridden on the CLI with `--model se3` or `--model hybrid` while keeping
   the same residual learning flow as the baseline.

## Evaluation and testing

DeltaMol ships reusable helpers for evaluating both the linear baseline and the
hybrid potential. Regression metrics (MSE, RMSE, MAE) are computed via
``deltamol.evaluation.metrics.compute_regression_metrics`` and convenience
wrappers in ``deltamol.evaluation.testing`` load trained models, run inference
on a held-out dataset, and (when requested) generate a scatter plot comparing
predictions against ground truth values. The plot is saved alongside the model
outputs to make quick sanity checks easy.

Training commands now support both fractional test splits and externally
supplied test datasets. Use ``--test-split`` to carve out a portion of the
training data for post-training evaluation or ``--test-dataset`` /
``--test-dataset-format`` to point at a dedicated test file. After fitting, the
trainers automatically evaluate the best checkpoint on the test data, log the
metrics, and save a ``*_test_predictions.png`` scatter plot in the run
directory.

### SchNet potential architecture

DeltaMol includes a SchNet-inspired continuous-filter convolutional potential
for energy and (optionally) force prediction, following Schütt et al. (2018)
and the reference implementation at
`atomistic-machine-learning/SchNet <https://github.com/atomistic-machine-learning/SchNet>`_.
Select the architecture with ``model.name: schnet`` in an experiment YAML or
``--model schnet`` on the CLI. Key hyperparameters:

* ``hidden_dim`` – shared embedding dimension for atomic features (default 128).
* ``schnet_num_filters`` – number of continuous filters inside interaction blocks
  (defaults to ``hidden_dim`` when omitted).
* ``schnet_num_interactions`` – interaction layers stacked sequentially to
  refine atomic features (default 3).
* ``schnet_num_gaussians`` – radial basis size for Gaussian distance expansion
  between neighbours (default 50) with edges truncated at ``model.cutoff``.
* ``predict_forces`` – when ``true``, enables analytic force prediction via
  gradients of the summed energy w.r.t. coordinates.

Neighbour graphs respect the provided adjacency matrix; combine ``model.cutoff``
with dataset-level cutoffs to prune long-range edges when building training
graphs.

### DimeNet potential architecture

DeltaMol also includes a DimeNet-inspired directional message passing model
following Klicpera, Groß, and Günnemann (ICLR 2020) and the reference
implementation at `gasteigerjo/dimenet <https://github.com/gasteigerjo/dimenet>`_.
Activate it with ``model.name: dimenet`` in the experiment YAML or ``--model
dimenet`` on the CLI. Key hyperparameters:

* ``dimenet_num_blocks`` – number of directional message passing blocks stacked
  sequentially (default 3).
* ``dimenet_num_radial`` – radial Bessel basis size used to expand pairwise
  distances up to ``model.cutoff`` (default 6).
* ``dimenet_num_spherical`` – number of Fourier components used to embed triplet
  angles (default 7).

Like SchNet, DimeNet supports optional analytic force prediction via gradients of
the summed energy with respect to atomic coordinates; enable it with
``predict_forces: true`` when fitting energies and forces jointly.

The CLI also exposes dedicated prediction commands for evaluating existing
checkpoints. ``deltamol predict-baseline <dataset> <checkpoint>`` loads a linear
baseline, computes regression metrics on the provided dataset, saves serialized
predictions, and writes a ``baseline_predictions_<dataset>.png`` scatter plot
plus a JSON metrics file. ``deltamol predict-potential`` mirrors this flow for
neural potentials, rebuilding the architecture from the saved
``experiment.yaml`` (or a path supplied via ``--experiment``) before running
inference, plotting, and saving metrics to the chosen output directory.

Both trainers support resuming interrupted runs. Pass ``--resume-from`` on the
CLI or set ``training.resume_from`` inside an experiment YAML file; either form
accepts an output directory containing the latest checkpoint or an explicit
checkpoint path such as ``potential_last.pt``.

## Loss configuration

Potential training exposes two optional normalisation schemes to balance the
relative scale of energy and force supervision:

* ``training.energy_per_atom_loss`` – when ``true``, the trainer divides both
  predicted and target energies by each molecule's atom count before computing
  the MSE. This normalises the extensive energy to an intensive per-atom value,
  preventing large systems from overwhelming the loss when forces are also
  supervised.
* ``training.relative_force_loss`` – when ``true``, the trainer computes force
  errors relative to the target magnitude: ``(pred − target) / (|target| +
  relative_force_epsilon)``. Set ``training.relative_force_epsilon`` to a small
  positive constant (default ``1e-3``) to avoid divide-by-zero. This mirrors the
  DeepMD relative force strategy, reducing the dominance of outlier atoms with
  very large forces and keeping small-force regions well represented.

Example experiment YAML fragment:

```yaml
training:
  energy_per_atom_loss: true
  relative_force_loss: true
  relative_force_epsilon: 1.0e-3
  energy_weight: 1.0
  force_weight: 1.0
```

When datasets use different field names, specify a mapping between the
canonical keys (``atoms``, ``coordinates``, ``energies``, ``forces``) and the raw
columns either on the CLI or inside the experiment YAML. Additional arrays are
preserved as metadata and remain accessible during downstream processing.

## Reproducibility

The training utilities seed Python, NumPy, and Torch when a ``seed`` value is
provided via the CLI or configuration files. Data splits, dataloader shuffles,
and parameter initialisation are therefore deterministic across distributed
runs. Choose from standard initialisation schemes (Xavier, Kaiming, orthogonal,
zeros) with the ``parameter_init`` option when constructing either baseline or
potential models.

## Distributed training options

Distributed settings are configured via the ``training.distributed`` block in
experiment YAML files. Setting ``enabled: true`` now works out-of-the-box on a
single node without pre-populating rendezvous environment variables – sensible
defaults for ``MASTER_ADDR`` and ``MASTER_PORT`` are injected automatically.
When targeting multiple GPUs, either specify explicit device IDs via
``devices`` (e.g. ``devices: [0, 2, 3]`` or ``devices: "0,2,3"``) or request
``world_size: all`` to match the number of visible CUDA devices. The resolved
device order is broadcast to ``local_rank`` so each process selects the
intended GPU even when using a non-contiguous list.

## Roadmap

* Extend the hybrid SOAP-guided potential with additional pooling and
  regularisation experiments.
* Benchmark the SE(3)-Transformer architecture for equivariant energy and force
  modelling.
* Add force supervision via analytic gradients and direct regression.
* Integrate experiment tracking and configuration management improvements.
* Provide example notebooks and benchmarking scripts.
