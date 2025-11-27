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
4. Train the hybrid SOAP-guided potential using the cached descriptors and the
   `train-potential` command.

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

Both trainers support resuming interrupted runs. Pass ``--resume-from`` on the
CLI or set ``training.resume_from`` inside an experiment YAML file; either form
accepts an output directory containing the latest checkpoint or an explicit
checkpoint path such as ``potential_last.pt``.

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

## Roadmap

* Extend the hybrid SOAP-guided potential with additional pooling and
  regularisation experiments.
* Add force supervision via analytic gradients and direct regression.
* Integrate experiment tracking and configuration management improvements.
* Provide example notebooks and benchmarking scripts.
