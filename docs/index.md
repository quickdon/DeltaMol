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

1. Prepare a dataset as a NumPy ``.npz`` archive containing ``atoms``,
   ``coordinates`` and ``Etot``/``energies`` arrays.
2. Cache descriptors using ``python -m deltamol.main cache-descriptors``. The
   cache is stored as an HDF5 file.
3. Train the baseline using ``python -m deltamol.main train-baseline``.
4. Plug the cached descriptors into graph/transformer models (under active
   development).

## Reproducibility

The training utilities seed Python, NumPy, and Torch when a ``seed`` value is
provided via the CLI or configuration files. Data splits, dataloader shuffles,
and parameter initialisation are therefore deterministic across distributed
runs. Choose from standard initialisation schemes (Xavier, Kaiming, orthogonal,
zeros) with the ``parameter_init`` option when constructing either baseline or
potential models.

## Roadmap

* Formalize the energy correction graph network and expose a high level trainer.
* Add force supervision via analytic gradients and direct regression.
* Integrate experiment tracking and configuration management improvements.
* Provide example notebooks and benchmarking scripts.
