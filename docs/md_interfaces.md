# MD engine interfaces (LAMMPS / OpenMM)

This guide shows how to plug a DeltaMol-trained potential into mainstream MD engines.
The goal is to reuse the `train-potential` outputs directly in LAMMPS or OpenMM.

## Prerequisites

* A trained potential run directory containing:
  * `experiment.yaml` – experiment configuration saved by DeltaMol
  * `potential.pt` – trained potential weights
* Runtime dependencies (PyTorch). Install the Python bindings for LAMMPS/OpenMM
  as needed (see the sections below).

## 1. Common loading workflow

DeltaMol exposes `load_trained_potential` to rebuild a model from the saved
experiment config and checkpoint. Use `PotentialCalculator` to convert atomic
numbers and coordinates into model inputs.

```python
from deltamol.interfaces import load_trained_potential, PotentialCalculator

bundle = load_trained_potential("runs/potential/experiment.yaml", device="cpu")
calculator = PotentialCalculator(
    bundle.model,
    species=bundle.experiment.dataset.species,
    cutoff=bundle.experiment.dataset.cutoff,
)
```

`PotentialCalculator` builds the neighbor adjacency from the training cutoff and
returns both energies and forces.

## 2. LAMMPS interface

LAMMPS offers `pair_style python`, which calls into Python to evaluate custom
potentials. The adapter below provides the callback LAMMPS expects:

```python
from deltamol.interfaces import LammpsPairPotential, LammpsTypeMap

# LAMMPS atom types (1-based) -> atomic numbers, e.g. 1=H, 2=C, 3=O
lammps_type_map = LammpsTypeMap([1, 6, 8])

pair = LammpsPairPotential(
    calculator,
    type_map=lammps_type_map,
)
```

In the LAMMPS input script:

```lammps
pair_style python 1
pair_coeff * * deltamol_lammps_pair.py pair
```

`deltamol_lammps_pair.py` should expose a `pair` object (the
`LammpsPairPotential` instance). LAMMPS will call `compute` each MD step to
populate forces and return the total energy.

## 3. OpenMM interface

OpenMM typically uses the `openmm-torch` plugin to run PyTorch models. DeltaMol
ships `DeltaMolTorchModule` to build a TorchScript module for that plugin:

```python
import torch
from deltamol.interfaces import (
    DeltaMolTorchModule,
    OpenMMTypeMap,
    export_torchscript_module,
    create_openmm_torch_force,
)

openmm_type_map = OpenMMTypeMap([1, 6, 8])
module = DeltaMolTorchModule(
    bundle.model,
    type_map=openmm_type_map,
    cutoff=bundle.experiment.dataset.cutoff,
)

# Provide an example structure for TorchScript tracing
example_positions = torch.zeros((5, 3))
example_types = torch.tensor([1, 1, 2, 2, 3])
script_module = export_torchscript_module(
    module,
    example_positions=example_positions,
    example_types=example_types,
)

force = create_openmm_torch_force(script_module)
```

Attach `force` to the OpenMM `System`. Note that OpenMM's TorchForce expects
models to return forces directly, so train with `predict_forces: true`.

## 4. Tips and caveats

* **Type mapping**: The atom type ordering used by LAMMPS/OpenMM must match the
  element ordering used during training.
* **Cutoff consistency**: The interface builds neighbor graphs using the same
  cutoff as training to avoid energy/force drift.
* **Export stability**: For OpenMM, keep the model weights and
  `predict_forces` setting fixed before exporting TorchScript.

For advanced usage (neighbor list optimization, GPU batching, multi-structure
inference), extend the utilities in `deltamol.interfaces`.
