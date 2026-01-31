"""OpenMM interface helpers for DeltaMol potentials."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class OpenMMTypeMap:
    """Map OpenMM atom type indices (1-based) to atomic numbers."""

    atomic_numbers: Sequence[int]

    def resolve(self, types: Iterable[int]) -> list[int]:
        mapping = list(self.atomic_numbers)
        return [mapping[int(t) - 1] for t in types]


class DeltaMolTorchModule(torch.nn.Module):
    """Torch module that wraps a DeltaMol potential for OpenMM-Torch."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        type_map: OpenMMTypeMap,
        cutoff: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model = model
        self.cutoff = float(cutoff)
        self.dtype = dtype
        self.register_buffer(
            "type_map",
            torch.tensor(type_map.atomic_numbers, dtype=torch.long),
        )

    def forward(self, positions: torch.Tensor, atom_types: torch.Tensor):
        node_indices = self.type_map[atom_types.long() - 1]
        positions = positions.to(dtype=self.dtype)
        distances = torch.cdist(positions, positions)
        adjacency = (distances < self.cutoff).to(positions.dtype)
        adjacency.fill_diagonal_(0.0)
        mask = torch.ones(node_indices.shape, dtype=torch.bool, device=positions.device)
        output = self.model(
            node_indices.unsqueeze(0),
            positions.unsqueeze(0),
            adjacency.unsqueeze(0),
            mask.unsqueeze(0),
        )
        energy = output.energy.squeeze(0)
        forces = output.forces
        if forces is None:
            raise RuntimeError(
                "OpenMM TorchForce export requires a model that returns forces. "
                "Enable predict_forces=True during training."
            )
        return energy, forces.squeeze(0)


def export_torchscript_module(
    module: DeltaMolTorchModule,
    *,
    example_positions: torch.Tensor,
    example_types: torch.Tensor,
) -> torch.jit.ScriptModule:
    """Trace a DeltaMolTorchModule for use with OpenMM-Torch."""

    return torch.jit.trace(module, (example_positions, example_types))


def create_openmm_torch_force(script_module: torch.jit.ScriptModule):
    """Create an OpenMM-Torch Force from a traced module."""

    try:
        from openmm_torch import TorchForce
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("openmm-torch is required to build a TorchForce") from exc
    return TorchForce(script_module)


__all__ = [
    "DeltaMolTorchModule",
    "OpenMMTypeMap",
    "create_openmm_torch_force",
    "export_torchscript_module",
]
