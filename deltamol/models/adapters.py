"""Adapters that normalise potential model inputs and outputs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import nn

from .potential import PotentialOutput


@dataclass
class AdapterInputs:
    """Container for prepared per-structure inputs for external models."""

    species: List[torch.Tensor]
    positions: List[torch.Tensor]
    neighbors: List[torch.Tensor]
    neighbor_distances: List[torch.Tensor]


def _standardise_energy_tensor(energy: torch.Tensor) -> torch.Tensor:
    """Ensure energies are shaped ``(batch,)`` and detached from padding."""

    if energy.ndim > 1:
        energy = energy.squeeze(-1)
    return energy


class PotentialModelAdapter(nn.Module):
    """Wrap arbitrary models so they emit :class:`PotentialOutput`."""

    def __init__(
        self,
        model: nn.Module,
        *,
        neighbor_strategy: str = "adjacency",
        neighbor_cutoff: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.neighbor_strategy = neighbor_strategy
        self.neighbor_cutoff = neighbor_cutoff

    def forward(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> PotentialOutput:
        prepared = self._prepare_inputs(node_indices, positions, adjacency, mask)
        raw_output = self._invoke_model(prepared)
        return self._to_potential_output(raw_output, mask)

    def _prepare_inputs(
        self,
        node_indices: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> AdapterInputs:
        species_list: List[torch.Tensor] = []
        position_list: List[torch.Tensor] = []
        neighbor_list: List[torch.Tensor] = []
        radii_list: List[torch.Tensor] = []
        batch_size = node_indices.size(0)
        for i in range(batch_size):
            valid = mask[i]
            species = node_indices[i, valid]
            coords = positions[i, valid]
            neighbors, distances = self._build_neighbors(coords, adjacency[i], valid)
            species_list.append(species)
            position_list.append(coords)
            neighbor_list.append(neighbors)
            radii_list.append(distances)
        return AdapterInputs(
            species=species_list,
            positions=position_list,
            neighbors=neighbor_list,
            neighbor_distances=radii_list,
        )

    def _build_neighbors(
        self,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.neighbor_strategy == "cutoff" and self.neighbor_cutoff is not None:
            dist_matrix = torch.cdist(positions, positions)
            dense_adj = (dist_matrix <= float(self.neighbor_cutoff)).to(positions.dtype)
        else:
            dense_adj = adjacency.to(positions.dtype)
        dense_adj = dense_adj * mask.unsqueeze(0) * mask.unsqueeze(1)
        edge_index = dense_adj.nonzero(as_tuple=False).t()
        distances = torch.tensor([], device=positions.device, dtype=positions.dtype)
        if edge_index.numel() > 0:
            src, dst = edge_index
            distances = torch.norm(positions[src] - positions[dst], dim=-1)
        return edge_index, distances

    def _invoke_model(self, prepared: AdapterInputs):
        try:
            return self.model(prepared)
        except TypeError:
            return self.model(
                species=prepared.species,
                positions=prepared.positions,
                neighbors=prepared.neighbors,
                neighbor_distances=prepared.neighbor_distances,
            )

    def _to_potential_output(
        self, raw_output, mask: torch.Tensor
    ) -> PotentialOutput:
        if isinstance(raw_output, PotentialOutput):
            return raw_output
        energy: Optional[torch.Tensor] = None
        forces: Optional[torch.Tensor] = None
        if isinstance(raw_output, dict):
            energy = raw_output.get("energy")
            if energy is None:
                energy = raw_output.get("energies")
            forces = raw_output.get("forces")
        elif isinstance(raw_output, (list, tuple)):
            if len(raw_output) > 0:
                energy = raw_output[0]
            if len(raw_output) > 1:
                forces = raw_output[1]
        elif torch.is_tensor(raw_output):
            energy = raw_output
        if energy is None:
            raise ValueError("External model did not return an energy prediction")
        energy = _standardise_energy_tensor(energy)
        if forces is not None and torch.is_tensor(forces):
            forces = forces * mask.unsqueeze(-1).to(forces.dtype)
        else:
            forces = None
        return PotentialOutput(energy=energy, forces=forces)


def load_external_model(factory_path: str) -> nn.Module:
    """Load a user-specified factory callable to build an external model."""

    module_path, _, attr = factory_path.partition(":")
    if not attr:
        module_path, _, attr = factory_path.rpartition(".")
    if not module_path or not attr:
        raise ValueError("External adapter path must look like 'module:factory'")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "Failed to import external model factory. Ensure optional dependencies are installed."
        ) from exc
    try:
        factory: Callable[..., nn.Module] = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_path}' does not export '{attr}'") from exc
    return factory()


__all__ = [
    "AdapterInputs",
    "PotentialModelAdapter",
    "load_external_model",
]

