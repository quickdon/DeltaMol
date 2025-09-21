"""Dataset helpers for potential energy training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.io import MolecularDataset
from ..models.baseline import build_formula_vector


@dataclass
class MolecularGraph:
    """Lightweight container describing a single molecular geometry."""

    node_indices: torch.Tensor
    positions: torch.Tensor
    adjacency: torch.Tensor
    energy: torch.Tensor
    formula_vector: torch.Tensor
    forces: Optional[torch.Tensor] = None


class MolecularGraphDataset(Dataset):
    """Construct dense graph representations from :class:`MolecularDataset`."""

    def __init__(
        self,
        dataset: MolecularDataset,
        *,
        species: Optional[Sequence[int]] = None,
        cutoff: float = 5.0,
        dtype: Union[torch.dtype, str] = torch.float32,
    ) -> None:
        self.cutoff = cutoff
        if species is None:
            unique_species = {int(z) for atoms in dataset.atoms for z in atoms}
            species = tuple(sorted(unique_species))
        self.species: Tuple[int, ...] = tuple(int(z) for z in species)
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(f"Unknown dtype string '{dtype}'") from exc
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Resolved dtype '{dtype}' is not a torch.dtype")
        self.dtype = dtype
        self.index_map: Dict[int, int] = {z: i + 1 for i, z in enumerate(self.species)}
        self.has_forces = dataset.forces is not None
        self.graphs: List[MolecularGraph] = []
        for i, atoms in enumerate(dataset.atoms):
            coordinates = dataset.coordinates[i]
            energy = dataset.energies[i] if dataset.energies is not None else 0.0
            graph = self._build_graph(atoms, coordinates, energy, dataset.forces[i] if self.has_forces else None)
            self.graphs.append(graph)

    def _atoms_to_indices(self, atoms: Iterable[int]) -> torch.Tensor:
        indices = [self.index_map[int(z)] for z in atoms]
        return torch.tensor(indices, dtype=torch.long)

    def _build_adjacency(self, positions: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(positions, positions)
        adjacency = (distances < self.cutoff).float()
        adjacency.fill_diagonal_(0.0)
        return adjacency

    def _build_graph(
        self,
        atoms: np.ndarray,
        coordinates: np.ndarray,
        energy: float,
        forces: Optional[np.ndarray],
    ) -> MolecularGraph:
        node_indices = self._atoms_to_indices(atoms)
        positions_array = np.asarray(coordinates, dtype=float)
        positions = torch.tensor(positions_array, dtype=self.dtype)
        adjacency = self._build_adjacency(positions)
        energy_tensor = torch.tensor(float(energy), dtype=self.dtype)
        formula_vector = build_formula_vector(atoms, species=self.species).to(self.dtype)
        forces_tensor = None
        if forces is not None:
            forces_tensor = torch.tensor(np.asarray(forces, dtype=float), dtype=self.dtype)
        return MolecularGraph(
            node_indices=node_indices,
            positions=positions,
            adjacency=adjacency,
            energy=energy_tensor,
            formula_vector=formula_vector,
            forces=forces_tensor,
        )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.graphs)

    def __getitem__(self, index: int) -> MolecularGraph:
        return self.graphs[index]


def collate_graphs(batch: Sequence[MolecularGraph]) -> Dict[str, torch.Tensor]:
    """Pad a batch of :class:`MolecularGraph` objects into dense tensors."""

    batch_size = len(batch)
    max_nodes = max(graph.node_indices.numel() for graph in batch)
    feature_dim = batch[0].formula_vector.numel()
    node_indices = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    positions = torch.zeros(batch_size, max_nodes, 3, dtype=batch[0].positions.dtype)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, dtype=batch[0].adjacency.dtype)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    formula_vectors = torch.zeros(batch_size, feature_dim, dtype=batch[0].formula_vector.dtype)
    energies = torch.zeros(batch_size, dtype=batch[0].energy.dtype)
    has_forces = batch[0].forces is not None
    forces = torch.zeros(batch_size, max_nodes, 3, dtype=batch[0].positions.dtype) if has_forces else None
    for i, graph in enumerate(batch):
        n = graph.node_indices.numel()
        node_indices[i, :n] = graph.node_indices
        positions[i, :n] = graph.positions
        adjacency[i, :n, :n] = graph.adjacency
        mask[i, :n] = True
        formula_vectors[i] = graph.formula_vector
        energies[i] = graph.energy
        if has_forces and graph.forces is not None:
            forces[i, :n] = graph.forces
    batch_dict = {
        "node_indices": node_indices,
        "positions": positions,
        "adjacency": adjacency,
        "mask": mask,
        "energies": energies,
        "formula_vectors": formula_vectors,
    }
    if has_forces:
        batch_dict["forces"] = forces
    return batch_dict


__all__ = ["MolecularGraph", "MolecularGraphDataset", "collate_graphs"]

