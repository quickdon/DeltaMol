"""Graph neural network models used to learn energy/force corrections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import TransformerConv
except ImportError as exc:  # pragma: no cover - optional dependency
    TransformerConv = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class GraphModelConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.0


class EnergyCorrectionNetwork(nn.Module):
    """Stack of TransformerConv layers that predicts an energy correction."""

    def __init__(self, config: GraphModelConfig):
        super().__init__()
        if TransformerConv is None:  # pragma: no cover - optional dependency
            raise ImportError("torch-geometric is required for the graph model") from _IMPORT_ERROR

        layers = []
        in_dim = config.node_dim
        for _ in range(config.num_layers):
            conv = TransformerConv(in_channels=in_dim, out_channels=config.hidden_dim, edge_dim=config.edge_dim)
            layers.append(conv)
            in_dim = config.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr):  # pragma: no cover - depends on torch-geometric
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr).relu()
        pooled = x.mean(dim=0, keepdim=True)
        return self.readout(pooled).squeeze(-1)


__all__ = ["GraphModelConfig", "EnergyCorrectionNetwork"]
