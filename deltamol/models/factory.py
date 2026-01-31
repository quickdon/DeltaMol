"""Factory helpers for building potential models."""
from __future__ import annotations

from typing import Sequence

import torch

from .adapters import PotentialModelAdapter, load_external_model
from .dimenet import DimeNetConfig, DimeNetPotential
from .equiformer_v2 import EquiformerV2Config, EquiformerV2Potential
from .gemnet import GemNetConfig, GemNetPotential
from .hybrid import HybridPotential, HybridPotentialConfig
from .leftnet import LeftNetConfig, LeftNetPotential
from .litefusion import LiteFusionConfig, LiteFusionPotential
from .mace import MACEConfig, MACEPotential
from .physnet import PhysNetConfig, PhysNetPotential
from .schnet import SchNetConfig, SchNetPotential
from .se3 import SE3TransformerConfig, SE3TransformerPotential
from .tensornet import TensorNetConfig, TensorNetPotential
from ..training.configs import ModelConfig


def build_potential_model(model_cfg: ModelConfig, species: Sequence[int]):
    """Construct a potential model from a :class:`ModelConfig`."""

    species_tuple = tuple(int(z) for z in species)
    name = model_cfg.name.lower()
    if name in {"transformer", "hybrid", "hybrid-potential", "soap-transformer"}:
        config = HybridPotentialConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            gcn_layers=model_cfg.gcn_layers,
            transformer_layers=model_cfg.transformer_layers,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            use_coordinate_features=model_cfg.use_coordinate_features,
            soap_num_radial=model_cfg.soap_num_radial,
            soap_cutoff=model_cfg.soap_cutoff,
            soap_gaussian_width=model_cfg.soap_gaussian_width,
            predict_forces=model_cfg.predict_forces,
        )
        return HybridPotential(config)
    if name in {"se3", "se3-transformer", "equivariant"}:
        config = SE3TransformerConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.se3_layers or model_cfg.transformer_layers,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            distance_embedding_dim=model_cfg.se3_distance_embedding,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return SE3TransformerPotential(config)
    if name == "schnet":
        config = SchNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_filters=model_cfg.schnet_num_filters or model_cfg.hidden_dim,
            num_interactions=model_cfg.schnet_num_interactions,
            num_gaussians=model_cfg.schnet_num_gaussians,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return SchNetPotential(config)
    if name in {"equiformer_v2", "equiformer", "equiformer-v2"}:
        config = EquiformerV2Config(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.se3_layers or model_cfg.transformer_layers,
            num_heads=model_cfg.num_heads,
            distance_embedding_dim=model_cfg.se3_distance_embedding,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return EquiformerV2Potential(config)
    if name == "dimenet":
        config = DimeNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_blocks=model_cfg.dimenet_num_blocks,
            num_radial=model_cfg.dimenet_num_radial,
            num_spherical=model_cfg.dimenet_num_spherical,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return DimeNetPotential(config)
    if name == "gemnet":
        config = GemNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_blocks=model_cfg.gemnet_num_blocks,
            num_radial=model_cfg.gemnet_num_radial,
            num_spherical=model_cfg.gemnet_num_spherical,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return GemNetPotential(config)
    if name == "tensornet":
        config = TensorNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.tensornet_num_layers,
            num_radial=model_cfg.tensornet_num_radial,
            direction_dim=model_cfg.tensornet_direction_dim,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return TensorNetPotential(config)
    if name == "physnet":
        config = PhysNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_blocks=model_cfg.physnet_num_blocks,
            num_basis=model_cfg.physnet_num_basis,
            cutoff=model_cfg.cutoff,
            predict_forces=model_cfg.predict_forces,
        )
        return PhysNetPotential(config)
    if name == "mace":
        config = MACEConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.mace_num_layers,
            num_radial=model_cfg.mace_num_radial,
            num_heads=model_cfg.num_heads,
            cutoff=model_cfg.cutoff,
            dropout=model_cfg.dropout,
            predict_forces=model_cfg.predict_forces,
        )
        return MACEPotential(config)
    if name == "litefusion":
        config = LiteFusionConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_blocks=model_cfg.litefusion_num_blocks,
            num_radial=model_cfg.litefusion_num_radial,
            num_gaussians=model_cfg.litefusion_num_gaussians,
            num_spherical=model_cfg.litefusion_num_spherical,
            rbf_dim=model_cfg.litefusion_rbf_dim,
            cutoff=model_cfg.cutoff,
            dropout=model_cfg.dropout,
            predict_forces=model_cfg.predict_forces,
        )
        return LiteFusionPotential(config)
    if name == "leftnet":
        config = LeftNetConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.leftnet_num_layers,
            num_radial=model_cfg.leftnet_num_radial,
            cutoff=model_cfg.cutoff,
            dropout=model_cfg.dropout,
            predict_forces=model_cfg.predict_forces,
        )
        return LeftNetPotential(config)
    if name == "gcn":
        config = HybridPotentialConfig(
            species=species_tuple,
            hidden_dim=model_cfg.hidden_dim,
            gcn_layers=model_cfg.gcn_layers,
            transformer_layers=0,
            num_heads=model_cfg.num_heads,
            ffn_dim=model_cfg.ffn_dim,
            dropout=model_cfg.dropout,
            cutoff=model_cfg.cutoff,
            use_coordinate_features=model_cfg.use_coordinate_features,
            soap_num_radial=model_cfg.soap_num_radial,
            soap_cutoff=model_cfg.soap_cutoff,
            soap_gaussian_width=model_cfg.soap_gaussian_width,
            predict_forces=model_cfg.predict_forces,
        )
        return HybridPotential(config)
    if name == "external":
        if not model_cfg.adapter:
            raise ValueError("External model requires an 'adapter' path to be provided")
        try:
            external_model = load_external_model(model_cfg.adapter)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "Failed to load external model. Install its dependencies or update the adapter path."
            ) from exc
        if model_cfg.adapter_weights is not None:
            checkpoint = torch.load(
                model_cfg.adapter_weights,
                map_location="cpu",
                weights_only=False,
            )
            try:
                external_model.load_state_dict(checkpoint)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive for custom loaders
                raise RuntimeError(
                    f"Unable to load weights from {model_cfg.adapter_weights}: {exc}"
                ) from exc
        return PotentialModelAdapter(
            external_model,
            neighbor_strategy=model_cfg.neighbor_strategy,
            neighbor_cutoff=model_cfg.neighbor_cutoff or model_cfg.cutoff,
        )
    raise ValueError(f"Unsupported potential model '{model_cfg.name}'")


__all__ = ["build_potential_model"]
