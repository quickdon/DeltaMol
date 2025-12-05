"""Utilities for configuring and interacting with distributed training."""
from __future__ import annotations

import datetime as _dt
import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

try:  # pragma: no cover - optional distributed import
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch without distributed
    dist = None  # type: ignore[assignment]

from torch.utils.data.distributed import DistributedSampler

LOGGER = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration options that describe a distributed process group."""

    enabled: bool = False
    backend: Optional[str] = None
    init_method: Optional[str] = None
    world_size: Optional[Union[int, str]] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    main_process: int = 0
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    timeout_minutes: float = 30.0
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    auto_discover: bool = True
    devices: Optional[Union[Sequence[int], str]] = None


@dataclass
class DistributedState:
    """Lightweight view over the active distributed process group."""

    config: DistributedConfig
    enabled: bool
    backend: Optional[str]
    world_size: int
    rank: int
    local_rank: int
    device: torch.device

    def is_main_process(self) -> bool:
        return self.rank == self.config.main_process

    def barrier(self) -> None:
        if self.enabled and _dist_available():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: "dist.ReduceOp" = None) -> torch.Tensor:  # type: ignore[name-defined]
        if not self.enabled or not _dist_available():
            return tensor
        reduce_op = op or dist.ReduceOp.SUM
        dist.all_reduce(tensor, op=reduce_op)
        return tensor

    def build_sampler(self, dataset, *, shuffle: bool, drop_last: bool = False) -> Optional[DistributedSampler]:
        if not self.enabled or self.world_size <= 1:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def sync_module_state(self, module: torch.nn.Module) -> None:
        if not self.enabled or not _dist_available():
            return
        state = module.state_dict()
        objects = [state] if self.is_main_process() else [None]
        dist.broadcast_object_list(objects, src=self.config.main_process)
        if not self.is_main_process():
            module.load_state_dict(objects[0])

    def broadcast_object(self, obj):  # type: ignore[override]
        if not self.enabled or not _dist_available():
            return obj
        payload = [obj] if self.is_main_process() else [None]
        dist.broadcast_object_list(payload, src=self.config.main_process)
        return payload[0]


_GLOBAL_STATE: Optional[DistributedState] = None


def _dist_available() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def get_distributed_state() -> Optional[DistributedState]:
    return _GLOBAL_STATE


def register_distributed_state(state: DistributedState) -> DistributedState:
    global _GLOBAL_STATE
    _GLOBAL_STATE = state
    return state


def is_main_process() -> bool:
    state = get_distributed_state()
    if state is None:
        return True
    return state.is_main_process()


def init_distributed(config: DistributedConfig, device: torch.device) -> DistributedState:
    """Initialise a distributed process group and register global state."""

    requested = config.enabled
    if not requested and config.auto_discover:
        requested = int(os.environ.get("WORLD_SIZE", "1")) > 1
    explicit_devices = _normalise_devices(config.devices)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        if _GLOBAL_STATE.enabled or not requested:
            return _GLOBAL_STATE
        # Promote previously disabled state to an active distributed group if requested now.
        _GLOBAL_STATE = None
    resolved_local_rank = config.local_rank
    if resolved_local_rank is None:
        local_rank_env = os.environ.get("LOCAL_RANK")
        resolved_local_rank = int(local_rank_env) if local_rank_env is not None else None
    resolved_rank = config.rank if config.rank is not None else int(os.environ.get("RANK", "0"))
    if resolved_local_rank is None:
        resolved_local_rank = resolved_rank
    world_size = _resolve_world_size(config.world_size, world_size_env, explicit_devices)
    resolved_device_index = None
    if explicit_devices is not None:
        resolved_device_index = _select_device_index(explicit_devices, resolved_local_rank)
    elif requested and resolved_local_rank is not None:
        resolved_device_index = resolved_local_rank
    if not requested:
        resolved_device = _resolve_device(device, local_rank=resolved_device_index)
        if config.enabled:
            LOGGER.info(
                "Distributed training requested but auto-discovery resolved world size=1; "
                "running single-process on %s",
                resolved_device,
            )
        return register_distributed_state(
            DistributedState(
                config=config,
                enabled=False,
                backend=None,
                world_size=1,
                rank=0,
                local_rank=0,
                device=resolved_device,
            )
        )

    if dist is None or not dist.is_available():
        raise RuntimeError("Distributed training requested but torch.distributed is not available")

    rank = resolved_rank
    local_rank = resolved_local_rank
    if explicit_devices is not None and world_size > len(explicit_devices):
        LOGGER.warning(
            "World size %d is larger than the number of configured devices (%d); "
            "devices will be cycled across ranks",
            world_size,
            len(explicit_devices),
        )
    if world_size <= 1:
        resolved_device = _resolve_device(device, local_rank=resolved_device_index)
        if rank == config.main_process:
            LOGGER.info(
                "Distributed training requested but world size resolved to 1; "
                "running single-process on %s",
                resolved_device,
            )
        return register_distributed_state(
            DistributedState(
                config=config,
                enabled=False,
                backend=None,
                world_size=1,
                rank=0,
                local_rank=0,
                device=resolved_device,
            )
        )

    backend = config.backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() and device.type == "cuda" else "gloo"
    _ensure_default_rendezvous(config)
    init_kwargs = {
        "backend": backend,
        "world_size": world_size,
        "rank": rank,
    }
    if config.init_method is not None:
        init_kwargs["init_method"] = config.init_method
    timeout_seconds = max(config.timeout_minutes, 0.1) * 60.0
    init_kwargs["timeout"] = _dt.timedelta(seconds=timeout_seconds)
    if not dist.is_initialized():
        dist.init_process_group(**init_kwargs)
    resolved_device_index = _select_device_index(explicit_devices, local_rank)
    resolved_device = _resolve_device(device, local_rank=resolved_device_index)
    state = DistributedState(
        config=config,
        enabled=True,
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=resolved_device_index if resolved_device_index is not None else local_rank,
        device=resolved_device,
    )
    register_distributed_state(state)
    if rank == config.main_process:
        LOGGER.info(
            "Initialised distributed process group | backend=%s, world_size=%d, rank=%d, "
            "local_rank=%d, device=%s",
            backend,
            world_size,
            rank,
            state.local_rank,
            resolved_device,
        )
    return state


def _resolve_device(device: torch.device, *, local_rank: Optional[int]) -> torch.device:
    if device.type == "cuda":
        if local_rank is not None:
            resolved = torch.device("cuda", local_rank)
            torch.cuda.set_device(resolved)
            return resolved
        if device.index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
            return torch.device("cuda", index)
    return device


def _normalise_devices(devices: Optional[Union[Sequence[int], str]]) -> Optional[Sequence[int]]:
    if devices is None:
        return None
    if isinstance(devices, str):
        tokenised = [item.strip() for item in devices.split(",") if item.strip()]
        if len(tokenised) == 1 and tokenised[0].lower() == "all":
            if not torch.cuda.is_available():
                LOGGER.warning("Device selection 'all' requested but CUDA is not available")
                return None
            return list(range(torch.cuda.device_count()))
        return [int(item) for item in tokenised]
    return list(devices)


def _resolve_world_size(
    world_size: Optional[Union[int, str]], env_world_size: int, devices: Optional[Sequence[int]]
) -> int:
    if env_world_size > 1:
        return env_world_size
    if isinstance(world_size, str) and world_size.lower() == "all":
        if devices is not None:
            return max(len(devices), 1)
        if torch.cuda.is_available():
            return max(torch.cuda.device_count(), 1)
        return 1
    if isinstance(world_size, int):
        return max(world_size, 1)
    if devices is not None:
        return max(len(devices), 1)
    return max(env_world_size, 1)


def _select_device_index(devices: Optional[Sequence[int]], local_rank: Optional[int]) -> Optional[int]:
    if devices is None:
        return local_rank
    if local_rank is None:
        return devices[0] if devices else None
    if not devices:
        return local_rank
    return devices[local_rank % len(devices)]


def _ensure_default_rendezvous(config: DistributedConfig) -> None:
    if config.init_method is not None:
        return
    if config.master_addr is not None:
        os.environ.setdefault("MASTER_ADDR", config.master_addr)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if config.master_port is not None:
        os.environ.setdefault("MASTER_PORT", str(config.master_port))
    else:
        os.environ.setdefault("MASTER_PORT", "29500")


__all__ = [
    "DistributedConfig",
    "DistributedState",
    "DistributedSampler",
    "get_distributed_state",
    "init_distributed",
    "is_main_process",
]
