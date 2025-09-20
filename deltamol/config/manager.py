"""Configuration management helpers."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Type, TypeVar

try:
    import yaml
except ImportError as exc:
    yaml = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

T = TypeVar("T")


def save_config(config: Any, path: Path) -> None:
    if yaml is None:
        raise ImportError("pyyaml is required to save configs") from _IMPORT_ERROR
    data = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def load_config(path: Path, cls: Type[T]) -> T:
    if yaml is None:
        raise ImportError("pyyaml is required to load configs") from _IMPORT_ERROR
    data = yaml.safe_load(path.read_text())
    return cls(**data)
