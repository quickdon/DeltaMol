"""Configuration management helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, Union, get_args, get_origin

try:
    import yaml
except ImportError as exc:
    yaml = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

T = TypeVar("T")


def _serialise_value(value: Any) -> Any:
    """Convert dataclass values to YAML friendly structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _serialise_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialise_value(item) for item in value]
    return value


def save_config(config: Any, path: Path) -> None:
    if yaml is None:
        raise ImportError("pyyaml is required to save configs") from _IMPORT_ERROR
    data = _serialise_value(asdict(config) if is_dataclass(config) else config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def load_config(path: Path, cls: Type[T]) -> T:
    if yaml is None:
        raise ImportError("pyyaml is required to load configs") from _IMPORT_ERROR
    data = yaml.safe_load(path.read_text())
    if is_dataclass(cls):
        if not isinstance(data, Mapping):
            raise TypeError("Configuration file must contain a mapping at the top level")
        return _build_dataclass(cls, data)
    return cls(**data)


def _build_dataclass(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Instantiate ``cls`` from the mapping ``data``."""

    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _coerce_value(field.type, data[field.name])
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    """Convert ``value`` so it matches the provided type annotation."""

    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is None:
        if annotation is Path:
            return Path(value)
        if is_dataclass(annotation) and isinstance(value, Mapping):
            return _build_dataclass(annotation, value)
        return value
    args = get_args(annotation)
    if origin in (list, Sequence):
        inner = args[0] if args else Any
        return [_coerce_value(inner, item) for item in value]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            return tuple(_coerce_value(inner, item) for item in value)
        return tuple(_coerce_value(arg, item) for arg, item in zip(args, value))
    if origin in (dict, Mapping):
        key_type, val_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key_type, key): _coerce_value(val_type, item)
            for key, item in value.items()
        }
    if origin is Union:
        for option in args:
            if option is type(None):  # noqa: E721 - Optional detection
                continue
            try:
                return _coerce_value(option, value)
            except Exception:
                continue
        return value
    return value
