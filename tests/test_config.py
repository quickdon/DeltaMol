from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from deltamol.config.manager import load_config, save_config


@dataclass
class InnerConfig:
    path: Path
    values: tuple[int, ...]


@dataclass
class OuterConfig:
    inner: InnerConfig
    names: tuple[str, ...]


def test_nested_config_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"
    original = OuterConfig(
        inner=InnerConfig(path=tmp_path / "data", values=(1, 2, 3)),
        names=("alpha", "beta"),
    )

    save_config(original, config_path)
    loaded = load_config(config_path, OuterConfig)

    assert loaded == original
