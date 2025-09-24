"""Logging helpers."""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
