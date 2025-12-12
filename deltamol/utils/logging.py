"""Logging helpers."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_logging(
    output_dir: Path, level: int = logging.INFO, *, resume: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    if log_path.exists() and not resume:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archived = output_dir / f"training-{timestamp}.log"
        counter = 1
        while archived.exists():
            counter += 1
            archived = output_dir / f"training-{timestamp}-{counter}.log"
        log_path.rename(archived)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
