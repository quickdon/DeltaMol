"""Logging helpers."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def _timestamped_log_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = output_dir / f"training-{timestamp}.log"
    counter = 1
    while log_path.exists():
        counter += 1
        log_path = output_dir / f"training-{timestamp}-{counter}.log"
    return log_path


def configure_logging(
    output_dir: Path, level: int = logging.INFO, *, resume: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        existing_logs = list(output_dir.glob("training-*.log"))
        log_path = max(existing_logs, key=lambda path: path.stat().st_mtime) if existing_logs else _timestamped_log_path(output_dir)
    else:
        log_path = _timestamped_log_path(output_dir)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
        force=True,
    )
