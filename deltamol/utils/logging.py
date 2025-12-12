"""Logging helpers."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from .distributed import get_distributed_state


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

    distributed_state = get_distributed_state()
    if distributed_state is not None:
        is_main_process = distributed_state.is_main_process()
    else:
        rank_env = os.environ.get("RANK")
        try:
            rank = int(rank_env) if rank_env is not None else None
        except ValueError:
            rank = None
        try:
            main_rank = int(os.environ.get("MAIN_PROCESS", "0"))
        except ValueError:
            main_rank = 0
        is_main_process = rank is None or rank == main_rank

    handlers = [logging.StreamHandler()]
    if is_main_process:
        if resume:
            existing_logs = list(output_dir.glob("training-*.log"))
            log_path = max(existing_logs, key=lambda path: path.stat().st_mtime) if existing_logs else _timestamped_log_path(output_dir)
        else:
            log_path = _timestamped_log_path(output_dir)
        handlers.insert(0, logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
