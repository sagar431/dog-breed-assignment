from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger


def configure_cpu(cpu_threads: int) -> None:
    cpu_threads = max(1, cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)


def configure_logging(logs_dir: str | Path) -> Path:
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "assignment.log"
    logger.add(log_path, level="INFO", rotation="10 MB", retention=5)
    return log_path


def parse_devices(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def batch_limit(value: int) -> int | None:
    return None if value <= 0 else value

