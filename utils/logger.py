"""
Structured logging via loguru.
All modules import `log` from here — one call to configure the whole pipeline.
"""
import sys
import os
from loguru import logger as log

import config


def setup_logging() -> None:
    log.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console
    log.add(sys.stdout, level=config.LOG_LEVEL, format=fmt, colorize=True)

    # Rotating file — 50 MB per file, keep 10 files
    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)
    log.add(
        config.LOG_FILE,
        level="DEBUG",
        format=fmt,
        rotation="50 MB",
        retention=10,
        compression="gz",
        enqueue=True,          # thread-safe
    )

    log.info("Logging initialised — level={}", config.LOG_LEVEL)


__all__ = ["log", "setup_logging"]
