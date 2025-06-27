"""Utilities and helper functions."""

import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: int = logging.INFO,
    log_file: str = "karaoke_collector.log",
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """Setup comprehensive logging configuration.

    Parameters
    ----------
    level: int
        Logging level.
    log_file: str
        Path to the log file.
    max_bytes: int
        Maximum size in bytes before rotating the log file.
    backup_count: int
        Number of rotated log files to keep.
    console_output: bool
        Whether to also log to the console.
    """

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
