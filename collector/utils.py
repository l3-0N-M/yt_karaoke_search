"""Utilities and helper functions."""

import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging(level=logging.INFO, log_file="karaoke_collector.log"):
    """Setup comprehensive logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=50*1024*1024, backupCount=5  # 50MB max, 5 backups
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('yt_dlp').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)