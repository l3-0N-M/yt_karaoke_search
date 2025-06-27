"""
Karaoke Video Collector

A comprehensive tool for collecting karaoke video data from YouTube
with confidence scoring and Return YouTube Dislike integration.
"""

__version__ = "2.1.0"
__author__ = "Karaoke Collector Team"

from .config import CollectorConfig
from .db import DatabaseManager
from .main import KaraokeCollector

__all__ = ["CollectorConfig", "KaraokeCollector", "DatabaseManager"]
