"""
Karaoke Video Collector

A comprehensive tool for collecting karaoke video data from YouTube
with confidence scoring and Return YouTube Dislike integration.
"""

__version__ = "2.0.0"
__author__ = "Karaoke Collector Team"

from .config import CollectorConfig
from .main import KaraokeCollector
from .db import DatabaseManager

__all__ = ["CollectorConfig", "KaraokeCollector", "DatabaseManager"]