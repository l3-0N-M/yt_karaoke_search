"""Utilities and helper functions."""

import asyncio
import logging
import re
import sys
import time
import unicodedata
from logging.handlers import RotatingFileHandler
from typing import Optional


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

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
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


def parse_duration(duration_str: str) -> int:
    """Parse a duration string (e.g., 'PT1M30S', '1:30') into seconds."""
    if not duration_str:
        return 0

    if duration_str.startswith("PT"):
        # ISO 8601 format (e.g., PT1H2M3S)
        import isodate  # type: ignore

        try:
            duration = isodate.parse_duration(duration_str)
            return int(duration.total_seconds())
        except isodate.ISO8601Error:
            return 0
    elif ":" in duration_str:
        # HH:MM:SS or MM:SS format
        parts = list(map(int, duration_str.split(":")))
        if len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        else:
            return 0
    else:
        # Assume it's already in seconds
        try:
            return int(duration_str)
        except ValueError:
            return 0


def format_duration(seconds: int) -> str:
    """Format seconds into a human-readable string (HH:MM:SS)."""
    if seconds < 0:
        return "00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"


def get_project_root() -> str:
    """Get the project root directory."""
    from pathlib import Path

    return str(Path(__file__).parent.parent.resolve())


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found at %s", config_path)
        return {}
    except yaml.YAMLError as e:
        logging.error("Error parsing YAML configuration: %s", e)
        return {}


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge two configurations, with the override taking precedence."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


# Normalization patterns for artist and song titles
ARTIST_NORMALIZATIONS = {
    # Articles
    r"^the\s+": "",
    r"^a\s+": "",
    r"^an\s+": "",
    # Punctuation and symbols
    r"[&]": "and",
    r"[\\.,\\-_]": " ",
    r"['\"\"]": "",
    r"\\s+": " ",
    # Common abbreviations
    r"\\bft\\.?\\b": "featuring",
    r"\\bfeat\\.?\\b": "featuring",
    r"\\bvs\\.?\\b": "versus",
    r"\\bw/\\b": "with",
}

SONG_NORMALIZATIONS = {
    # Remove parentheticals that don't affect matching
    r"\\s*\\([^)]*(?:remix|edit|version|mix|remaster)[^)]*\\)": "",
    r"\\s*\\([^)]*(?:live|acoustic|demo|instrumental)[^)]*\\)": "",
    r"\\s*\[[^\]]*(?:remix|edit|version|mix|remaster)[^\]]*\]": "",
    # Normalize punctuation
    r"['\"\"]": "",
    r"[\\-_]": " ",
    r"\\s+": " ",
}

_NORMALIZATION_CACHE = {}


def normalize_text(text: str, text_type: str = "general") -> str:
    """Normalize text for better matching."""
    if not isinstance(text, str):
        text = str(text)

    cache_key = (text_type, text)
    if cache_key in _NORMALIZATION_CACHE:
        return _NORMALIZATION_CACHE[cache_key]

    # Unicode normalization
    normalized = unicodedata.normalize("NFKD", text.lower().strip())

    # Remove accents
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    # Apply type-specific normalizations
    patterns = {}
    if text_type == "artist":
        patterns = ARTIST_NORMALIZATIONS
    elif text_type == "song":
        patterns = SONG_NORMALIZATIONS
    else:
        patterns = {**ARTIST_NORMALIZATIONS, **SONG_NORMALIZATIONS}

    for pattern, replacement in patterns.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    normalized = normalized.strip()
    _NORMALIZATION_CACHE[cache_key] = normalized
    return normalized


class DiscogsRateLimiter:
    """Rate limiter for Discogs API requests following their limits.
    
    Discogs API limits:
    - 60 requests per minute for authenticated requests
    - 25 requests per minute for unauthenticated requests  
    - Initial burst of 5 requests allowed
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0
        
        self.tokens = 5.0  # Initial burst tokens
        self.max_tokens = 5.0  # Maximum burst tokens
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
        self.logger = logging.getLogger(__name__)
        
    async def wait_for_request(self) -> None:
        """Wait until a request can be made according to rate limits."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= 1.0:
                # We have tokens available
                self.tokens -= 1.0
                return
            
            # Need to wait for next token
            wait_time = (1.0 - self.tokens) / self.requests_per_second
            self.logger.debug(f"Rate limit hit, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.tokens = 0.0
    
    def get_remaining_tokens(self) -> float:
        """Get the current number of available tokens."""
        now = time.time()
        elapsed = now - self.last_update
        
        return min(
            self.max_tokens,
            self.tokens + elapsed * self.requests_per_second
        )
    
    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        self.tokens = self.max_tokens
        self.last_update = time.time()
