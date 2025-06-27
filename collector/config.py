"""Configuration models using simple dataclasses."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    import json as yaml  # type: ignore


@dataclass
class DatabaseConfig:
    """Database configuration with backup settings."""

    path: str = "karaoke_videos.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    vacuum_threshold_mb: int = 100
    vacuum_on_startup: bool = False


@dataclass
class ScrapingConfig:
    """Scraping configuration with performance tuning."""

    max_concurrent_workers: int = 5
    max_retries: int = 3
    timeout_seconds: int = 60  # Increased for large playlists

    user_agents: List[str] = field(
        default_factory=lambda: [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
    )


@dataclass
class DataSourceConfig:
    """External data source configuration."""

    ryd_api_enabled: bool = True
    ryd_api_url: str = "https://returnyoutubedislikeapi.com/votes"
    ryd_timeout: int = 10
    ryd_confidence_threshold: float = 0.1

    # Music metadata APIs
    musicbrainz_enabled: bool = True
    musicbrainz_timeout: int = 5
    musicbrainz_user_agent: str = "KaraokeCollector/2.1 (https://github.com/your/repo)"


@dataclass
class SearchConfig:
    """Search configuration and query categories."""

    primary_method: str = "yt_dlp"
    max_results_per_query: int = 100

    search_categories: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "general": ["karaoke", "karaoke version", "sing along"],
            "features": ["karaoke with lyrics", "karaoke instrumental", "karaoke backing track"],
            "instruments": ["piano karaoke", "guitar karaoke", "acoustic karaoke"],
            "genres": ["pop karaoke", "rock karaoke", "country karaoke", "R&B karaoke"],
            "decades": [
                "70s karaoke",
                "80s karaoke",
                "90s karaoke",
                "2000s karaoke",
                "2010s karaoke",
            ],
            "quality": ["HD karaoke", "4K karaoke", "high quality karaoke"],
        }
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file_path: str = "karaoke_collector.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    console_output: bool = True


@dataclass
class UIConfig:
    """User interface configuration."""

    show_progress_bar: bool = True
    progress_update_interval: int = 10
    save_thumbnails: bool = False
    thumbnail_directory: str = "thumbnails"


@dataclass
class CollectorConfig:
    """Main configuration model."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    incremental_mode: bool = True
    skip_existing: bool = True
    dry_run: bool = False


def load_config(config_path: Optional[str] = None) -> CollectorConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return CollectorConfig(
            database=DatabaseConfig(**config_data.get("database", {})),
            scraping=ScrapingConfig(**config_data.get("scraping", {})),
            data_sources=DataSourceConfig(**config_data.get("data_sources", {})),
            search=SearchConfig(**config_data.get("search", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            ui=UIConfig(**config_data.get("ui", {})),
            incremental_mode=config_data.get("incremental_mode", True),
            skip_existing=config_data.get("skip_existing", True),
            dry_run=config_data.get("dry_run", False),
        )
    return CollectorConfig()


def save_config_template(output_path: str = "config_template.yaml"):
    """Save a template configuration file."""
    config = CollectorConfig()
    config_dict = asdict(config)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print(f"Configuration template saved to: {output_path}")
