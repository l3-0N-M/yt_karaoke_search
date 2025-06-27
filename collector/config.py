"""Configuration models using Pydantic for validation."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    import json as yaml  # type: ignore

class DatabaseConfig(BaseModel):
    """Database configuration with backup settings."""
    path: str = "karaoke_videos.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    vacuum_threshold_mb: int = 100
    vacuum_on_startup: bool = False

class ScrapingConfig(BaseModel):
    """Scraping configuration with performance tuning."""
    max_concurrent_workers: int = Field(default=5, ge=1, le=20)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=60, ge=5, le=180)  # Increased for large playlists
    
    user_agents: List[str] = Field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ])

class DataSourceConfig(BaseModel):
    """External data source configuration."""
    ryd_api_enabled: bool = True
    ryd_api_url: str = "https://returnyoutubedislikeapi.com/votes"
    ryd_timeout: int = 10
    ryd_confidence_threshold: float = 0.1
    
    # Music metadata APIs
    musicbrainz_enabled: bool = True
    musicbrainz_timeout: int = 5
    musicbrainz_user_agent: str = "KaraokeCollector/2.1 (https://github.com/your/repo)"

class SearchConfig(BaseModel):
    """Search configuration and query categories."""
    primary_method: str = Field(default="yt_dlp", regex="^(yt_dlp|hybrid)$")
    max_results_per_query: int = Field(default=100, ge=1, le=500)
    
    search_categories: Dict[str, List[str]] = Field(default_factory=lambda: {
        'general': ['karaoke', 'karaoke version', 'sing along'],
        'features': ['karaoke with lyrics', 'karaoke instrumental', 'karaoke backing track'],
        'instruments': ['piano karaoke', 'guitar karaoke', 'acoustic karaoke'],
        'genres': ['pop karaoke', 'rock karaoke', 'country karaoke', 'R&B karaoke'],
        'decades': ['70s karaoke', '80s karaoke', '90s karaoke', '2000s karaoke', '2010s karaoke'],
        'quality': ['HD karaoke', '4K karaoke', 'high quality karaoke']
    })

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$")
    file_path: str = "karaoke_collector.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    console_output: bool = True

class UIConfig(BaseModel):
    """User interface configuration."""
    show_progress_bar: bool = True
    progress_update_interval: int = 10
    save_thumbnails: bool = False
    thumbnail_directory: str = "thumbnails"

class CollectorConfig(BaseModel):
    """Main configuration model."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    incremental_mode: bool = True
    skip_existing: bool = True
    dry_run: bool = False

def load_config(config_path: Optional[str] = None) -> CollectorConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        return CollectorConfig(**config_data)
    return CollectorConfig()

def save_config_template(output_path: str = "config_template.yaml"):
    """Save a template configuration file."""
    config = CollectorConfig()
    config_dict = config.dict()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Configuration template saved to: {output_path}")