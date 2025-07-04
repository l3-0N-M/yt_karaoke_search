"""Configuration models using simple dataclasses."""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

try:
    import yaml  # type: ignore

    YamlError = getattr(yaml, "YAMLError", Exception)
except ImportError:  # pragma: no cover - optional dependency
    import json

    class yaml:
        @staticmethod
        def safe_load(f):
            return json.load(f)

        @staticmethod
        def dump(data, f, default_flow_style=False, indent=2):
            json.dump(data, f, indent=indent)


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration with backup settings."""

    path: str = "karaoke_videos.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    vacuum_threshold_mb: int = 100
    vacuum_on_startup: bool = False
    connection_pool_size: int = 10
    connection_timeout: float = 30.0


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

    # Discogs API configuration
    discogs_enabled: bool = True
    discogs_token: str = ""
    discogs_user_agent: str = "KaraokeCollector/2.1 +https://github.com/karaoke/search"
    discogs_timeout: int = 10
    discogs_requests_per_minute: int = 60
    discogs_use_as_fallback: bool = True
    discogs_min_musicbrainz_confidence: float = 0.6
    discogs_max_results_per_search: int = 20  # Increased from 10 for better coverage
    discogs_confidence_threshold: float = 0.4  # Lowered from 0.5 for better coverage


@dataclass
class FuzzyMatchingConfig:
    """Fuzzy matching configuration."""

    min_similarity: float = 0.7
    min_phonetic: float = 0.8
    max_edit_distance: int = 3


@dataclass
class MultiPassPassConfig:
    """Configuration for a single pass in the multi-pass ladder."""

    enabled: bool = True
    confidence_threshold: float = 0.8
    timeout_seconds: float = 30.0
    max_retries: int = 3
    cpu_budget_limit: float = 10.0  # seconds
    api_budget_limit: int = 10  # API calls
    exponential_backoff_base: float = 2.0
    exponential_backoff_max: float = 60.0


@dataclass
class MultiPassConfig:
    """Configuration for the multi-pass parsing ladder."""

    # Global settings
    enabled: bool = True  # Enabled by default for enhanced parsing
    max_total_retries: int = 5
    global_timeout_seconds: float = 300.0
    stop_on_first_success: bool = True
    always_enrich_metadata: bool = True  # Continue to enrichment even after successful parsing
    require_metadata: bool = True  # Whether genre/year metadata is required

    # Budget management
    total_cpu_budget: float = 60.0  # seconds per video
    total_api_budget: int = 100  # API calls per video

    # Backoff and retry
    base_retry_delay: float = 1.0
    max_retry_delay: float = 300.0
    retry_exponential_base: float = 2.0

    # Per-pass configurations in optimized priority order
    channel_template: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.75,
            timeout_seconds=10.0,
            cpu_budget_limit=2.0,
            api_budget_limit=0,
        )
    )
    musicbrainz_search: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.65,
            timeout_seconds=30.0,
            cpu_budget_limit=5.0,
            api_budget_limit=10,
        )
    )
    discogs_search: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.6,
            timeout_seconds=20.0,
            cpu_budget_limit=3.0,
            api_budget_limit=5,
        )
    )
    musicbrainz_validation: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.8,
            timeout_seconds=20.0,
            cpu_budget_limit=3.0,
            api_budget_limit=5,
        )
    )
    ml_embedding: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.7,
            timeout_seconds=60.0,
            cpu_budget_limit=10.0,
            api_budget_limit=5,
        )
    )
    auto_retemplate: MultiPassPassConfig = field(
        default_factory=lambda: MultiPassPassConfig(
            confidence_threshold=0.6, timeout_seconds=30.0, cpu_budget_limit=5.0, api_budget_limit=2
        )
    )


@dataclass
class SearchConfig:
    """Search configuration and query categories."""

    primary_method: str = "yt_dlp"
    max_results_per_query: int = 100
    use_multi_strategy: bool = False

    # Enhanced search settings
    fallback_threshold: int = 10
    max_fallback_providers: int = 2
    enable_parallel_search: bool = False

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

    # Optional regex patterns for extracting artist and song from titles
    title_patterns: List[str] = field(default_factory=list)

    # Advanced parser settings
    use_advanced_parser: bool = True
    enable_fuzzy_matching: bool = True
    enable_pattern_learning: bool = True
    min_confidence_threshold: float = 0.5
    enable_multi_language: bool = True

    # Fuzzy matching configuration
    fuzzy_matching: FuzzyMatchingConfig = field(default_factory=FuzzyMatchingConfig)

    # Multi-pass parsing configuration
    multi_pass: MultiPassConfig = field(default_factory=MultiPassConfig)


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


def _filter_fields(data: Dict[str, Any], cls: Type[Any]) -> Dict[str, Any]:
    """Return only keys present on the dataclass to avoid TypeErrors."""
    valid_fields = cls.__dataclass_fields__.keys()
    return {k: v for k, v in data.items() if k in valid_fields}


def validate_config(cfg: CollectorConfig) -> None:
    """Enhanced validation with bounds checking and security validation."""
    # Database validation with reasonable bounds
    if not (1 <= cfg.database.backup_interval_hours <= 8760):  # 1 hour to 1 year
        raise ValueError("backup_interval_hours must be between 1 and 8760 (1 year)")
    if cfg.database.backup_retention_days < 0:
        raise ValueError("backup_retention_days cannot be negative")
    if not (1 <= cfg.database.vacuum_threshold_mb <= 100000):  # 1MB to 100GB
        raise ValueError("vacuum_threshold_mb must be between 1 and 100000 MB")

    # Scraping validation with reasonable bounds to prevent resource exhaustion
    if not (1 <= cfg.scraping.max_concurrent_workers <= 50):
        raise ValueError("max_concurrent_workers must be between 1 and 50")
    if not (0 <= cfg.scraping.max_retries <= 10):
        raise ValueError("max_retries must be between 0 and 10")
    if not (1 <= cfg.scraping.timeout_seconds <= 3600):  # Max 1 hour
        raise ValueError("timeout_seconds must be between 1 and 3600")

    # Data sources validation
    if not 0 <= cfg.data_sources.ryd_confidence_threshold <= 1:
        raise ValueError("ryd_confidence_threshold must be between 0 and 1")

    # Discogs validation
    if not 0 <= cfg.data_sources.discogs_confidence_threshold <= 1:
        raise ValueError("discogs_confidence_threshold must be between 0 and 1")
    if not 0 <= cfg.data_sources.discogs_min_musicbrainz_confidence <= 1:
        raise ValueError("discogs_min_musicbrainz_confidence must be between 0 and 1")
    if not (1 <= cfg.data_sources.discogs_requests_per_minute <= 300):
        raise ValueError("discogs_requests_per_minute must be between 1 and 300")
    if not (1 <= cfg.data_sources.discogs_max_results_per_search <= 100):
        raise ValueError("discogs_max_results_per_search must be between 1 and 100")
    if not (1 <= cfg.data_sources.discogs_timeout <= 60):
        raise ValueError("discogs_timeout must be between 1 and 60 seconds")

    # URL validation for security
    def validate_url(url: str, name: str):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"{name} must be a valid HTTP/HTTPS URL")
            if not parsed.netloc:
                raise ValueError(f"{name} must have a valid hostname")
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")

    validate_url(cfg.data_sources.ryd_api_url, "ryd_api_url")

    # Search validation with reasonable bounds
    if not (1 <= cfg.search.max_results_per_query <= 1000):
        raise ValueError("max_results_per_query must be between 1 and 1000")

    # Multi-pass validation
    if cfg.search.multi_pass.enabled:
        if not (1 <= cfg.search.multi_pass.max_total_retries <= 20):
            raise ValueError("multi_pass.max_total_retries must be between 1 and 20")
        if not (1 <= cfg.search.multi_pass.global_timeout_seconds <= 3600):
            raise ValueError("multi_pass.global_timeout_seconds must be between 1 and 3600")
        if not (1 <= cfg.search.multi_pass.total_cpu_budget <= 600):
            raise ValueError("multi_pass.total_cpu_budget must be between 1 and 600 seconds")
        if not (1 <= cfg.search.multi_pass.total_api_budget <= 1000):
            raise ValueError("multi_pass.total_api_budget must be between 1 and 1000")

    # Fuzzy matching validation
    if not (0.1 <= cfg.search.fuzzy_matching.min_similarity <= 1.0):
        raise ValueError("fuzzy_matching.min_similarity must be between 0.1 and 1.0")
    if not (0.1 <= cfg.search.fuzzy_matching.min_phonetic <= 1.0):
        raise ValueError("fuzzy_matching.min_phonetic must be between 0.1 and 1.0")
    if not (1 <= cfg.search.fuzzy_matching.max_edit_distance <= 10):
        raise ValueError("fuzzy_matching.max_edit_distance must be between 1 and 10")

    # Logging validation
    if not (1 <= cfg.logging.max_file_size_mb <= 1000):  # 1MB to 1GB
        raise ValueError("max_file_size_mb must be between 1 and 1000 MB")
    if not (0 <= cfg.logging.backup_count <= 100):
        raise ValueError("backup_count must be between 0 and 100")

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if cfg.logging.level.upper() not in valid_levels:
        raise ValueError(f"logging.level must be one of: {valid_levels}")

    # UI validation
    if not (0.1 <= cfg.ui.progress_update_interval <= 60):  # 0.1s to 1 minute
        raise ValueError("progress_update_interval must be between 0.1 and 60 seconds")

    # Path validation (basic security check)
    def validate_path(path: str, name: str):
        try:
            resolved = Path(path).resolve()
            # Warn about potentially unsafe paths
            if ".." in str(resolved) or str(resolved).count("/") > 10:
                logger.warning(f"{name} path may be unsafe: {path}")
        except Exception as e:
            raise ValueError(f"Invalid {name} path: {e}")

    validate_path(cfg.database.path, "database.path")
    validate_path(cfg.logging.file_path, "logging.file_path")


def load_config(config_path: Optional[str] = None) -> CollectorConfig:
    """Load configuration from YAML file or return defaults."""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Handle None/empty/non-dict configs
            if config_data is None:
                logger.warning(f"Configuration file {config_path} is empty, using defaults")
                config_data = {}
            elif not isinstance(config_data, dict):
                raise ValueError(
                    f"Configuration file must contain a dictionary, got {type(config_data).__name__}"
                )

        except YamlError as e:
            raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}")
        except (IOError, OSError) as e:
            raise ValueError(f"Cannot read configuration file {config_path}: {e}")

        try:
            # Handle nested search configuration
            search_data = config_data.get("search", {})

            # Extract fuzzy matching config
            fuzzy_data = search_data.pop("fuzzy_matching", {})
            fuzzy_config = FuzzyMatchingConfig(**_filter_fields(fuzzy_data, FuzzyMatchingConfig))

            # Extract multi-pass config
            multi_pass_data = search_data.pop("multi_pass", {})

            # Extract per-pass configs
            pass_configs = {}
            for pass_name in [
                "channel_template",
                "auto_retemplate",
                "ml_embedding",
                "discogs_search",
            ]:
                pass_data = multi_pass_data.pop(pass_name, {})
                pass_configs[pass_name] = MultiPassPassConfig(
                    **_filter_fields(pass_data, MultiPassPassConfig)
                )

            multi_pass_config = MultiPassConfig(
                **_filter_fields(multi_pass_data, MultiPassConfig), **pass_configs
            )

            cfg = CollectorConfig(
                database=DatabaseConfig(
                    **_filter_fields(config_data.get("database", {}), DatabaseConfig)
                ),
                scraping=ScrapingConfig(
                    **_filter_fields(config_data.get("scraping", {}), ScrapingConfig)
                ),
                data_sources=DataSourceConfig(
                    **_filter_fields(config_data.get("data_sources", {}), DataSourceConfig)
                ),
                search=SearchConfig(
                    **_filter_fields(search_data, SearchConfig),
                    fuzzy_matching=fuzzy_config,
                    multi_pass=multi_pass_config,
                ),
                logging=LoggingConfig(
                    **_filter_fields(config_data.get("logging", {}), LoggingConfig)
                ),
                ui=UIConfig(**_filter_fields(config_data.get("ui", {}), UIConfig)),
                incremental_mode=config_data.get("incremental_mode", True),
                skip_existing=config_data.get("skip_existing", True),
                dry_run=config_data.get("dry_run", False),
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid configuration values in {config_path}: {e}")

        validate_config(cfg)
        return cfg
    cfg = CollectorConfig()
    validate_config(cfg)
    return cfg


def save_config_template(output_path: str = "config_template.yaml"):
    """Save a template configuration file."""
    config = CollectorConfig()
    config_dict = asdict(config)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    print(f"Configuration template saved to: {output_path}")
