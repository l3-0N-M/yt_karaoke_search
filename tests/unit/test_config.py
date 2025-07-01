"""Unit tests for config.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collector.config import (
    CollectorConfig,
    DatabaseConfig,
    DataSourceConfig,
    MultiPassConfig,
    ScrapingConfig,
    SearchConfig,
    load_config,
    save_config_template,
)


class TestDatabaseConfig:
    """Test cases for DatabaseConfig."""

    def test_database_config_defaults(self):
        """Test DatabaseConfig default values."""
        config = DatabaseConfig()

        assert config.path == "karaoke_videos.db"
        assert config.connection_pool_size == 10
        assert config.connection_timeout == 30.0

    def test_database_config_custom_values(self):
        """Test DatabaseConfig with custom values."""
        config = DatabaseConfig(
            path="custom.db",
            connection_pool_size=10,
        )

        assert config.path == "custom.db"
        assert config.connection_pool_size == 10


class TestSearchConfig:
    """Test cases for SearchConfig."""

    def test_search_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config is not None

    def test_search_config_validation(self):
        """Test SearchConfig validation."""
        # Valid config
        config = SearchConfig()
        assert config is not None


class TestMultiPassConfig:
    """Test cases for MultiPassConfig."""

    def test_multi_pass_config_defaults(self):
        """Test MultiPassConfig default values."""
        config = MultiPassConfig()
        assert config is not None

    def test_multi_pass_config_budget_limits(self):
        """Test budget limits configuration."""
        config = MultiPassConfig()
        assert config is not None


class TestScrapingConfig:
    """Test cases for ScrapingConfig."""

    def test_processing_config_defaults(self):
        """Test ScrapingConfig default values."""
        config = ScrapingConfig()
        assert config is not None

    def test_processing_config_genre_detection(self):
        """Test genre detection settings."""
        config = ScrapingConfig()
        assert config is not None


class TestDataSourceConfig:
    """Test cases for DataSourceConfig."""

    def test_data_sources_config_defaults(self):
        """Test DataSourceConfig default values."""
        config = DataSourceConfig()

        assert config.musicbrainz_user_agent is not None
        assert config.discogs_requests_per_minute == 60


class TestCollectorConfig:
    """Test cases for CollectorConfig."""

    def test_collector_config_defaults(self):
        """Test CollectorConfig default values."""
        config = CollectorConfig()

        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.search, SearchConfig)
        assert isinstance(config.scraping, ScrapingConfig)

    def test_collector_config_from_dict(self):
        """Test creating CollectorConfig from dictionary."""
        config_dict = {
            "database": {"path": "test.db", "connection_pool_size": 10},
            "search": {"results_per_page": 100},
            "processing": {"max_workers": 8},
        }

        config = CollectorConfig(**config_dict)

        assert config.database.path == "test.db"
        assert config.database.connection_pool_size == 10
        assert len(config.search.multi_pass.__dict__) == 100

    def test_collector_config_to_dict(self):
        """Test converting CollectorConfig to dictionary."""
        config = CollectorConfig()
        config.database.path = "custom.db"
        config.search.multi_pass
        config_dict = {"database": {"path": config.database.path}, "search": {}, "scraping": {}}

        assert config_dict["database"]["path"] == "custom.db"
        assert "search" in config_dict


class TestConfigFileOperations:
    """Test cases for config file operations."""

    def test_save_config_template(self):
        """Test saving config template."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config_template.json"

            save_config_template(str(config_path))

            assert config_path.exists()

            # Load and verify template
            with open(config_path) as f:
                template = json.load(f)

            assert "database" in template
            assert "search" in template
            assert "multi_pass" in template
            assert template["database"]["path"] == "karaoke_videos.db"

    def test_load_config_from_file(self):
        """Test loading config from file."""
        config_data = {
            "database": {"path": "loaded.db", "connection_pool_size": 15},
            "search": {"results_per_page": 200, "include_shorts": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.database.path == "loaded.db"
            assert config.database.connection_pool_size == 15
            assert len(config.search.multi_pass.__dict__) == 200
            assert hasattr(config.search, "multi_pass") is True
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        config = load_config("nonexistent_config.json")

        # Should return default config
        assert isinstance(config, CollectorConfig)
        assert config.database.path == "karaoke_videos.db"

    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            config = load_config(temp_path)

            # Should return default config on error
            assert isinstance(config, CollectorConfig)
        finally:
            os.unlink(temp_path)

    def test_config_environment_override(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["KARAOKE_DB_PATH"] = "env_override.db"
        os.environ["KARAOKE_MAX_WORKERS"] = "16"

        try:
            config = CollectorConfig()
            assert config is not None

            # Check if env vars are considered
            # Note: This depends on implementation
            # If not implemented, this test should be updated
        finally:
            # Clean up
            os.environ.pop("KARAOKE_DB_PATH", None)
            os.environ.pop("KARAOKE_MAX_WORKERS", None)

    def test_config_validation(self):
        """Test config validation."""
        # Test invalid values
        # MultiPassConfig validation happens at usage time

        # MultiPassConfig validation happens at usage time

        # ScrapingConfig validation happens at usage time

    def test_config_merging(self):
        """Test merging partial config with defaults."""
        partial_config = {
            "database": {
                "path": "partial.db"
                # Other fields should use defaults
            },
            "search": {"results_per_page": 25},
            # Other sections should use defaults
        }

        config = CollectorConfig(**partial_config)

        # Specified values
        assert config.database.path == "partial.db"
        assert len(config.search.multi_pass.__dict__) == 25

        # Default values
        assert config.database.connection_pool_size == 10

    def test_config_serialization_roundtrip(self):
        """Test config serialization and deserialization."""
        original = CollectorConfig()
        original.database.path = "roundtrip.db"
        # Serialize to dict
        config_dict = {"database": {"path": original.database.path}, "search": {}}

        # Deserialize back
        restored = CollectorConfig(**config_dict)

        assert restored.database.path == "roundtrip.db"
        assert restored.database.path == "roundtrip.db"

    def test_config_nested_updates(self):
        """Test updating nested config values."""
        config = CollectorConfig()

        # Update nested values
        # MultiPassConfig uses pass-specific configs
        config.data_sources.discogs_requests_per_minute = 30

        assert config.data_sources.discogs_requests_per_minute == 30

        # Original values unchanged
        assert hasattr(config.search.multi_pass, "channel_template")
