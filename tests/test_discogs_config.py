"""Tests for Discogs configuration integration."""

import pytest
import yaml

from collector.config import (
    CollectorConfig,
    DataSourceConfig,
    MultiPassConfig,
    load_config,
    validate_config,
)


class TestDiscogsConfiguration:
    """Test Discogs configuration integration."""

    def test_default_discogs_config(self):
        """Test default Discogs configuration values."""
        config = CollectorConfig()
        discogs_config = config.data_sources
        
        assert discogs_config.discogs_enabled is True
        assert discogs_config.discogs_token == ""
        assert discogs_config.discogs_user_agent == "KaraokeCollector/2.1 +https://github.com/karaoke/search"
        assert discogs_config.discogs_timeout == 10
        assert discogs_config.discogs_requests_per_minute == 60
        assert discogs_config.discogs_use_as_fallback is True
        assert discogs_config.discogs_min_musicbrainz_confidence == 0.6
        assert discogs_config.discogs_max_results_per_search == 10
        assert discogs_config.discogs_confidence_threshold == 0.5

    def test_discogs_multi_pass_config(self):
        """Test Discogs multi-pass configuration."""
        config = CollectorConfig()
        multi_pass_config = config.search.multi_pass
        
        assert hasattr(multi_pass_config, 'discogs_search')
        discogs_pass_config = multi_pass_config.discogs_search
        
        assert discogs_pass_config.enabled is True
        assert discogs_pass_config.confidence_threshold == 0.6
        assert discogs_pass_config.timeout_seconds == 20.0
        assert discogs_pass_config.cpu_budget_limit == 3.0
        assert discogs_pass_config.api_budget_limit == 5

    def test_discogs_config_validation_valid(self):
        """Test validation of valid Discogs configuration."""
        config = CollectorConfig()
        
        # Should not raise any exception
        validate_config(config)

    def test_discogs_config_validation_invalid_confidence(self):
        """Test validation of invalid confidence values."""
        config = CollectorConfig()
        
        # Test invalid confidence threshold
        config.data_sources.discogs_confidence_threshold = 1.5
        with pytest.raises(ValueError, match="discogs_confidence_threshold must be between 0 and 1"):
            validate_config(config)
        
        config.data_sources.discogs_confidence_threshold = -0.1
        with pytest.raises(ValueError, match="discogs_confidence_threshold must be between 0 and 1"):
            validate_config(config)

    def test_discogs_config_validation_invalid_musicbrainz_confidence(self):
        """Test validation of invalid MusicBrainz confidence values."""
        config = CollectorConfig()
        
        config.data_sources.discogs_min_musicbrainz_confidence = 1.5
        with pytest.raises(ValueError, match="discogs_min_musicbrainz_confidence must be between 0 and 1"):
            validate_config(config)

    def test_discogs_config_validation_invalid_requests_per_minute(self):
        """Test validation of invalid requests per minute."""
        config = CollectorConfig()
        
        config.data_sources.discogs_requests_per_minute = 0
        with pytest.raises(ValueError, match="discogs_requests_per_minute must be between 1 and 300"):
            validate_config(config)
        
        config.data_sources.discogs_requests_per_minute = 500
        with pytest.raises(ValueError, match="discogs_requests_per_minute must be between 1 and 300"):
            validate_config(config)

    def test_discogs_config_validation_invalid_max_results(self):
        """Test validation of invalid max results."""
        config = CollectorConfig()
        
        config.data_sources.discogs_max_results_per_search = 0
        with pytest.raises(ValueError, match="discogs_max_results_per_search must be between 1 and 100"):
            validate_config(config)
        
        config.data_sources.discogs_max_results_per_search = 200
        with pytest.raises(ValueError, match="discogs_max_results_per_search must be between 1 and 100"):
            validate_config(config)

    def test_discogs_config_validation_invalid_timeout(self):
        """Test validation of invalid timeout."""
        config = CollectorConfig()
        
        config.data_sources.discogs_timeout = 0
        with pytest.raises(ValueError, match="discogs_timeout must be between 1 and 60 seconds"):
            validate_config(config)
        
        config.data_sources.discogs_timeout = 120
        with pytest.raises(ValueError, match="discogs_timeout must be between 1 and 60 seconds"):
            validate_config(config)

    def test_load_discogs_config_from_yaml(self, tmp_path):
        """Test loading Discogs configuration from YAML file."""
        config_content = """
        data_sources:
          discogs_enabled: false
          discogs_token: "test_token_123"
          discogs_timeout: 15
          discogs_requests_per_minute: 30
          discogs_use_as_fallback: false
          discogs_min_musicbrainz_confidence: 0.8
          discogs_max_results_per_search: 5
          discogs_confidence_threshold: 0.7
        
        search:
          multi_pass:
            discogs_search:
              enabled: false
              confidence_threshold: 0.8
              timeout_seconds: 30.0
              cpu_budget_limit: 5.0
              api_budget_limit: 10
        """
        
        config_path = tmp_path / "discogs_config.yaml"
        config_path.write_text(config_content)
        
        config = load_config(str(config_path))
        
        # Test data sources config
        assert config.data_sources.discogs_enabled is False
        assert config.data_sources.discogs_token == "test_token_123"
        assert config.data_sources.discogs_timeout == 15
        assert config.data_sources.discogs_requests_per_minute == 30
        assert config.data_sources.discogs_use_as_fallback is False
        assert config.data_sources.discogs_min_musicbrainz_confidence == 0.8
        assert config.data_sources.discogs_max_results_per_search == 5
        assert config.data_sources.discogs_confidence_threshold == 0.7
        
        # Test multi-pass config
        discogs_pass_config = config.search.multi_pass.discogs_search
        assert discogs_pass_config.enabled is False
        assert discogs_pass_config.confidence_threshold == 0.8
        assert discogs_pass_config.timeout_seconds == 30.0
        assert discogs_pass_config.cpu_budget_limit == 5.0
        assert discogs_pass_config.api_budget_limit == 10

    def test_load_partial_discogs_config(self, tmp_path):
        """Test loading partial Discogs configuration with defaults."""
        config_content = """
        data_sources:
          discogs_enabled: true
          discogs_token: "partial_token"
        """
        
        config_path = tmp_path / "partial_config.yaml"
        config_path.write_text(config_content)
        
        config = load_config(str(config_path))
        
        # Test that specified values are loaded
        assert config.data_sources.discogs_enabled is True
        assert config.data_sources.discogs_token == "partial_token"
        
        # Test that defaults are used for unspecified values
        assert config.data_sources.discogs_timeout == 10  # Default
        assert config.data_sources.discogs_requests_per_minute == 60  # Default
        assert config.data_sources.discogs_confidence_threshold == 0.5  # Default

    def test_discogs_config_yaml_serialization(self, tmp_path):
        """Test that Discogs config can be serialized to YAML."""
        from collector.config import save_config_template
        
        output_path = tmp_path / "template.yaml"
        save_config_template(str(output_path))
        
        # Load the template and check Discogs fields are present
        with open(output_path, 'r') as f:
            template_data = yaml.safe_load(f)
        
        assert "data_sources" in template_data
        data_sources = template_data["data_sources"]
        
        # Check Discogs fields are included
        discogs_fields = [
            "discogs_enabled",
            "discogs_token", 
            "discogs_user_agent",
            "discogs_timeout",
            "discogs_requests_per_minute",
            "discogs_use_as_fallback",
            "discogs_min_musicbrainz_confidence",
            "discogs_max_results_per_search",
            "discogs_confidence_threshold"
        ]
        
        for field in discogs_fields:
            assert field in data_sources
        
        # Check multi-pass config includes Discogs
        assert "search" in template_data
        assert "multi_pass" in template_data["search"]
        assert "discogs_search" in template_data["search"]["multi_pass"]

    def test_discogs_config_edge_values(self):
        """Test Discogs configuration with edge values."""
        config = CollectorConfig()
        
        # Test minimum valid values
        config.data_sources.discogs_confidence_threshold = 0.0
        config.data_sources.discogs_min_musicbrainz_confidence = 0.0
        config.data_sources.discogs_requests_per_minute = 1
        config.data_sources.discogs_max_results_per_search = 1
        config.data_sources.discogs_timeout = 1
        
        # Should not raise exception
        validate_config(config)
        
        # Test maximum valid values
        config.data_sources.discogs_confidence_threshold = 1.0
        config.data_sources.discogs_min_musicbrainz_confidence = 1.0
        config.data_sources.discogs_requests_per_minute = 300
        config.data_sources.discogs_max_results_per_search = 100
        config.data_sources.discogs_timeout = 60
        
        # Should not raise exception
        validate_config(config)

    def test_discogs_config_unknown_fields_ignored(self, tmp_path):
        """Test that unknown Discogs fields are ignored during loading."""
        config_content = """
        data_sources:
          discogs_enabled: true
          discogs_unknown_field: "should_be_ignored"
          discogs_another_unknown: 123
        """
        
        config_path = tmp_path / "unknown_fields.yaml"
        config_path.write_text(config_content)
        
        # Should load without error and ignore unknown fields
        config = load_config(str(config_path))
        assert config.data_sources.discogs_enabled is True
        
        # Unknown fields should not be present
        assert not hasattr(config.data_sources, 'discogs_unknown_field')
        assert not hasattr(config.data_sources, 'discogs_another_unknown')

    def test_discogs_config_boolean_conversion(self, tmp_path):
        """Test boolean field conversion in Discogs config."""
        config_content = """
        data_sources:
          discogs_enabled: "true"  # String that should be converted
          discogs_use_as_fallback: 1  # Number that should be converted
        """
        
        config_path = tmp_path / "boolean_config.yaml"
        config_path.write_text(config_content)
        
        config = load_config(str(config_path))
        
        # Should handle string/number to boolean conversion gracefully
        # Note: YAML parser should handle this automatically
        assert isinstance(config.data_sources.discogs_enabled, bool)
        assert isinstance(config.data_sources.discogs_use_as_fallback, bool)