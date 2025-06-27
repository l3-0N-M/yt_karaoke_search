import pytest
import yaml

from collector.config import load_config, save_config_template


def test_load_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
        database:
          path: test.db
          backup_enabled: false
        logging:
          level: DEBUG
        """
    )
    cfg = load_config(str(cfg_path))
    assert cfg.database.path == "test.db"
    assert not cfg.database.backup_enabled
    assert cfg.logging.level == "DEBUG"


def test_save_config_template(tmp_path):
    output = tmp_path / "template.yaml"
    save_config_template(str(output))
    assert output.exists()
    data = yaml.safe_load(output.read_text())
    assert "database" in data and "logging" in data


def test_load_config_ignores_extra_keys(tmp_path):
    cfg_path = tmp_path / "extra.yaml"
    cfg_path.write_text(
        """
        database:
          path: sample.db
          unknown: 123
        """
    )
    cfg = load_config(str(cfg_path))
    assert cfg.database.path == "sample.db"


def test_custom_title_patterns(tmp_path):
    cfg_path = tmp_path / "pattern.yaml"
    cfg_path.write_text(
        """
        search:
          title_patterns:
            - "(.+?) -- (.+)"
        """
    )

    cfg = load_config(str(cfg_path))
    assert cfg.search.title_patterns == ["(.+?) -- (.+)"]


def test_use_multi_strategy_flag(tmp_path):
    cfg_path = tmp_path / "multi.yaml"
    cfg_path.write_text(
        """
        search:
          use_multi_strategy: true
        """
    )

    cfg = load_config(str(cfg_path))
    assert cfg.search.use_multi_strategy is True


def test_validation_error(tmp_path):
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(
        """
        database:
          backup_interval_hours: 0
        """
    )
    with pytest.raises(ValueError):
        load_config(str(cfg_path))


def test_custom_database_pool_settings(tmp_path):
    cfg_path = tmp_path / "pool.yaml"
    cfg_path.write_text(
        """
        database:
          connection_pool_size: 5
          connection_timeout: 5.5
        """
    )

    cfg = load_config(str(cfg_path))
    assert cfg.database.connection_pool_size == 5
    assert cfg.database.connection_timeout == 5.5
