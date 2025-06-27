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
