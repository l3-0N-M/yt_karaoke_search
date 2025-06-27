from click.testing import CliRunner

from collector import cli


class DummyCollector:
    def __init__(self, *a, **k):
        pass

    async def collect_videos(self, queries, max_per_query):
        return 1

    async def get_statistics(self):
        return {}

    async def collect_from_channel(self, url, max_videos, incremental):
        return 1

    async def collect_from_channels(self, urls, max_videos):
        return len(urls)

    async def get_channel_statistics(self):
        return {"channels": [], "total_channels": 0, "total_videos_from_channels": 0}

    async def cleanup(self):
        pass


def test_create_config_command(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out.yaml"
    result = runner.invoke(cli.cli, ["create-config", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_collect_command_event_loop(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli, "KaraokeCollector", lambda cfg: DummyCollector())
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(cli.cli, ["collect", "-q", "test", "--dry-run"])
    assert result.exit_code == 0
    assert "Event loop is closed" not in result.output


def test_collect_command_multi_strategy(monkeypatch):
    runner = CliRunner()
    captured = {}

    def _collector(cfg):
        captured["flag"] = cfg.search.use_multi_strategy
        return DummyCollector()

    monkeypatch.setattr(cli, "KaraokeCollector", _collector)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(cli.cli, ["collect", "-q", "test", "--multi-strategy"])
    assert result.exit_code == 0
    assert captured.get("flag") is True


def test_collect_channel_command_event_loop(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(cli, "KaraokeCollector", lambda cfg: DummyCollector())
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(cli.cli, ["collect-channel", "http://example.com"])
    assert result.exit_code == 0
    assert "Event loop is closed" not in result.output


def test_collect_channel_multi_strategy(monkeypatch):
    runner = CliRunner()
    captured = {}

    def _collector(cfg):
        captured["flag"] = cfg.search.use_multi_strategy
        return DummyCollector()

    monkeypatch.setattr(cli, "KaraokeCollector", _collector)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(
        cli.cli,
        ["collect-channel", "http://example.com", "--multi-strategy"],
    )
    assert result.exit_code == 0
    assert captured.get("flag") is True


def test_collect_channel_multi_pass(monkeypatch):
    runner = CliRunner()
    captured = {}

    def _collector(cfg):
        captured["flag"] = cfg.search.multi_pass.enabled
        return DummyCollector()

    monkeypatch.setattr(cli, "KaraokeCollector", _collector)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(
        cli.cli,
        ["collect-channel", "http://example.com", "--multi-pass"],
    )
    assert result.exit_code == 0
    assert captured.get("flag") is True


def test_collect_channels_command_event_loop(tmp_path, monkeypatch):
    path = tmp_path / "channels.txt"
    path.write_text("http://a\nhttp://b\n")
    runner = CliRunner()
    monkeypatch.setattr(cli, "KaraokeCollector", lambda cfg: DummyCollector())
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)

    result = runner.invoke(cli.cli, ["collect-channels", str(path)])
    assert result.exit_code == 0
    assert "Event loop is closed" not in result.output
