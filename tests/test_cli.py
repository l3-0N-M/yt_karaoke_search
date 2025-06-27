from click.testing import CliRunner
from collector import cli


def test_create_config_command(tmp_path):
    runner = CliRunner()
    out = tmp_path / "out.yaml"
    result = runner.invoke(cli.cli, ["create-config", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()

