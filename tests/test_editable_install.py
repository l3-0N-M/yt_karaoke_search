import importlib.util
import subprocess
import sys
import venv
from pathlib import Path

import pytest


def test_editable_install(tmp_path):
    if importlib.util.find_spec("wheel") is None:
        pytest.skip("wheel package is required for editable install test")
    project_root = Path(__file__).resolve().parents[1]
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
    python = env_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    cmd = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "--no-index",
        "--no-deps",
        "-e",
        str(project_root) + "[dev]",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    check = subprocess.run(
        [str(python), "-c", "import karaoke_video_collector"], capture_output=True, text=True
    )
    assert check.returncode == 0, check.stderr
