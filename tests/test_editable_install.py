import subprocess
import sys
from pathlib import Path
import venv


def test_editable_install(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    cmd = [str(python), "-m", "pip", "install", "--no-index", "--no-deps", "-e", str(project_root) + "[dev]"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    check = subprocess.run([str(python), "-c", "import karaoke_video_collector"], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
