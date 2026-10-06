import os
import pathlib
import subprocess
import sys

import pytest


def test_demo_script_runs():
    pytest.importorskip("matplotlib")
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "demo.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "All solvers finished." in result.stdout
