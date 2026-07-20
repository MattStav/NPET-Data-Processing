import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def uv_tool_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Environment pointing uv's tool install dir/bin dir at isolated tmp dirs."""
    tool_dir: Path = tmp_path_factory.mktemp("uv_tool_dir")
    tool_bin_dir: Path = tmp_path_factory.mktemp("uv_tool_bin_dir")
    env: dict[str, str] = os.environ.copy()
    env["UV_TOOL_DIR"] = str(tool_dir)
    env["UV_TOOL_BIN_DIR"] = str(tool_bin_dir)
    return env


@pytest.fixture(scope="session")
def npet_dp_installed_as_uv_tool(wheel_file: Path, uv_tool_env: dict[str, str]) -> Path:
    """Install the built wheel as uv tool and return the tool's bin dir."""
    result: subprocess.CompletedProcess = subprocess.run(
        ["uv", "tool", "install", "--force", str(wheel_file)],
        capture_output=True,
        text=True,
        env=uv_tool_env,
    )
    assert result.returncode == 0, f"Failed to install tool: {result.stderr}"
    return Path(uv_tool_env["UV_TOOL_BIN_DIR"])


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_npet_dp_command_callable(npet_dp_installed_as_uv_tool: Path) -> None:
    """Test that `npet-dp` launches and can be terminated via the menu."""
    executable: str = "npet-dp.exe" if sys.platform == "win32" else "npet-dp"
    binary: Path = npet_dp_installed_as_uv_tool / executable
    assert binary.is_file(), f"npet-dp executable not found: {binary}"
    result: subprocess.CompletedProcess = subprocess.run(
        [str(binary)],
        input="0\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"npet-dp failed: {result.stderr}"
    assert "terminated" in result.stdout, "Expected app to launch and exit cleanly"


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_uv_tool_run_npet_dp_callable(
    wheel_file: Path,
    uv_tool_env: dict[str, str],
) -> None:
    """Test that `uv tool run NPET_DP` launches and can be terminated via the menu."""
    result: subprocess.CompletedProcess = subprocess.run(
        ["uv", "tool", "run", "--from", str(wheel_file), "npet-dp"],
        input="0\n",
        capture_output=True,
        text=True,
        env=uv_tool_env,
        timeout=60,
    )
    assert result.returncode == 0, f"uv tool run NPET_DP failed: {result.stderr}"
    assert "terminated" in result.stdout, "Expected app to launch and exit cleanly"
