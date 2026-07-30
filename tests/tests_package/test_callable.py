import os
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from NPET_DP.__main__ import npet_dp as main_module_npet_dp
from NPET_DP.main_cli import npet_dp as main_cli_npet_dp


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
    result: CompletedProcess = subprocess.run(  # noqa: S603
        ["uv", "tool", "install", "--force", str(wheel_file)],  # noqa: S607
        capture_output=True,
        text=True,
        env=uv_tool_env,
        check=False,
    )
    assert result.returncode == 0, f"Failed to install tool: {result.stderr}"
    return Path(uv_tool_env["UV_TOOL_BIN_DIR"])


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_npet_dp_command_callable(npet_dp_installed_as_uv_tool: Path) -> None:
    """Test `npet-dp` command.

    Test that `npet-dp` launches the app and can be terminated via the menu.
    """
    executable: str = "npet-dp.exe" if sys.platform == "win32" else "npet-dp"
    binary: Path = npet_dp_installed_as_uv_tool / executable
    assert binary.is_file(), f"npet-dp executable not found: {binary}"
    result: CompletedProcess = subprocess.run(  # noqa: S603
        [str(binary)],
        input="0\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"npet-dp failed: {result.stderr}"
    assert "terminated" in result.stdout, "Expected app to launch and exit cleanly"


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_uv_tool_run_npet_dp_callable(
    wheel_file: Path,
    uv_tool_env: dict[str, str],
) -> None:
    """Test launching as python package.

    Test that `uv tool run NPET_DP` launches the app and can be terminated via the menu.
    """
    result: CompletedProcess = subprocess.run(  # noqa: S603
        ["uv", "tool", "run", "--from", str(wheel_file), "npet-dp"],  # noqa: S607
        input="0\n",
        capture_output=True,
        text=True,
        env=uv_tool_env,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"uv tool run NPET_DP failed: {result.stderr}"
    assert "terminated" in result.stdout, "Expected app to launch and exit cleanly"


@pytest.mark.smoke
def test_python_dash_m_launches_app() -> None:
    """Test that `python -m NPET_DP` launches the app and can be terminated via the menu.

    Failing this test means the `if __name__ == "__main__":` guard
    no longer calls npet_dp() when the module is run directly.
    """
    result: CompletedProcess = subprocess.run(
        [sys.executable, "-m", "NPET_DP"],
        input="0\n",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"python -m NPET_DP failed: {result.stderr}"
    assert "terminated" in result.stdout, "Expected app to launch and exit cleanly"


def test_main_exposes_main_cli_npet_dp() -> None:
    """Test that __main__ exposes the same npet_dp used by the `npet-dp` entry point.

    The `npet-dp` console script points at `NPET_DP.__main__:npet_dp`, so this name
    must resolve to the actual CLI app, or the installed command would break.
    """
    assert main_module_npet_dp is main_cli_npet_dp
