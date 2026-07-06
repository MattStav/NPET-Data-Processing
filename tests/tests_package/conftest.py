import subprocess
import sys
from subprocess import CompletedProcess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def build_wheel(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CompletedProcess, Path]:
    """Build the wheel and return the result and the build directory."""
    build_dir: Path = tmp_path_factory.mktemp("build")
    result: CompletedProcess = subprocess.run(
        ["uv", "build", "--out-dir", str(build_dir), "--wheel"],
        capture_output=True,
        text=True,
    )
    return result, build_dir


@pytest.fixture(scope="session")
def wheel_path(build_wheel: tuple[CompletedProcess, Path]) -> Path:
    """Get the path to the built wheel."""
    return build_wheel[1].resolve()


@pytest.fixture(scope="session")
def build_result(build_wheel: tuple[CompletedProcess, Path]) -> CompletedProcess:
    """Get the result of the build."""
    return build_wheel[0]


@pytest.fixture(scope="session")
def wheel_file(wheel_path: Path) -> Path:
    """Get the path to the built-wheel file."""
    wheel_files = [*wheel_path.glob("npet_dp-*.whl")]
    assert len(wheel_files) == 1, f"Expected one wheel file, found {len(wheel_files)}"
    return wheel_files[0]


def get_system_specific_pypath(venv: Path) -> Path:
    """Get the system-specific python path."""
    return (
        venv / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv / "bin" / "python"
    )
