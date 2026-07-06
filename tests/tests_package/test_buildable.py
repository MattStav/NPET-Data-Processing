from pathlib import Path
from subprocess import CompletedProcess

import pytest


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_package_buildable(build_result: CompletedProcess, wheel_path: Path):
    """Test that the package is buildable."""
    ret_code: int = build_result.returncode
    assert ret_code == 0, f"Build failed, exit code: {str(ret_code)}"
    assert wheel_path.iterdir(), "Build generated no files"


@pytest.mark.xdist_group(name="package")
def test_wheel_file_exists(wheel_path: Path):
    """Test that the wheel file exists."""
    wheel_files = [*wheel_path.glob("npet_dp-*.whl")]
    assert len(wheel_files) == 1, f"Expected one wheel file, found {len(wheel_files)}"
