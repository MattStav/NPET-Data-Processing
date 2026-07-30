from pathlib import Path
from subprocess import CompletedProcess

import pytest


@pytest.mark.smoke
@pytest.mark.xdist_group(name="package")
def test_package_buildable(build_result: CompletedProcess, wheel_path: Path) -> None:
    """Test that the package is buildable.

    The build process should exit with 0 exit code and
    there should be generated files in the output.
    """
    ret_code: int = build_result.returncode
    assert ret_code == 0, f"Build failed, exit code: {ret_code!s}"
    assert wheel_path.iterdir(), "Build generated no files"


@pytest.mark.xdist_group(name="package")
def test_wheel_file_exists(wheel_path: Path) -> None:
    """Test that the wheel file exists.

    The built files should contain the package .whl file.
    """
    wheel_files = [*wheel_path.glob("npet_dp-*.whl")]
    assert len(wheel_files) == 1, f"Expected one wheel file, found {len(wheel_files)}"
