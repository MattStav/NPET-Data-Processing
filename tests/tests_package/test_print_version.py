import subprocess
import sys
from importlib.metadata import version
from typing import Literal

import pytest

from NPET_DP.framework.constants import APP_NAME, PACKAGE_NAME


@pytest.mark.parametrize("arg", ("-v", "--version"))
@pytest.mark.flaky(reruns=3, reason="Version may not print correctly when there are uncommitted changes")
def test_version_prints_correctly(arg: Literal["-v", "--version"]) -> None:
    """Test version printing is correct.

    Test that the package can properly print its own version when prompted by arguments.
    This should also print the package name.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "NPET_DP", arg],
        capture_output=True,
        text=True,
        check=True,
    )
    expected: str = f"{APP_NAME} {version(PACKAGE_NAME)}\n"
    assert result.returncode == 0
    assert expected == result.stdout, "Expected version to be printed"
