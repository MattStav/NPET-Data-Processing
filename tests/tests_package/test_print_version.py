import subprocess
import sys
from importlib.metadata import version

import pytest

from NPET_DP.framework.constants import APP_NAME, PACKAGE_NAME


@pytest.mark.parametrize("arg", ("-v", "--version"))
def test_version_prints_correctly(arg) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "NPET_DP", arg],
        capture_output=True,
        text=True,
        check=True,
    )
    expected: str = f"{APP_NAME} {version(PACKAGE_NAME)}\n"
    assert result.returncode == 0
    assert expected == result.stdout, "Expected version to be printed"
